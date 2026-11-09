import logging

from django.contrib import messages
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse

from .forms import DishImageForm
from .inference import PredictionError, predict_dish
from .models import Dish


logger = logging.getLogger(__name__)


def home(request):
    return render(request, "catalog/home.html", _home_context(request))


def _home_context(request, image_form=None):
    query = request.GET.get("q", "").strip()
    dishes = Dish.objects.annotate(ingredient_count=Count("dish_ingredients"))
    if query:
        dishes = dishes.filter(
            Q(name__icontains=query)
            | Q(description__icontains=query)
            | Q(dish_ingredients__ingredient__name__icontains=query)
        ).distinct()

    return {
        "dishes": dishes,
        "query": query,
        "dish_count": Dish.objects.count(),
        "ingredient_count": Dish.objects.aggregate(
            count=Count("dish_ingredients__ingredient", distinct=True)
        )["count"],
        "regions": Dish.Region.choices,
        "image_form": image_form if image_form is not None else DishImageForm(),
    }


def classify_dish(request):
    if request.method != "POST":
        return redirect("catalog_pages:home")

    form = DishImageForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(
            request,
            "catalog/home.html",
            _home_context(request, image_form=form),
            status=400,
        )

    try:
        prediction = predict_dish(form.cleaned_data["image"])
    except PredictionError:
        logger.exception("No se pudo completar la predicción del platillo.")
        messages.error(
            request,
            "No pudimos analizar la imagen. Inténtalo más tarde o contacta con quien "
            "administra el sitio.",
        )
        return redirect(f"{reverse('catalog_pages:home')}#identificar")

    dish = Dish.objects.filter(name=prediction.dish_name).first()
    if dish is None:
        logger.error(
            "La clase predicha %r no tiene una ficha en el catálogo.",
            prediction.dish_name,
        )
        messages.error(
            request,
            "El platillo reconocido todavía no está disponible en el catálogo.",
        )
        return redirect(f"{reverse('catalog_pages:home')}#identificar")

    request.session["dish_prediction"] = {
        "slug": dish.slug,
        "confidence": round(prediction.confidence * 100, 1),
    }
    return redirect("catalog_pages:dish_detail", slug=dish.slug)


def dish_detail(request, slug):
    dish = get_object_or_404(
        Dish.objects.prefetch_related("dish_ingredients__ingredient"),
        slug=slug,
    )
    reference_url = next(
        (
            item.source_url
            for item in dish.dish_ingredients.all()
            if item.source_url
        ),
        "",
    )
    prediction = request.session.pop("dish_prediction", None)
    if not isinstance(prediction, dict) or prediction.get("slug") != dish.slug:
        prediction = None
    return render(
        request,
        "catalog/dish_detail.html",
        {
            "dish": dish,
            "reference_url": reference_url,
            "prediction": prediction,
        },
    )
