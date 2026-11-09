from io import BytesIO
from unittest.mock import patch

import numpy as np
from PIL import Image
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.core.management import call_command
from rest_framework.test import APIClient

from .inference import CLASS_NAMES, DishPrediction, predict_dish
from .management.commands.seed_dishes import RECIPES
from .models import Dish, DishIngredient, Ingredient


def png_upload(name="plato.png"):
    content = BytesIO()
    Image.new("RGB", (32, 24), color=(120, 80, 40)).save(content, format="PNG")
    return SimpleUploadedFile(name, content.getvalue(), content_type="image/png")


class ImagePredictionTests(TestCase):
    @patch("catalog.inference._load_model")
    def test_prediction_prepares_image_and_selects_highest_scoring_class(self, load_model):
        class FakeModel:
            def get_input_details(self):
                return [{"index": 1}]

            def get_output_details(self):
                return [{"index": 2}]

            def set_tensor(self, index, batch):
                self.batch = batch
                self.input_index = index

            def invoke(self):
                scores = np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)
                scores[0, 12] = 0.9
                self.scores = scores

            def get_tensor(self, index):
                self.output_index = index
                return self.scores

        model = FakeModel()
        load_model.return_value = model

        prediction = predict_dish(png_upload())

        self.assertEqual(prediction.dish_name, "Ceviche")
        self.assertAlmostEqual(prediction.confidence, 0.9, places=6)
        self.assertEqual(model.batch.shape, (1, 224, 224, 3))
        self.assertGreaterEqual(float(model.batch.min()), 0)
        self.assertLessEqual(float(model.batch.max()), 1)
        self.assertEqual(model.input_index, 1)
        self.assertEqual(model.output_index, 2)


class CatalogApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_health_reports_database_status(self):
        response = self.client.get("/api/health/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "database": "ok"})

    def test_dish_list_includes_typical_ingredients_and_sources(self):
        dish = Dish.objects.create(
            name="Ceviche",
            slug="ceviche",
            description="Platillo tradicional peruano.",
        )
        ingredient = Ingredient.objects.create(name="Pescado")
        DishIngredient.objects.create(
            dish=dish,
            ingredient=ingredient,
            notes="Puede variar según la receta.",
            source_url="https://example.com/receta",
        )

        response = self.client.get("/api/dishes/")

        self.assertEqual(response.status_code, 200)
        item = response.json()["results"][0]
        self.assertEqual(item["slug"], "ceviche")
        self.assertEqual(item["ingredients"][0]["ingredient"]["name"], "Pescado")
        self.assertEqual(
            item["ingredients"][0]["source_url"],
            "https://example.com/receta",
        )

    def test_dish_detail_uses_slug(self):
        Dish.objects.create(
            name="Causa limeña",
            slug="causa-limena",
            description="Platillo tradicional peruano.",
        )

        response = self.client.get("/api/dishes/causa-limena/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "Causa limeña")

    def test_unknown_dish_returns_not_found(self):
        response = self.client.get("/api/dishes/no-existe/")

        self.assertEqual(response.status_code, 404)


@override_settings(
    STORAGES={
        "default": {
            "BACKEND": "django.core.files.storage.FileSystemStorage",
        },
        "staticfiles": {
            "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
        },
    }
)
class PublicCatalogPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("seed_dishes", verbosity=0)

    def test_homepage_renders_seeded_catalog(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Sabores del Perú")
        self.assertContains(response, "Identificar un platillo")
        self.assertContains(response, "Ceviche")
        self.assertContains(response, "COSTA · SIERRA · SELVA")
        self.assertContains(response, "no representa por igual")
        self.assertContains(response, 'data-clicks-needed="7"')
        self.assertContains(response, "Te amo &lt;# D")
        self.assertEqual(response.context["dish_count"], 40)

    def test_model_class_order_matches_catalog(self):
        catalog_names = list(Dish.objects.values_list("name", flat=True).order_by("name"))

        self.assertEqual(list(CLASS_NAMES), catalog_names)

    def test_homepage_searches_by_ingredient(self):
        response = self.client.get("/", {"q": "huacatay"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ocopa")
        self.assertNotContains(response, "Ceviche")

    def test_dish_detail_shows_ingredients_and_reference(self):
        response = self.client.get("/platillo/ocopa/")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Ingredientes habituales")
        self.assertContains(response, "Huacatay")
        self.assertContains(response, "consultar fuente")

    @patch("catalog.pages.predict_dish", return_value=DishPrediction("Ceviche", 0.94))
    def test_uploaded_image_redirects_to_predicted_dish(self, predict):
        response = self.client.post("/predecir/", {"image": png_upload()})

        self.assertRedirects(
            response,
            "/platillo/ceviche/",
            fetch_redirect_response=False,
        )
        predict.assert_called_once()

        detail = self.client.get(response.url)

        self.assertContains(detail, "RESULTADO DE TU FOTO")
        self.assertContains(detail, "94%")

    @patch("catalog.pages.predict_dish")
    def test_invalid_upload_is_rejected_without_running_model(self, predict):
        upload = SimpleUploadedFile(
            "invalida.png",
            b"no es una imagen",
            content_type="image/png",
        )
        response = self.client.post("/predecir/", {"image": upload})

        self.assertEqual(response.status_code, 400)
        self.assertIn("image", response.context["image_form"].errors)
        predict.assert_not_called()

    def test_seed_command_is_idempotent(self):
        call_command("seed_dishes", verbosity=0)

        self.assertEqual(Dish.objects.count(), 40)

    def test_seeded_regions_use_the_current_three_regions_or_diverse(self):
        valid_regions = {value for value, _ in Dish.Region.choices}

        self.assertEqual(
            valid_regions,
            {"costa", "sierra", "selva", "diversas"},
        )
        self.assertTrue(
            all(recipe[1] in valid_regions for recipe in RECIPES.values())
        )
        self.assertFalse(
            Dish.objects.filter(region__in=["andes", "lima_chifa"]).exists()
        )

    def test_seed_marks_alternative_ingredients_as_optional(self):
        for dish_name, ingredient_name in (
            ("Escabeche", "pollo"),
            ("Escabeche", "pescado"),
            ("Olluquito", "carne de res"),
            ("Olluquito", "charqui"),
            ("Tallarin Saltado", "carne de res"),
            ("Tallarin Saltado", "pollo"),
            ("Tacu Tacu", "bistec"),
            ("Tallarines Rojos", "pollo"),
            ("Tallarines Verdes", "bistec"),
            ("Humitas", "anís"),
        ):
            relation = DishIngredient.objects.get(
                dish__name=dish_name,
                ingredient__name=ingredient_name,
            )
            self.assertTrue(relation.is_optional, (dish_name, ingredient_name))
            self.assertTrue(relation.notes)

    def test_description_ingredients_are_present_in_the_seed(self):
        self.assertTrue(
            DishIngredient.objects.filter(
                dish__name="Caldo de Gallina",
                ingredient__name="limón",
            ).exists()
        )
