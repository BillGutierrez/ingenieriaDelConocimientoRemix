from django.db import models


class Dish(models.Model):
    class Region(models.TextChoices):
        COSTA = "costa", "Costa"
        SIERRA = "sierra", "Sierra"
        SELVA = "selva", "Selva"
        DIVERSAS = "diversas", "Varias regiones"

    name = models.CharField("nombre", max_length=120, unique=True)
    slug = models.SlugField("identificador", max_length=140, unique=True)
    description = models.TextField("descripción")
    region = models.CharField(
        "región o tradición",
        max_length=20,
        choices=Region.choices,
        default=Region.DIVERSAS,
    )
    emoji = models.CharField("ícono", max_length=8, default="🍽️")
    ingredients = models.ManyToManyField(
        "Ingredient",
        through="DishIngredient",
        related_name="dishes",
        verbose_name="ingredientes",
        blank=True,
    )

    class Meta:
        ordering = ["name"]
        verbose_name = "platillo"
        verbose_name_plural = "platillos"

    def __str__(self):
        return self.name


class Ingredient(models.Model):
    name = models.CharField("nombre", max_length=120, unique=True)
    description = models.CharField("descripción", max_length=250, blank=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "ingrediente"
        verbose_name_plural = "ingredientes"

    def __str__(self):
        return self.name


class DishIngredient(models.Model):
    dish = models.ForeignKey(
        Dish,
        on_delete=models.CASCADE,
        related_name="dish_ingredients",
        verbose_name="platillo",
    )
    ingredient = models.ForeignKey(
        Ingredient,
        on_delete=models.CASCADE,
        related_name="dish_ingredients",
        verbose_name="ingrediente",
    )
    is_optional = models.BooleanField("opcional", default=False)
    notes = models.CharField("notas", max_length=250, blank=True)
    source_url = models.URLField("fuente de referencia", blank=True)

    class Meta:
        ordering = ["ingredient__name"]
        constraints = [
            models.UniqueConstraint(
                fields=["dish", "ingredient"],
                name="unique_ingredient_per_dish",
            )
        ]
        verbose_name = "ingrediente del platillo"
        verbose_name_plural = "ingredientes del platillo"

    def __str__(self):
        return f"{self.ingredient} en {self.dish}"
