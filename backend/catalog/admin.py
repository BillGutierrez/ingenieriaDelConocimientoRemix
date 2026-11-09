from django.contrib import admin

from .models import Dish, DishIngredient, Ingredient


class DishIngredientInline(admin.TabularInline):
    model = DishIngredient
    extra = 1
    autocomplete_fields = ["ingredient"]


@admin.register(Dish)
class DishAdmin(admin.ModelAdmin):
    list_display = ["name", "region", "slug"]
    list_filter = ["region"]
    search_fields = ["name", "slug", "description"]
    prepopulated_fields = {"slug": ["name"]}
    inlines = [DishIngredientInline]


@admin.register(Ingredient)
class IngredientAdmin(admin.ModelAdmin):
    search_fields = ["name"]
