from rest_framework import serializers

from .models import Dish, DishIngredient, Ingredient


class IngredientSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ingredient
        fields = ["id", "name", "description"]


class DishIngredientSerializer(serializers.ModelSerializer):
    ingredient = IngredientSerializer(read_only=True)

    class Meta:
        model = DishIngredient
        fields = ["ingredient", "is_optional", "notes", "source_url"]


class DishSerializer(serializers.ModelSerializer):
    ingredients = DishIngredientSerializer(
        source="dish_ingredients",
        many=True,
        read_only=True,
    )

    class Meta:
        model = Dish
        fields = ["id", "name", "slug", "description", "region", "emoji", "ingredients"]
