from django.urls import path

from . import pages

app_name = "catalog_pages"

urlpatterns = [
    path("", pages.home, name="home"),
    path("predecir/", pages.classify_dish, name="classify_dish"),
    path("platillo/<slug:slug>/", pages.dish_detail, name="dish_detail"),
]
