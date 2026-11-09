from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import DishViewSet, health

router = DefaultRouter()
router.register("dishes", DishViewSet, basename="dish")

urlpatterns = [
    path("health/", health, name="health"),
    path("", include(router.urls)),
]
