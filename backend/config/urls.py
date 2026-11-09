from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("catalog.pages_urls")),
    path("api/", include("catalog.urls")),
]
