from django import forms
from django.core.exceptions import ValidationError


MAX_IMAGE_SIZE = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000


class DishImageForm(forms.Form):
    image = forms.ImageField(
        label="Fotografía del platillo",
        help_text="JPG, PNG o WebP · máximo 8 MB.",
        widget=forms.FileInput(
            attrs={
                "accept": "image/jpeg,image/png,image/webp",
                "class": "upload-input",
            }
        ),
    )

    def clean_image(self):
        image = self.cleaned_data["image"]
        if image.size > MAX_IMAGE_SIZE:
            raise ValidationError("La imagen no debe superar los 8 MB.")
        if image.image.width * image.image.height > MAX_IMAGE_PIXELS:
            raise ValidationError(
                "La imagen tiene demasiados píxeles. Prueba con una versión más pequeña."
            )
        return image
