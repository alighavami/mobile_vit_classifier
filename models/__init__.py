# models/__init__.py
from .mobilevit import MobileViTClassifier, MobileViT, MobileViTBlock

def build_model(image_size=(224, 224), num_classes=1000, variant="small", **kwargs):
    """
    Simple factory wrapper. You can extend `variant` later to switch dims/channels.
    """
    return MobileViTClassifier(image_size=image_size, num_classes=num_classes, **kwargs)

__all__ = [
    "MobileViTClassifier",
    "MobileViT",
    "MobileViTBlock",
    "build_model",
]
