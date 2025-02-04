from .custom_nodes.image_selector import ImageSelector
from .custom_nodes.simple_image_loader import SimpleImageLoader

NODE_CLASS_MAPPINGS = {
    "ImageSelector": ImageSelector,
    "SimpleImageLoader": SimpleImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSelector": "Image Selector",
    "SimpleImageLoader": "Simple Image Loader",
}

WEB_DIRECTORY = "./web/extensions"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']