from easy_nodes import ComfyNode, ImageTensor, NumberInput, StringInput
import folder_paths
from PIL import Image
import torch
import numpy as np
import os

@ComfyNode(
    category="mickster_nodes",
    display_name="ImageSwitchSelect"
)
def image_switch_select(
    image: str = StringInput("", hidden=True),
    selected_index: int = NumberInput(0, 0, 5, 1, hidden=True)
) -> ImageTensor:
    """Load a single image with preview"""
    if not image:
        return torch.zeros((1, 64, 64, 3))

    image_path = folder_paths.get_annotated_filepath(image)
    i = Image.open(image_path)
    i = i.convert('RGB')
    image_tensor = np.array(i).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_tensor)[None,]

    # Store both the image path and selected index
    setattr(image_switch_select, "last_image", image)
    setattr(image_switch_select, "selected_index", selected_index)

    return image_tensor

def is_changed(self, **kwargs):
    """Return a value that changes when the output should change"""
    if not kwargs.get('image'):
        return None

    image_path = folder_paths.get_annotated_filepath(kwargs['image'])
    return os.path.getmtime(image_path)

# Update UI output to include selected index
image_switch_select.ui = {
    "image": getattr(image_switch_select, "last_image", None),
    "selected_index": getattr(image_switch_select, "selected_index", 0),
}

def run(self, **kwargs):
    # Get the selected index and image path
    selected_index = kwargs.get("selected_index", 0)
    image_path = kwargs.get("image", "")
    
    print(f"Python run: selected_index={selected_index}, image_path={image_path}")  # Debug
    
    if not image_path:
        return { "image": None }
        
    # Load and return the selected image
    image = Image.open(image_path)
    return { "image": image }
