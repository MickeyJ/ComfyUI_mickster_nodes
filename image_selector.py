from easy_nodes import ComfyNode, ImageTensor, NumberInput, show_image
import folder_paths
from PIL import Image
import torch
import numpy as np
import os
import hashlib

@ComfyNode(
    category="mickster_nodes",
    display_name="Image Selector"
)
def image_selector(
    image1: str = ("FILE", {"filetypes": ["image/*"]}),
    image2: str = ("FILE", {"filetypes": ["image/*"]}),
    image3: str = ("FILE", {"filetypes": ["image/*"]}),
    image4: str = ("FILE", {"filetypes": ["image/*"]}),
    image5: str = ("FILE", {"filetypes": ["image/*"]}),
    image6: str = ("FILE", {"filetypes": ["image/*"]}),
    image7: str = ("FILE", {"filetypes": ["image/*"]}),
    image8: str = ("FILE", {"filetypes": ["image/*"]}),
    image9: str = ("FILE", {"filetypes": ["image/*"]}),
    selected_index: int = NumberInput(0, 0, 8)
) -> tuple[ImageTensor, ImageTensor]:
    """Select from multiple images with grid preview"""
    image_list = []
    
    # Convert all input images to tensors
    for i in range(1, 10):
        image_name = f"image{i}"
        image_path = locals()[image_name]
        if image_path:
            image_path = folder_paths.get_annotated_filepath(image_path)
            i = Image.open(image_path)
            i = i.convert('RGB')
            image = np.array(i).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            image_list.append(image)

    if not image_list:
        empty = torch.zeros((1, 64, 64, 3))
        return empty, empty

    # Create grid preview
    grid = create_grid(image_list)
    
    # Show preview in the node
    show_image(grid)
    
    # Return selected image and grid
    selected_index = min(selected_index, len(image_list) - 1)
    return image_list[selected_index], grid

def create_grid(images):
    if not images:
        return torch.zeros((1, 64, 64, 3))
        
    cell_height = 200
    aspect_ratio = images[0].shape[2] / images[0].shape[1]
    cell_width = int(cell_height * aspect_ratio)
    
    grid = torch.zeros((1, cell_height * 3, cell_width * 3, 3))
    
    for idx, img in enumerate(images):
        if idx >= 9:
            break
            
        row = idx // 3
        col = idx % 3
        
        img_pil = Image.fromarray((img[0].cpu().numpy() * 255).astype(np.uint8))
        img_pil = img_pil.resize((cell_width, cell_height))
        img_resized = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)
        
        y_start = row * cell_height
        x_start = col * cell_width
        grid[0, y_start:y_start + cell_height, x_start:x_start + cell_width, :] = img_resized

    return grid

def is_changed(self, **kwargs):
    """Return a value that changes when the output should change"""
    image_paths = []
    for i in range(1, 10):
        image_name = f"image{i}"
        if image_name in kwargs and kwargs[image_name]:
            image_path = folder_paths.get_annotated_filepath(kwargs[image_name])
            image_paths.append(image_path)
    
    m = hashlib.sha256()
    for path in image_paths:
        with open(path, 'rb') as f:
            m.update(f.read())
    return m.digest().hex()

@classmethod
def VALIDATE_INPUTS(s, **kwargs):
    for i in range(1, 10):
        image_name = f"image{i}"
        if image_name in kwargs and kwargs[image_name]:
            if not folder_paths.exists_annotated_filepath(kwargs[image_name]):
                return f"Invalid image file: {kwargs[image_name]}"
    return True