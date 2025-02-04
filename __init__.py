import os
import easy_nodes

# Initialize easy_nodes first
easy_nodes.initialize_easy_nodes(default_category="mickster_nodes", auto_register=False)

# Import nodes after initialization
from .image_selector import image_selector
from .simple_image_loader import simple_image_loader

# Get node mappings from easy_nodes
NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = easy_nodes.get_node_mappings()

# Debug print
print("\nRegistered nodes:", NODE_CLASS_MAPPINGS.keys())
print("Display names:", NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web/extensions"

# Export for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

# Save node list for ComfyUI-Manager
easy_nodes.save_node_list(os.path.join(os.path.dirname(__file__), "node_list.json"))