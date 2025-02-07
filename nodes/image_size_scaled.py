from easy_nodes import ComfyNode, NumberInput


@ComfyNode(
    category="mickster_nodes", 
    display_name="Image Size Scaled",
    return_names=["Scaled Width", "Scaled Height", "Scale"],
)
def image_size_scaled(
    width: int = NumberInput(100, 1, 1000, 1, display="Width"),  # Default width
    height: int = NumberInput(100, 1, 1000, 1, display="Height"),  # Default height
    scale: float = NumberInput(1.0, 0.1, 4.0, 0.1, display="Scale"),  # Default scale
) -> tuple[int, int, float]:
    """Outputs scaled width and height based on the scale input."""

    # Calculate scaled dimensions
    scaled_width = int(width * scale)
    scaled_height = int(height * scale)

    return scaled_width, scaled_height, scale

