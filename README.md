# ComfyUI Custom Nodes

A collection of custom nodes for ComfyUI, focusing on image handling and LoRA training.

## Nodes

### Image Switch Select
A node that allows selecting between multiple loaded images with a grid preview interface.
- Located in: `nodes/image_switch_select.py`
- UI Component: `web/extensions/image_switch_select.js`
- Features:
  - Grid preview of loaded images
  - Click-to-select functionality
  - Maintains selection state
  - Proper scaling and positioning

### LoRA Training
A script for training LoRA adapters for Stable Diffusion models.
- Located in: `train_lora_detailed.py`
- Features:
  - Mixed precision training (fp16)
  - Validation split
  - Checkpoint saving
  - Progress tracking
  - Memory efficient with 8-bit optimizers
  - Detailed technical documentation

## Installation

1. Clone this repository into your ComfyUI custom_nodes directory
2. Restart ComfyUI
3. The nodes will be available in the node menu under "mickster_nodes"

## Dependencies

- torch
- transformers
- diffusers
- accelerate
- bitsandbytes
- peft
- tqdm

## Usage

### Image Switch Select
1. Add the node to your workflow
2. Upload images using the upload buttons
3. Click on image cells to select which image to output

### LoRA Training
```py
train_lora(
    image_dir="path/to/images",
    output_dir="path/to/save",
    instance_prompt="your prompt",
    num_epochs=100
)
```