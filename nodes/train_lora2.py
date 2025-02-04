# Core Python libraries
import os  # File and path operations
import torch  # Main PyTorch library for tensor operations and neural networks

# PyTorch data handling and processing
from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split,
)  # Dataset management and loading
from torchvision import transforms  # Image transformation operations
from PIL import Image  # Image loading and basic operations

# Stable Diffusion and model components
from diffusers import (
    StableDiffusionPipeline,  # Main pipeline for Stable Diffusion
    AutoencoderKL,  # VAE for image encoding/decoding
    UNet2DConditionModel,  # U-Net model for noise prediction
)
from transformers import (
    CLIPTextModel,  # CLIP text encoder
    CLIPTokenizer,  # CLIP text tokenizer
)

# Training utilities
from accelerate import Accelerator  # Handles distributed training and mixed precision
from tqdm.auto import tqdm  # Progress bar for training loops
import bitsandbytes as bnb  # Memory-efficient optimizers
from peft import LoraConfig, get_peft_model  # LoRA adaptation tools


class PersonDataset(Dataset):
    """Dataset class that handles loading and processing of training images"""

    def __init__(self, image_dir, instance_prompt, tokenizer, size=512):
        # Store initialization parameters
        self.image_dir = image_dir  # Root directory containing training images
        self.instance_prompt = (
            instance_prompt  # Fallback text prompt if no text file exists
        )
        self.tokenizer = tokenizer  # CLIP tokenizer for text processing
        self.size = (
            size  # Target size for image processing (512x512 is standard for SD)
        )

        # Get all valid image paths - supports PNG, JPG, JPEG
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

        # Get corresponding text files (same name but .txt extension)
        self.text_paths = [p.rsplit(".", 1)[0] + ".txt" for p in self.image_paths]

        # Define image preprocessing pipeline
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),  # Scale image to target size
                transforms.CenterCrop(size),  # Crop to square from center
                transforms.ToTensor(),  # Convert PIL image to PyTorch tensor (0-1 range)
                transforms.Normalize(
                    [0.5], [0.5]
                ),  # Scale to [-1, 1] range expected by SD
            ]
        )

    def __len__(self):
        """Return dataset size - required by PyTorch Dataset class"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load and process a single image-text pair - required by PyTorch Dataset"""
        # Load image from disk and convert to RGB
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        # Try to load specific prompt from text file, fall back to instance_prompt
        text_path = self.text_paths[idx]
        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                prompt = f.read().strip()
        else:
            prompt = self.instance_prompt

        # Convert text prompt to token IDs
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",  # Add padding tokens to reach max length
            max_length=self.tokenizer.model_max_length,  # Usually 77 for CLIP
            truncation=True,  # Cut off if too long
            return_tensors="pt",  # Return PyTorch tensors
        )

        # Return processed image and text tokens
        return {
            "pixel_values": image,  # Shape: [3, size, size]
            "input_ids": text_inputs.input_ids[0],  # Shape: [max_length]
        }


def train_lora(
    image_dir: str,  # Path to directory containing training images
    output_dir: str,  # Path to save model checkpoints and final weights
    instance_prompt: str = "photo of sks person",  # Text prompt for training
    batch_size: int = 1,  # Images per batch (limited by GPU memory)
    num_epochs: int = 100,  # Number of complete passes through dataset
    learning_rate: float = 1e-4,  # Step size for gradient updates
    rank: int = 4,  # Rank of LoRA update matrices (higher = more capacity)
    model_id: str = "runwayml/stable-diffusion-v1-5",  # Base model to fine-tune
    validation_split: float = 0.1,  # Fraction of data to use for validation
    checkpoint_freq: int = 10,  # Save checkpoint every N epochs
):
    """Train a LoRA adapter for Stable Diffusion fine-tuning"""

    # Setup mixed precision training for memory efficiency
    accelerator = Accelerator(
        gradient_accumulation_steps=1,  # Update weights after every batch
        mixed_precision="fp16",  # Use 16-bit floating point where possible
    )

    # Load model components from HuggingFace hub
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Freeze VAE and text encoder parameters - only train U-Net
    vae.requires_grad_(False)  # VAE converts images to/from latent space
    text_encoder.requires_grad_(False)  # Text encoder creates prompt embeddings

    # Configure LoRA adaptation
    lora_config = LoraConfig(
        r=rank,  # Rank of update matrices
        lora_alpha=rank,  # Scaling factor for updates
        target_modules=["q_proj", "v_proj"],  # Which attention layers to adapt
        lora_dropout=0.1,  # Dropout probability for regularization
    )

    # Apply LoRA to U-Net model
    unet = get_peft_model(unet, lora_config)  # Wraps model with LoRA layers

    # Create and split dataset
    dataset = PersonDataset(image_dir, instance_prompt, tokenizer)
    val_size = int(len(dataset) * validation_split)  # Calculate validation set size
    train_size = len(dataset) - val_size  # Remaining data for training
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize optimizer with 8-bit precision for memory efficiency
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),  # Only optimize U-Net parameters
        lr=learning_rate,  # Learning rate for gradient updates
        betas=(0.9, 0.999),  # Exponential moving average factors for gradients
        weight_decay=1e-2,  # L2 regularization factor
    )

    # Prepare for distributed training
    unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader
    )

    # Track best model for saving
    best_val_loss = float("inf")  # Initialize to infinity

    # Main training loop
    for epoch in range(num_epochs):
        # Training phase
        unet.train()  # Set model to training mode (enables dropout, etc.)
        train_loss = 0  # Accumulator for average loss
        train_progress = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")

        for batch in train_progress:
            with accelerator.accumulate(unet):  # Handle gradient accumulation
                # Move image data to GPU and convert to float16 precision for efficiency
                pixel_values = batch["pixel_values"].to(
                    dtype=torch.float16
                )  # Shape: [batch_size, 3, 512, 512]

                # Get text tokens but keep on CPU since text encoder handles device movement
                input_ids = batch[
                    "input_ids"
                ]  # Shape: [batch_size, 77] (77 is CLIP's max tokens)

                # Pass text tokens through CLIP text encoder to get embeddings
                # Returns tuple (hidden_states, pooled_output) - we want hidden_states
                encoder_hidden_states = text_encoder(input_ids)[
                    0
                ]  # Shape: [batch_size, 77, 768]

                # Use VAE to encode images into latent space representation
                # sample() gets random point from latent distribution (uses reparameterization trick)
                latents = vae.encode(
                    pixel_values
                ).latent_dist.sample()  # Shape: [batch_size, 4, 64, 64]

                # Scale latents by magic number from Stable Diffusion training
                latents = latents * 0.18215  # Scaling factor for numerical stability

                # Generate random noise for denoising training
                noise = torch.randn_like(latents)  # Random noise same shape as latents
                timesteps = torch.randint(
                    0, 1000, (latents.shape[0],)
                )  # Random timesteps
                noisy_latents = (
                    noise + timesteps.reshape(-1, 1, 1, 1) * latents
                )  # Add noise to latents

                # Predict noise using U-Net
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Calculate mean squared error between predicted and actual noise
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # Backpropagate and optimize
                accelerator.backward(loss)  # Compute gradients
                optimizer.step()  # Update weights using gradients
                optimizer.zero_grad()  # Clear gradients for next batch

            # Update progress bar with current loss
            train_loss += loss.item()  # Accumulate batch loss
            train_progress.set_postfix({"loss": loss.item()})

        # Calculate average training loss for epoch
        avg_train_loss = train_loss / len(train_dataloader)

        # Validation phase
        unet.eval()  # Set model to evaluation mode (disable dropout, etc.)
        val_loss = 0
        val_progress = tqdm(val_dataloader, desc=f"Validation Epoch {epoch+1}")

        with torch.no_grad():  # Disable gradient computation for validation
            for batch in val_progress:
                # Move image data to GPU and convert to float16 precision for efficiency
                pixel_values = batch["pixel_values"].to(
                    dtype=torch.float16
                )  # Shape: [batch_size, 3, 512, 512]
                # What's happening:
                # 1. batch["pixel_values"] starts on CPU
                # 2. .to(dtype=torch.float16) does two things:
                #    - Converts to float16 precision (saves memory)
                #    - Automatically moves to GPU if accelerator is using CUDA
                #    This is because accelerator was initialized with mixed_precision="fp16"
                # Get text tokens but keep on CPU since text encoder handles device movement
                input_ids = batch[
                    "input_ids"
                ]  # Shape: [batch_size, 77] (77 is CLIP's max tokens)
                # What's happening:
                # 1. Stays on CPU intentionally
                # 2. No .to() call because:
                #    - text_encoder will handle moving it to the right device
                #    - token IDs are integers, so no need for float16 conversion

                # Pass text tokens through CLIP text encoder to get embeddings
                # Returns tuple (hidden_states, pooled_output) - we want hidden_states
                encoder_hidden_states = text_encoder(input_ids)[
                    0
                ]  # Shape: [batch_size, 77, 768]

                # Use VAE to encode images into latent space representation
                # sample() gets random point from latent distribution (uses reparameterization trick)
                latents = vae.encode(
                    pixel_values
                ).latent_dist.sample()  # Shape: [batch_size, 4, 64, 64]

                # Scale latents by magic number from Stable Diffusion training
                latents = latents * 0.18215  # Scaling factor for numerical stability

                # Generate random noise for validation comparison
                noise = torch.randn_like(latents)  # Random noise same shape as latents
                timesteps = torch.randint(
                    0, 1000, (latents.shape[0],)
                )  # Random timesteps
                noisy_latents = (
                    noise + timesteps.reshape(-1, 1, 1, 1) * latents
                )  # Add noise to latents

                # Get noise prediction from U-Net
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Calculate validation loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                val_loss += loss.item()  # Accumulate batch validation loss
                val_progress.set_postfix({"loss": loss.item()})  # Update progress bar

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)

        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model! (Val Loss: {avg_val_loss:.4f})")
            accelerator.wait_for_everyone()  # Sync distributed processes
            unwrapped_unet = accelerator.unwrap_model(unet)  # Get base model
            unwrapped_unet.save_pretrained(
                os.path.join(output_dir, "best_model")  # Save best model weights
            )

        # Regular checkpoint saving
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint = {
                "epoch": epoch,  # Current epoch number
                "model_state_dict": accelerator.unwrap_model(
                    unet
                ).state_dict(),  # Model weights
                "optimizer_state_dict": optimizer.state_dict(),  # Optimizer state
                "train_loss": avg_train_loss,  # Average training loss
                "val_loss": avg_val_loss,  # Average validation loss
                "best_val_loss": best_val_loss,  # Best validation loss so far
            }
            # Save checkpoint to disk
            torch.save(checkpoint, os.path.join(output_dir, f"checkpoint-{epoch+1}.pt"))

    def resume_from_checkpoint(checkpoint_path):
        """Load training state from a checkpoint file

        Args:
            checkpoint_path: Path to the checkpoint file to load

        Returns:
            start_epoch: Epoch number to resume from
            best_val_loss: Best validation loss achieved before checkpoint
        """
        checkpoint = torch.load(checkpoint_path)  # Load checkpoint data
        unet.load_state_dict(checkpoint["model_state_dict"])  # Restore model weights
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )  # Restore optimizer state
        start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        best_val_loss = checkpoint["best_val_loss"]  # Restore best validation loss
        return start_epoch, best_val_loss


if __name__ == "__main__":
    # Example usage of the training function
    train_lora(
        image_dir="path/to/person/images",  # Directory with training images
        output_dir="path/to/save/model",  # Where to save model files
        instance_prompt="photo of sks person",  # Text prompt for training
        num_epochs=100,  # Number of training epochs
    )
