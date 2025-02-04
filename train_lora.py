import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from accelerate import Accelerator
from tqdm.auto import tqdm
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model

class PersonDataset(Dataset):
    def __init__(self, image_dir, instance_prompt, tokenizer, size=512):
        self.image_dir = image_dir
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Tokenize text
        text_inputs = self.tokenizer(
            self.instance_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0]
        }

def train_lora(
    image_dir: str,
    output_dir: str,
    instance_prompt: str = "photo of sks person",
    batch_size: int = 1,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    rank: int = 4,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    validation_split: float = 0.1,  # 10% for validation
    checkpoint_freq: int = 10,  # Save every 10 epochs
):
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
    )

    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
    )

    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    
    # Split dataset into train and validation
    dataset = PersonDataset(image_dir, instance_prompt, tokenizer)
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Setup optimizer
    optimizer = bnb.optim.AdamW8bit(
        unet.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2
    )

    # Prepare for training
    unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader
    )

    # Track best model
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        unet.train()
        train_loss = 0
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Get input values
                pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                input_ids = batch["input_ids"]

                # Encode text
                encoder_hidden_states = text_encoder(input_ids)[0]

                # Get latent space representation
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215

                # Add noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],))
                noisy_latents = noise + timesteps.reshape(-1, 1, 1, 1) * latents

                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Calculate loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)

                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # Validation phase
        unet.eval()
        val_loss = 0
        with torch.no_grad():  # Don't compute gradients during validation
            for batch in val_dataloader:
                # Same forward pass as training
                pixel_values = batch["pixel_values"].to(dtype=torch.float16)
                input_ids = batch["input_ids"]
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, 1000, (latents.shape[0],))
                noisy_latents = noise + timesteps.reshape(-1, 1, 1, 1) * latents
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Log progress
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint if it's the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            accelerator.wait_for_everyone()
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(
                os.path.join(output_dir, "best_model")
            )
        
        # Regular checkpointing
        if (epoch + 1) % checkpoint_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': accelerator.unwrap_model(unet).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }
            torch.save(
                checkpoint,
                os.path.join(output_dir, f"checkpoint-{epoch+1}.pt")
            )

    # To resume from checkpoint:
    def resume_from_checkpoint(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        return start_epoch, best_val_loss

if __name__ == "__main__":
    # Example usage
    train_lora(
        image_dir="path/to/person/images",
        output_dir="path/to/save/model",
        instance_prompt="photo of sks person",
        num_epochs=100
    )