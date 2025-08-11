import os
import json
import torch
from torch.utils.data import DataLoader, random_split
from transformers import CLIPImageProcessor, AutoTokenizer
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

import config
from dataset import Flickr8kDataset, collate_fn
from model import ImageCaptioningModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_epoch(model, train_loader, optimizer, device, epoch, scheduler=None):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        pixel_values = batch['pixel_values'].to(device) # Move data to same device -> GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM) # Gradient clipping
        optimizer.step() # Update weights
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        if batch_idx % config.LOGGING_STEPS == 0:
            print(f"Step {batch_idx}, Loss: {loss.item():.4f}")
    
    return avg_loss


def evaluate(model, val_loader, device):
    model.eval()
    
    all_predictions = []
    all_references = []
    total_loss = 0
    
    print("-"*50)
    print("\nEvaluating model...")
    progress_bar = tqdm(val_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            captions = batch['captions']  # Original captions from dataset
            
            # Loss Calculation
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            total_loss += outputs.loss.item()
            
            # Generation
            generated_captions = model.generate(
                pixel_values=pixel_values,
                max_length=config.MAX_LENGTH,
                num_beams=config.NUM_BEAMS,
                do_sample=False
            )
            
            all_predictions.extend(generated_captions)
            all_references.extend(captions)
    
    # BLEU score calculation
    smoothing = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, ref in zip(all_predictions, all_references):
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)
        bleu_scores.append(score)
    
    avg_bleu = np.mean(bleu_scores)
    avg_loss = total_loss / len(val_loader)
    
    # Some example to indicate performance
    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(all_predictions))):
        print(f"\nExample {i+1}:")
        print(f"Generated: {all_predictions[i]}")
        print(f"Reference: {all_references[i]}")
        print(f"BLEU Score: {bleu_scores[i]:.4f}")
    
    print(f"\n=== Evaluation Results ===")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    return avg_bleu, avg_loss


def save_checkpoint(model, optimizer, epoch, best_bleu, checkpoint_path):
    trainable_state_dict = {} # save the trainable parameters only
    for name, param in model.mapping_network.named_parameters():
        if param.requires_grad:
            trainable_state_dict[f"mapping_network.{name}"] = param.data
    
    if config.USE_LORA:
        for name, param in model.llm.named_parameters():
            if param.requires_grad:
                trainable_state_dict[f"llm.{name}"] = param.data

    config_dict = {
        'CLIP_MODEL_ID': config.CLIP_MODEL_ID,
        'LLM_MODEL_ID': config.LLM_MODEL_ID,
        'DEVICE': config.DEVICE,
        'PREFIX_LENGTH': config.PREFIX_LENGTH,
        'PROJECTION_DIM': config.PROJECTION_DIM,
        'USE_LORA': config.USE_LORA,
        'LORA_R': config.LORA_R,
        'LORA_ALPHA': config.LORA_ALPHA,
        'LORA_DROPOUT': config.LORA_DROPOUT,
        'LORA_TARGET_MODULES': config.LORA_TARGET_MODULES,
        'LEARNING_RATE': config.LEARNING_RATE,
        'BATCH_SIZE': config.BATCH_SIZE,
        'NUM_EPOCHS': config.NUM_EPOCHS
    }
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': trainable_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_bleu': best_bleu,
        'config': config_dict
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def main():
    """
    Main training loop
    """
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("Loading processors and tokenizers...")
    clip_processor = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL_ID)
    llm_tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL_ID,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Add padding token if not present
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    print("Loading dataset...")
    full_dataset = Flickr8kDataset(
        dataset_path=config.DATASET_PATH,
        image_dir=config.IMAGE_DIR,
        clip_processor=clip_processor,
        llm_tokenizer=llm_tokenizer
    )
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    print("Initializing model...")
    model = ImageCaptioningModel(config)

    model.llm.tokenizer = llm_tokenizer
    
    if config.DEVICE == "cuda" and not hasattr(model.llm, 'hf_device_map'):
        model = model.to(config.DEVICE)
    
    # Only optimize trainable parameters
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"Trainable: {name}")
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.NUM_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=config.LEARNING_RATE * 0.1
    )
    
    best_bleu = 0
    
    print("\nStarting training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        print(f"{'='*50}")
        
        # Train for one epoch
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=config.DEVICE,
            epoch=epoch,
            scheduler=scheduler
        )
        
        print(f"\nEpoch {epoch} - Average training loss: {train_loss:.4f}")
        
        bleu_score, val_loss = evaluate(
            model=model,
            val_loader=val_loader,
            device=config.DEVICE
        )
        
        # Save checkpoint if better performance
        if bleu_score > best_bleu:
            best_bleu = bleu_score
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"best_model_epoch_{epoch}_bleu_{bleu_score:.4f}.pt"
            )
            save_checkpoint(model, optimizer, epoch, best_bleu, checkpoint_path)
        
        # Save regular checkpoint, every 2 epochs
        if epoch % 2 == 0:
            checkpoint_path = os.path.join(
                config.CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch}.pt"
            )
            save_checkpoint(model, optimizer, epoch, best_bleu, checkpoint_path)
    
    print(f"\n{'='*50}")
    print("Training completed")
    print(f"Best BLEU score: {best_bleu:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    try:
        import nltk
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        import nltk
        nltk.download('punkt')
    
    main()
