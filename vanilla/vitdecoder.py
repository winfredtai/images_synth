import torch
import torch.nn as nn
import math
import timm
from transformers import BertTokenizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
from tqdm import tqdm
import argparse
import numpy as np
import random
from collections import defaultdict

# ==============================================================================
# 1. Model Definition
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Injects positional information into the input embeddings."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class ImageCaptioningModel(nn.Module):
    """The complete model for image captioning, built using idiomatic PyTorch."""
    def __init__(self, vocab_size, n_heads, num_decoder_layers, d_feedforward):
        super(ImageCaptioningModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        for param in self.vit.parameters():
            param.requires_grad = False
        self.d_model = self.vit.head.in_features
        self.vit.head = nn.Identity()
        self.caption_embedding = nn.Embedding(vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_heads, dim_feedforward=d_feedforward, batch_first=True
        )
        self.transformer_decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer, num_layers=num_decoder_layers
        )
        self.final_layer = nn.Linear(self.d_model, vocab_size)

    def _generate_causal_mask(self, size, device):
        mask = torch.triu(torch.full((size, size), float('-inf')), diagonal=1)
        return mask.to(device)

    def forward(self, image_tensor, caption_tokens):
        image_features_seq = self.vit.forward_features(image_tensor)
        image_features = image_features_seq[:, 0]
        img_seq = image_features.unsqueeze(1)
        cap_embed = self.caption_embedding(caption_tokens)
        cap_embed = self.positional_encoding(cap_embed)
        combined_seq = torch.cat([img_seq, cap_embed], dim=1)
        seq_len = combined_seq.size(1)
        device = caption_tokens.device
        causal_mask = self._generate_causal_mask(seq_len, device)
        decoder_output = self.transformer_decoder(combined_seq, mask=causal_mask)
        final_output = self.final_layer(decoder_output)
        return final_output

# ==============================================================================
# 2. Data Loading and Preparation (Leak-Free)
# ==============================================================================

class FlickrDataset(Dataset):
    """
    Modified FlickrDataset that accepts a pre-made list of (image, caption) pairs.
    """
    def __init__(self, image_dir, data_pairs, transform=None, tokenizer=None, max_len=50):
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = data_pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224))
            print("Image not found, replace with blank image.")
        if self.transform:
            image = self.transform(image)
        
        caption_tensor = self.tokenizer.encode(
            caption, truncation=True, max_length=self.max_len, return_tensors='pt'
        )[0]
        return image, caption_tensor

def collate_fn(batch, padding_value):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=padding_value)
    return images, captions

# ==============================================================================
# 3. Training & Validation Functions
# ==============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, vocab_size):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")
    for images, captions in progress_bar:
        images, captions = images.to(device), captions.to(device)
        input_captions, target_captions = captions[:, :-1], captions[:, 1:]
        optimizer.zero_grad()
        outputs = model(images, input_captions)
        loss = criterion(outputs[:, :-1].reshape(-1, vocab_size), target_captions.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device, vocab_size):
    model.eval()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for images, captions in progress_bar:
            images, captions = images.to(device), captions.to(device)
            input_captions, target_captions = captions[:, :-1], captions[:, 1:]
            outputs = model(images, input_captions)
            loss = criterion(outputs[:, :-1].reshape(-1, vocab_size), target_captions.reshape(-1))
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
    return total_loss / len(dataloader)


# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

def main(args):
    """Main function to run the training and evaluation."""
    
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random.seed(42) # for reproducibility of splits

    # --- Tokenizer and Transforms ---
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    VOCAB_SIZE = tokenizer.vocab_size
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Data Loading and Splitting (Leak-Free Method) ---
    image_path = os.path.join(args.dataset_dir, "Images")
    captions_path = os.path.join(args.dataset_dir, "captions.txt")
    
    # 1. Load all captions into a dictionary mapping image_name -> [captions]
    captions_map = defaultdict(list)
    with open(captions_path, 'r', encoding='utf-8') as f:
        next(f) # Skip header
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2: continue
            image, caption = parts
            captions_map[image.strip()].append(caption.strip().lower())

    # 2. Split the unique image names
    unique_images = list(captions_map.keys())
    random.shuffle(unique_images)
    
    train_size = int(0.8 * len(unique_images))
    val_size = int(0.1 * len(unique_images))
    
    train_img_names = unique_images[:train_size]
    val_img_names = unique_images[train_size : train_size + val_size]
    test_img_names = unique_images[train_size + val_size :]

    print(f"Unique images: Train={len(train_img_names)}, Val={len(val_img_names)}, Test={len(test_img_names)}")

    # 3. Create data pairs for each set
    train_pairs = [(img, cap) for img in train_img_names for cap in captions_map[img]]
    val_pairs = [(img, cap) for img in val_img_names for cap in captions_map[img]]
    test_pairs = [(img, cap) for img in test_img_names for cap in captions_map[img]]

    # 4. Create Datasets and DataLoaders
    train_set = FlickrDataset(image_path, train_pairs, transform, tokenizer)
    val_set = FlickrDataset(image_path, val_pairs, transform, tokenizer)
    test_set = FlickrDataset(image_path, test_pairs, transform, tokenizer)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    # --- Model, Loss, Optimizer ---
    model = ImageCaptioningModel(
        vocab_size=VOCAB_SIZE,
        n_heads=args.n_heads,
        num_decoder_layers=args.num_layers,
        d_feedforward=args.d_feedforward
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # --- Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        avg_train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, VOCAB_SIZE)
        avg_val_loss = validate_one_epoch(model, val_loader, criterion, device, VOCAB_SIZE)
        
        print(f"Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_captioning_model.pth')
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    print("\nTraining complete!")

    final_model_path = f'final_captioning_model_epoch_{args.epochs}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model from last epoch saved to '{final_model_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an Image Captioning Model")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory containing Images/ and captions.txt')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads in the decoder')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers in the decoder stack')
    parser.add_argument('--d_feedforward', type=int, default=2048, help='Dimension of the feed-forward network in the decoder')
    
    args = parser.parse_args()
    main(args)
