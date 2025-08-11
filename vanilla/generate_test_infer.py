import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
import json
import textwrap
import random
from collections import defaultdict
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

# ==============================================================================
# 1. Import Model Definition from Training Script
# ==============================================================================
# Make sure 'vitdecoder.py' is in the same directory.
try:
    from vitdecoder import ImageCaptioningModel
except ImportError:
    print("Error: Could not import from 'vitdecoder.py'.")
    print("Please ensure the training script is in the same directory and named correctly.")
    exit()


# ==============================================================================
# 2. Data Loading Definitions for Testing
# ==============================================================================
# Define a new Dataset class specifically for testing to include the image name.

class FlickrDatasetTest(Dataset):
    """
    Modified FlickrDataset for testing that returns the image name along with tensors.
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
            
        if self.transform:
            image = self.transform(image)
        
        caption_tensor = self.tokenizer.encode(
            caption, truncation=True, max_length=self.max_len, return_tensors='pt'
        )[0]
        # Return the image name as the first item
        return img_name, image, caption_tensor

def collate_fn_test(batch, padding_value):
    """
    Collator function for the test set that handles the image name.
    """
    # Unzip three items now: image names, image tensors, and caption tensors
    image_names, images, captions = zip(*batch)
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=padding_value)
    # Return the image names as a tuple along with the batched tensors
    return image_names, images, captions

# ==============================================================================
# 3. Inference Function
# ==============================================================================

def generate_caption(model, image_tensor, tokenizer, device, max_length=50):
    """Generates a caption for a single image tensor."""
    model.eval()
    caption_tokens = [tokenizer.cls_token_id]
    with torch.no_grad():
        for _ in range(max_length - 1):
            input_seq = torch.LongTensor([caption_tokens]).to(device)
            output = model(image_tensor, input_seq)
            predicted_id = torch.argmax(output[:, -1, :], axis=-1)
            if predicted_id.item() == tokenizer.sep_token_id:
                break
            caption_tokens.append(predicted_id.item())
    return tokenizer.decode(caption_tokens, skip_special_tokens=True)

# ==============================================================================
# 4. Main Execution Block
# ==============================================================================

def main(args):
    """Main function to load the model and run inference on the entire test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random.seed(42) # for reproducibility

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_path = os.path.join(args.dataset_dir, "Images")
    captions_path = os.path.join(args.dataset_dir, "captions.txt")

    captions_map = defaultdict(list)
    with open(captions_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(',', 1)
            if len(parts) != 2: continue
            image, caption = parts
            captions_map[image.strip()].append(caption.strip().lower())

    unique_images = list(captions_map.keys())
    random.shuffle(unique_images)
    
    train_size = int(0.8 * len(unique_images))
    val_size = int(0.1 * len(unique_images))
    
    test_img_names = unique_images[train_size + val_size :]
    test_pairs = [(img, cap) for img in test_img_names for cap in captions_map[img]]
    
    test_set = FlickrDatasetTest(image_path, test_pairs, transform, tokenizer)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn_test(b, tokenizer.pad_token_id))

    # --- Load Model ---
    print(f"Loading model weights from '{args.model_path}'...")
    model = ImageCaptioningModel(
        vocab_size=tokenizer.vocab_size,
        n_heads=args.n_heads,
        num_decoder_layers=args.num_layers,
        d_feedforward=args.d_feedforward
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # --- Run Inference on the ENTIRE test set ---
    print(f"Running inference on all {len(test_set)} samples in the test set...")
    results = []
    progress_bar = tqdm(test_loader, desc="Generating Captions")

    for image_names, images, gt_caption_tokens in progress_bar:
        images = images.to(device)
        
        for i in range(len(images)):
            image_for_model = images[i].unsqueeze(0)
            image_name = image_names[i]
            
            # Generate caption
            gen_cap = generate_caption(model, image_for_model, tokenizer, device)
            
            # Decode ground truth
            gt_cap = tokenizer.decode(gt_caption_tokens[i], skip_special_tokens=True)
            
            # Append result to list
            results.append({
                "image_name": image_name,
                "caption": gen_cap,
                "ground_truth_caption": gt_cap
            })

    # --- Save Results to JSON ---
    output_filename = 'inference_results.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nInference complete. Results for {len(results)} samples saved to '{output_filename}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and save results for an Image Captioning Model")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, default='best_captioning_model.pth', help='Path to the saved model weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads (must match trained model)')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of decoder layers (must match trained model)')
    parser.add_argument('--d_feedforward', type=int, default=2048, help='Dimension of the feed-forward network (must match trained model)')
    
    args = parser.parse_args()
    main(args)
