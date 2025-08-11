import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
import matplotlib.pyplot as plt
import textwrap
import random
from collections import defaultdict
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# ==============================================================================
# 1. Import Definitions from Training Script
# ==============================================================================
# Make sure 'vitdecoder.py' is in the same directory.
try:
    from vitdecoder import ImageCaptioningModel, FlickrDataset, collate_fn
except ImportError:
    print("Error: Could not import from 'vitdecoder.py'.")
    print("Please ensure the training script is in the same directory and named correctly.")
    exit()


# ==============================================================================
# 2. Inference and Plotting Functions
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

def plot_inference_results(images, gt_captions, gen_captions, filename='inference_results.png'):
    """Un-normalizes images and plots them with their captions."""
    num_images = len(images)
    cols = 8
    rows = (num_images + cols - 1) // cols
    plt.figure(figsize=(25, 3 * rows))
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    for i in range(num_images):
        ax = plt.subplot(rows, cols, i + 1)
        img = inv_normalize(images[i]).cpu().permute(1, 2, 0)
        ax.imshow(img)
        title_text = f"GT: {textwrap.fill(gt_captions[i], 25)}\n\nGen: {textwrap.fill(gen_captions[i], 25)}"
        ax.set_title(title_text, fontsize=8)
        ax.axis("off")
    plt.tight_layout(pad=2.0)
    plt.savefig(filename)
    print(f"Inference plot saved to '{filename}'")

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================

def main(args):
    """Main function to load the model and run inference."""
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
    
    # We only need the test image names for this script
    test_img_names = unique_images[train_size + val_size :]
    test_pairs = [(img, cap) for img in test_img_names for cap in captions_map[img]]
    
    test_set = FlickrDataset(image_path, test_pairs, transform, tokenizer)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

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

    # --- Run Inference ---
    print(f"Running inference on the first batch of {args.batch_size} images from the test set...")
    images, gt_caption_tokens = next(iter(test_loader))
    images = images.to(device)
    
    generated_captions, ground_truth_captions = [], []

    for i in range(len(images)):
        image_for_model = images[i].unsqueeze(0)
        gen_cap = generate_caption(model, image_for_model, tokenizer, device)
        generated_captions.append(gen_cap)
        gt_cap = tokenizer.decode(gt_caption_tokens[i], skip_special_tokens=True)
        ground_truth_captions.append(gt_cap)

    plot_inference_results(images, ground_truth_captions, generated_captions)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and plot results for an Image Captioning Model")
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, default='best_captioning_model.pth', help='Path to the saved model weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of images to infer and plot')
    parser.add_argument('--n_heads', type=int, default=12, help='Number of attention heads (must match trained model)')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of decoder layers (must match trained model)')
    parser.add_argument('--d_feedforward', type=int, default=2048, help='Dimension of the feed-forward network (must match trained model)')
    
    args = parser.parse_args()
    main(args)
