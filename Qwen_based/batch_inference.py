import torch
from PIL import Image
from transformers import CLIPImageProcessor, AutoTokenizer
import argparse
import os
import json
from tqdm import tqdm
import glob

import config
from model import ImageCaptioningModel


def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE,  weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    
    for name, param in model_state_dict.items():
        if name in dict(model.named_parameters()):
            model.state_dict()[name].copy_(param)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best BLEU score: {checkpoint['best_bleu']:.4f}")
    
    return model


def process_batch(model, image_paths, clip_processor, device, batch_size=8):
    all_captions = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                batch_images.append(image)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                batch_images.append(Image.new('RGB', (224, 224), color='black'))
        
        processed = clip_processor(images=batch_images, return_tensors="pt")
        pixel_values = processed.pixel_values.to(device)
        
        with torch.no_grad():
            captions = model.generate(
                pixel_values=pixel_values,
                max_length=config.MAX_LENGTH,
                num_beams=config.NUM_BEAMS,
                do_sample=False
            )
        
        all_captions.extend(captions)
    
    return all_captions


def main():
    parser = argparse.ArgumentParser(description='Generate captions for multiple images')
    parser.add_argument('--checkpoint', type=str, required=True, 
                      help='Path to model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing input images')
    parser.add_argument('--output_file', type=str, default='captions_output.json',
                      help='Output JSON file for captions')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='Batch size for processing')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'jpeg', 'png'],
                      help='Image file extensions to process')
    args = parser.parse_args()
    
    print("Loading processors...")
    clip_processor = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL_ID)
    llm_tokenizer = AutoTokenizer.from_pretrained(
        config.LLM_MODEL_ID,
        trust_remote_code=True
    )
    
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    
    print("Initializing model...")
    model = ImageCaptioningModel(config)
    model.llm.tokenizer = llm_tokenizer
    
    model = load_checkpoint(model, args.checkpoint)
    
    if config.DEVICE == "cuda" and not hasattr(model.llm, 'hf_device_map'):
        model = model.to(config.DEVICE)
    
    model.eval()
    
    image_paths = []
    for ext in args.extensions:
        pattern = os.path.join(args.input_dir, f"*.{ext}")
        image_paths.extend(glob.glob(pattern))
        pattern = os.path.join(args.input_dir, f"*.{ext.upper()}")
        image_paths.extend(glob.glob(pattern))
    
    image_paths = list(set(image_paths))  # Remove duplicates
    print(f"Found {len(image_paths)} images to process")
    
    if len(image_paths) == 0:
        print("No images found!")
        return
    
    print("Generating captions...")
    captions = process_batch(model, image_paths, clip_processor, 
                           config.DEVICE, args.batch_size)
    
    results = []
    for path, caption in zip(image_paths, captions):
        results.append({
            'image_path': path,
            'image_name': os.path.basename(path),
            'caption': caption
        })
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nCaptions saved to {args.output_file}")
    
    # Some examples
    print("\n=== Sample Results ===")
    for i in range(min(5, len(results))):
        print(f"\nImage: {results[i]['image_name']}")
        print(f"Caption: {results[i]['caption']}")


if __name__ == "__main__":
    main()
