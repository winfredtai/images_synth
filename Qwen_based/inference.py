import torch
from PIL import Image
from transformers import CLIPImageProcessor, AutoTokenizer
import argparse
import os

import config
from model import ImageCaptioningModel


def load_checkpoint(model, checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model_state_dict = checkpoint['model_state_dict']
    for name, param in model_state_dict.items():
        if name in dict(model.named_parameters()):
            model.state_dict()[name].copy_(param)
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best BLEU score: {checkpoint['best_bleu']:.4f}")
    
    return model


def generate_caption(model, image_path, clip_processor, device):
    """
    Generate caption for a single image
    """
    image = Image.open(image_path).convert("RGB")
    processed_image = clip_processor(images=image, return_tensors="pt")
    pixel_values = processed_image.pixel_values.to(device)
    model.eval()
    with torch.no_grad():
        captions = model.generate(
            pixel_values=pixel_values,
            max_length=config.MAX_LENGTH,
            num_beams=config.NUM_BEAMS,
            do_sample=False
        )
    
    return captions[0]


def main():
    parser = argparse.ArgumentParser(description='Generate captions for images')
    parser.add_argument('--checkpoint', type=str, required=True, 
                      help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to input image')
    parser.add_argument('--beam_size', type=int, default=4,
                      help='Beam size for generation')
    parser.add_argument('--max_length', type=int, default=50,
                      help='Maximum caption length')
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
    
    config.NUM_BEAMS = args.beam_size
    config.MAX_LENGTH = args.max_length
    
    print(f"\nGenerating caption for: {args.image}")
    caption = generate_caption(model, args.image, clip_processor, config.DEVICE)
    
    print(f"\nGenerated Caption: {caption}")


if __name__ == "__main__":
    main()
