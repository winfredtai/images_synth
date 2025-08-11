import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class Flickr8kDataset(Dataset):    
    def __init__(self, dataset_path, image_dir, clip_processor, llm_tokenizer):
        """
        Dataloader for the Flickr8k dataset
        """
        self.image_dir = image_dir
        self.clip_processor = clip_processor
        self.llm_tokenizer = llm_tokenizer
        
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # Create flattened list of image-caption pairs
        self.image_caption_pairs = []
        for item in data:
            image_name = item['name']
            captions = item['captions']
            used_caption = captions[0]
            
            self.image_caption_pairs.append({
                'image_path': os.path.join(self.image_dir, image_name),
                'caption': used_caption
            })

    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        item = self.image_caption_pairs[idx]
        image_path = item['image_path']
        caption = item['caption']
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Process image using CLIP processor, squeeze to 224x224x3
        processed_image = self.clip_processor(images=image, return_tensors="pt")
        
        # [Caption + EOS token] as the target text
        full_text = f"{caption}{self.llm_tokenizer.eos_token}"
        
        tokenized_text = self.llm_tokenizer(
            full_text,
            padding="max_length",
            max_length=128,  # average caption length is 55.27 as we calculated, so set to 128, double of that
            truncation=True,
            return_tensors="pt"
        )
        
        # Return dictionary as squeezed tensors -> remove batch dimension
        return {
            'pixel_values': processed_image.pixel_values.squeeze(0),
            'input_ids': tokenized_text.input_ids.squeeze(0),
            'attention_mask': tokenized_text.attention_mask.squeeze(0),
            'caption': caption  # original caption for evaluation
        }


def collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    captions = [item['caption'] for item in batch]
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'captions': captions
    }
