import json
import shutil
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))


def process_captions():
    '''
    Some examples from the captions.txt file:
    image,caption
    xxx
    1009434119_febe49276a.jpg,A Boston Terrier is running on lush green grass in front of a white fence .
    1009434119_febe49276a.jpg,A dog runs on the green grass near a wooden fence .
    1012212859_01547e3f17.jpg,"A dog shakes its head near the shore , a red ball next to it ."
    1012212859_01547e3f17.jpg,A white dog shakes on the edge of a beach with an orange ball .
    xxx
    '''
    image_data = {}
    with open(f'{script_dir}/captions.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    # Skip header line -> image,caption
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
            
        # Split by first comma only
        parts = line.split(',', 1)
        if len(parts) == 2:
            image_name, caption = parts[0].strip(), parts[1].strip()
            
            # Remove leading and trailing double quotes
            if caption.startswith('"') and caption.endswith('"'):
                caption = caption[1:-1]
            
            # Remove trailing space + period
            if caption.endswith(' .'):
                caption = caption[:-2]
            
            if image_name not in image_data:
                image_data[image_name] = []
            image_data[image_name].append(caption)
    
    result = []
    for image_name, captions in image_data.items():
        result.append({
            "name": image_name,
            "path": f"images/{image_name}",
            "absolute_path": str(Path(script_dir) / "Images" / image_name),
            "captions": captions
        })
    
    with open(f'{script_dir}/captions.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    
    print(f"Processed {len(result)} images")
    print("Output saved to captions.json")

def check_image_path():
    with open (f'{script_dir}/captions.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    image_file = {"exist": sum(os.path.exists(item['absolute_path']) for item in data),
                  "not_exist": sum(not os.path.exists(item['absolute_path']) for item in data)}
    print(f"Images exist: {image_file['exist']}, Images not exist: {image_file['not_exist']}")


def create_train_n_val_test(json_path, train_n_val_ratio=0.9):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    import random
    random.seed(42)
    random.shuffle(dataset)
    
    # Split
    split_idx = int(len(dataset) * train_n_val_ratio)
    train_n_val_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    base_path = os.path.splitext(json_path)[0]
    
    train_path = f"{base_path}_train_n_val.json"
    with open(train_path, 'w') as f:
        json.dump(train_n_val_data, f, indent=2)
    
    val_path = f"{base_path}_test.json"
    with open(val_path, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nTraining and validation set saved to {train_path} ({len(train_n_val_data)} images)")
    print(f"Testing set saved to {val_path} ({len(test_data)} images)")


def create_subset(json_path, output_path, num_samples=100):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    subset = dataset[:num_samples]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(subset, f, indent=2)
    
    print(f"Subset of {num_samples} samples saved to {output_path}")

def dataset_summary(json_path):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    total_images = len(dataset)
    total_captions = sum(len(item['captions']) for item in dataset)
    
    print(f"Total images: {total_images}")
    print(f"Total captions: {total_captions}")
    print(f"Average captions per image: {total_captions / total_images if total_images > 0 else 0:.2f}")  # 5 captions per image
    print(f"Average lenght of the captions: {sum(len(caption) for item in dataset for caption in item['captions']) / total_captions if total_captions > 0 else 0:.2f}")  # average caption length


def create_single_caption_json(json_path, output_path):
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    single_caption_data = []
    for item in dataset:
        if item['captions']:
            single_caption_data.append({
                "name": item['name'],
                "path": item['path'],
                "absolute_path": item['absolute_path'],
                "caption": item['captions'][0]  # Use the first caption
            })
    
    with open(output_path, 'w') as f:
        json.dump(single_caption_data, f, indent=2)
    
    print(f"Single caption dataset saved to {output_path}")

def append_ground_truth_captions(generated_captions_path, ground_truth_path, output_path):
    with open(generated_captions_path, 'r') as f:
        generated_captions = json.load(f)
    
    with open(ground_truth_path, 'r') as f:
        ground_truth_captions = json.load(f)
    
    # Create a lookup dictionary for ground truth captions using image_name as key
    gt_lookup = {item['name']: item['caption'] for item in ground_truth_captions}
    
    updated_captions = []
    for item in generated_captions:
        updated_item = item.copy()
        image_name = item['image_name']
        if image_name in gt_lookup:
            updated_item['ground_truth_caption'] = gt_lookup[image_name]
        else:
            updated_item['ground_truth_caption'] = None
            print(f"Warning: No ground truth found for {image_name}")
        
        updated_captions.append(updated_item)

    with open(output_path, 'w') as f:
        json.dump(updated_captions, f, indent=2)
    
    print(f"Successfully merged {len(updated_captions)} captions")
    print(f"Results saved to: {output_path}")
    
    return updated_captions

def batch_copy_img(json_file_path, destination_dir):
    copied = 0
    os.makedirs(destination_dir, exist_ok=True)
    with open(json_file_path, 'r') as f:
        image_data = json.load(f)
    print(f"loaded {len(image_data)} images")

    try:
        for item in image_data:
            source_path = item['absolute_path']
            image_name = item['name']
            destination_path = os.path.join(destination_dir, image_name)
            shutil.copy2(source_path, destination_path)
            copied += 1
        print(f"Copied {copied} images to {destination_dir}")
    except Exception as e:
        print(f"Error copying images: {e}")
        return
        

if __name__ == "__main__":
    # process_captions()
    # check_image_path()
    # create_train_n_val_test(f'{script_dir}/captions.json', train_n_val_ratio=0.9)
    # num_samples = 100
    # create_subset(f'{script_dir}/captions.json', f'{script_dir}/captions_subset_{num_samples}.json', num_samples=num_samples) 
    # dataset_summary(f'{script_dir}/captions.json')
    # create_single_caption_json(f'{script_dir}/captions.json', f'{script_dir}/captions_single.json')
    # batch_copy_img(f'{script_dir}/captions_test.json', f'{script_dir}/Images_test')

    generated_path = "/root/autodl-tmp/dir_tzh/ECE1508S25/repo/imageSythn/datasets/generated_captions_0805.json"
    ground_truth_path = "/root/autodl-tmp/dir_tzh/ECE1508S25/repo/imageSythn/datasets/captions_single.json"
    output_path = "/root/autodl-tmp/dir_tzh/ECE1508S25/repo/imageSythn/datasets/merged_captions_0805_2.json"
    result = append_ground_truth_captions(generated_path, ground_truth_path, output_path)

