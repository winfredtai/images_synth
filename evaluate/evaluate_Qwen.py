### 
"""
Test data: 810 images in Flickr8k dataset
Metrics: BLEU, METEOR, ROUGE-L, CIDEr, LLM (deepseek-v3), LLM (gpt-4.1-mini)
Sample and plot
"""

"""
pip install nltk matplotlib scikit-image rouge-score
pip install git+https://github.com/tylin/coco-caption.git   
# Can't pip install: just copy the cider.py and cider_scorer.py file from this repo to the current directory
# modify some snyntax in cider_scorer.py (xrange()->range(), iteritems()->items()) to make it compatible with Python 3
pip install openai
"""

import json
import os
from skimage.io import imread
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from cider import Cider
import matplotlib.pyplot as plt
import random

from openai import OpenAI

nltk.download("punkt")
nltk.download("wordnet")
nltk.download('punkt_tab')

random.seed(42)  # for reproducibility
image_path = "./Flickr8k/Images/"   
json_path = "./"
# json_file = "merged_captions_0805.json"   # CLIP+LoRA(QWen): LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
json_file = "merged_captions_0805_2.json" # CLIP+LoRA(QWen): LORA_TARGET_MODULES = ["q_proj", "v_proj"] 
# json_file = "inference_results.json"      # ViT + 4 * TransformerDecoder
visualization_dir = "./samples"
os.makedirs(visualization_dir, exist_ok=True)

n_samples = 4 

caption = "A boy in a green shirt sits on the grass"
ground_truth_caption = "A boy on a playground swing strikes a medatative pose"

api_key_Winfred = "sk-proj-"
api_key_gpt41mini = "sk-proj-"
api_key_deepseekv3 = "sk-"

def llm_rate_deepseekv3(generated_caption, ground_truth_caption):
    client = OpenAI(
        api_key=api_key_deepseekv3,
        base_url="https://api.lkeap.cloud.tencent.com/v1"
        )
    prompt = f"""
        You are an expert image caption evaluator.
        Given a generated caption, and a ground truth caption, please evaluate the quality of the generated caption based on the following criteria:
        1. Accuracy and correctness: Does the generated caption accurately capture the objects, actions, and relationships present in the image? Are there any mistakes or hallucinations?
        2. Semantic similarity to the reference caption: How similar is the generated caption to the reference (ground truth) caption in meaning, even if the wording is different?
        Instructions:
        - Give a single overall rating score from 0 to 5, where 5 means the generated caption is perfectly accurate, highly relevant to the image, and semantically matches the reference caption; 0 means it is completely wrong or unrelated.
        - Please consider all criteria above, and be as objective and strict as possible.
        - Output only the numeric score and nothing else.

        Generated caption:
        {generated_caption}

        Reference (ground truth) caption:
        {ground_truth_caption}
        """
    completion = client.chat.completions.create(
        model="deepseek-v3",
        messages=[
            {"role": "user", "content": f"Rate the caption based on the ground truth caption. The rating is from 1 to 5, with 5 being the best. The caption is: {caption}. The ground truth caption is: {ground_truth_caption}. Provide only the rating number as output."}
        ],
    )
    # print(completion.choices[0].message.content)
    return float(completion.choices[0].message.content.strip())
# llm_rate_deepseekv3(caption, ground_truth_caption)

def llm_rate_gpt41mini(image, generated_caption, ground_truth_caption):
    client = OpenAI(
        api_key=api_key_gpt41mini   # api_key_Winfred
    )

    prompt = f"""
        You are an expert image caption evaluator.
        Given an image, a generated caption, and a ground truth caption, please evaluate the quality of the generated caption based on the following criteria:
        1. Relevance to the image: How well does the generated caption describe the actual content of the image?
        2. Accuracy and correctness: Does the generated caption accurately capture the objects, actions, and relationships present in the image? Are there any mistakes or hallucinations?
        3. Semantic similarity to the reference caption: How similar is the generated caption to the reference (ground truth) caption in meaning, even if the wording is different?
        Instructions:
        - Give a single overall rating score from 0 to 5, where 5 means the generated caption is perfectly accurate, highly relevant to the image, and semantically matches the reference caption; 0 means it is completely wrong or unrelated.
        - Please consider all criteria above, and be as objective and strict as possible.
        - Output only the numeric score (e.g., 0.78) and nothing else.

        Generated caption:
        {generated_caption}

        Reference (ground truth) caption:
        {ground_truth_caption}
        """

    # Function to create a file with the Files API
    def create_file(file_path):
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id

    # Getting the file ID
    file_id = create_file(image)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", 
                "text": prompt,
    },
                {
                    "type": "input_image",
                    "file_id": file_id,
                },
            ],
        }],
    )

    # print(response.output_text)
    return float(response.output_text.strip())
# quick test
# image = "./Flickr8k/Images/1463732807_0cdf4f22c7.jpg"
# caption = "A boy in a green shirt sits on the grass"
# ground_truth_caption = "A boy on a playground swing strikes a medatative pose"
# llm_rate_deepseekv3(caption, ground_truth_caption)
# llm_rate_gpt41mini(image, caption, ground_truth_caption)
# exit()

with open(os.path.join(json_path, json_file)) as f:
    print(f"Loading JSON file: {json_file}")
    json_contents = json.load(f)

image_names = []
captions = []
gt_captions = []

for item in json_contents:
    image_names.append(item['image_name'])
    captions.append(item['caption'])
    gt_captions.append(item['ground_truth_caption'])

bleu_scores = []
meteor_scores = []
# cider_scores = [] # cider_mean
rouge_l_scores = []
llm_scores_deepseekv3 = []
llm_scores_gpt41mini = []
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
cider_eval = Cider()

coco_refs = {i: [gt_captions[i]] for i in range(len(gt_captions))}
coco_hyps = {i: [captions[i]] for i in range(len(captions))}


# 计算 CIDEr
cider_mean, cider_score = cider_eval.compute_score(coco_refs, coco_hyps)
for i in range(len(gt_captions)):  
    gt_captions_tokenized = nltk.word_tokenize(gt_captions[i])
    captions_tokenized = nltk.word_tokenize(captions[i])
    # LLM (deepseek-v3)
    llm_score_deepseekv3 = llm_rate_deepseekv3(captions[i], gt_captions[i])
    llm_scores_deepseekv3.append(llm_score_deepseekv3)

    # LLM (gpt-4.1-mini)
    llm_score_gpt41mini = llm_rate_gpt41mini(os.path.join(image_path, image_names[i]), captions[i], gt_captions[i])
    llm_scores_gpt41mini.append(llm_score_gpt41mini)

    # BLEU-4
    bleu = sentence_bleu([gt_captions_tokenized], captions_tokenized, smoothing_function=SmoothingFunction().method1)
    bleu_scores.append(bleu)
    # METEOR
    meteor = meteor_score([gt_captions_tokenized], captions_tokenized)
    meteor_scores.append(meteor)

    # ROUGE-L
    rouge_l = rouge.score(gt_captions[i], captions[i])['rougeL'].fmeasure
    rouge_l_scores.append(rouge_l)

avg_BLEU = sum(bleu_scores)/len(bleu_scores)
avg_METEOR = sum(meteor_scores)/len(meteor_scores)
avg_CIDEr = cider_mean
avg_ROUGE_L = sum(rouge_l_scores)/len(rouge_l_scores)
avg_LLM_deepseekv3 = sum(llm_scores_deepseekv3)/len(llm_scores_deepseekv3)
avg_LLM_gpt41mini = sum(llm_scores_gpt41mini)/len(llm_scores_gpt41mini)

print(f"avg_BLEU: {avg_BLEU:.3f}")
print(f"avg_METEOR: {avg_METEOR:.3f}")
print(f"avg_CIDEr: {avg_CIDEr:.3f}")
print(f"avg_ROUGE-L: {avg_ROUGE_L:.3f}")
print(f"avg_LLM_deepseekv3: {avg_LLM_deepseekv3:.3f}")
print(f"avg_LLM_gpt41mini: {avg_LLM_gpt41mini:.3f}")

# visualization
assert n_samples <= len(image_names), "n_samples should be <= len(image_names)"
fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))

if n_samples == 1:
    axes = [axes]

# random sample images to visualize
random_indices = random.sample(range(len(image_names)), n_samples)
print("Randomly sampled indices:", random_indices)
sampled_image_names = [image_names[i] for i in random_indices]
sampled_gt_captions = [gt_captions[i] for i in random_indices]
sampled_captions = [captions[i] for i in random_indices]
# for i, (img_name, true_caption, pred_caption) in enumerate(zip(image_names, gt_captions, captions)):
for i, (img_name, true_caption, pred_caption) in enumerate(zip(sampled_image_names, sampled_gt_captions, sampled_captions)):
    # plot
    if i >= n_samples:
        break
    print("Processing image:", img_name)

    image = imread(os.path.join(image_path, img_name))
    axes[i][0].imshow(image)
    axes[i][0].axis('off')
    axes[i][0].set_title(img_name)

    # 显示caption
    true_annotation = f"TRUE: {true_caption}"
    pred_annotation = f"PRED: {pred_caption}"
    axes[i][1].text(0, 0.7, true_annotation, fontsize=14, wrap=True)
    axes[i][1].text(0, 0.3, pred_annotation, fontsize=14, wrap=True)
    axes[i][1].axis('off')
    axes[i][1].set_title(f"BLEU: {bleu_scores[i]:.2f}\nMETEOR: {meteor_scores[i]:.2f}\nROUGE-L: {rouge_l_scores[i]:.2f}\navgCIDEr: {avg_CIDEr:.2f}\n deepseek-v3: {llm_scores_deepseekv3[i]:.2f}\ngpt-4.1-mini: {llm_scores_gpt41mini[i]:.2f}")

fig.suptitle(
    f"avgBLEU: {avg_BLEU:.3f} | avgMETEOR: {avg_METEOR:.3f} \n"
    f"avgROUGE-L: {avg_ROUGE_L:.3f} | avgCIDEr: {avg_CIDEr:.3f} \n"
    f"avgLLM (deepseek-v3): {avg_LLM_deepseekv3:.3f} | avgLLM (gpt-4.1-mini): {avg_LLM_gpt41mini:.3f}",
    fontsize=20, ha='center', va='bottom',
    y=1.0
)


save_fig_path = f"show_{json_file}.png"
save_path = os.path.join(visualization_dir, save_fig_path)
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight', dpi=150)
plt.close(fig)
print(f"Saved image to {save_path}")

# def save_results_to_json():
#     results = {
#         "image_name": image_names[:n_samples],
#         "caption": captions[:n_samples],
#         "ground_truth_caption": gt_captions[:n_samples],
#         "BLEU": bleu_scores[:n_samples],
#         "METEOR": meteor_scores[:n_samples],
#         "CIDEr_mean": cider_mean,
#         "ROUGE_L": rouge_l_scores[:n_samples],
#         "LLM_deepseekv3": llm_scores_deepseekv3[:n_samples],
#         "LLM_gpt41mini": llm_scores_gpt41mini[:n_samples]
#     }
#     output_path = "./evaluation_results"
#     os.makedirs(output_path, exist_ok=True)
#     output_file = os.path.join(output_path, f"evaluate_{os.path.basename(json_file)}")

#     with open(output_file, 'w') as f:
#         json.dump(results, f, indent=2)
#     print(f"Results saved to {output_file}")
# print(len(image_names), len(bleu_scores), len(meteor_scores), len(cider_score), len(rouge_l_scores), len(llm_scores_deepseekv3), len(llm_scores_gpt41mini))

def save_results_to_json():
    results = []
    for i in range(len(image_names)):  
        record = {
            "image_name": image_names[i],
            "caption": captions[i],
            "ground_truth_caption": gt_captions[i],
            "BLEU": bleu_scores[i],
            "METEOR": meteor_scores[i],
            "CIDEr_mean": cider_mean,
            "ROUGE_L": rouge_l_scores[i],
            "LLM_deepseekv3": llm_scores_deepseekv3[i],
            "LLM_gpt41mini": llm_scores_gpt41mini[i]
        }
        results.append(record)

    output_path = "./evaluation_results"
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"evaluate_{os.path.basename(json_file)}")

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


save_results_to_json()


