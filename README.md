# ImageSynth Captions: A Unified Framework for Visual to Textual Generation

This repository implements two distinct approaches for image captioning using vision-language models, trained and evaluated on the Flickr8k dataset.

## Table of Contents
- [Overview](#overview)
- [Models Architecture](#models-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)

## Overview

This project explores multimodal generative models that take images as input and generate descriptive captions. We implement and compare two architectures:

1. **Vanilla Model**: ViT encoder with Transformer decoder
2. **Advanced Model**: CLIP encoder with Qwen2.5-3B LLM using LoRA fine-tuning

Both models are trained on the Flickr8k dataset and evaluated using multiple metrics including BLEU, METEOR, ROUGE-L, CIDEr, and LLM-based evaluation (GPT-4.1-mini).

## Models Architecture

### Model 1: ViT + Transformer Decoder

**Architecture Components:**
- **Vision Encoder**: Frozen pre-trained `vit-base-patch16-224`
  - Extracts [CLS] token as image feature (768-dim)
- **Decoder Stack**: 4 Transformer decoder layers
  - Hidden size: 768
  - Attention heads: 12
  - Feed-forward dimension: 2048
- **Caption Tokenizer**: Pre-trained BERT base model
- **Input Processing**: Concatenates image [CLS] token with embedded caption tokens including positional encoding

### Model 2: CLIP + Qwen-2.5 with LoRA

**Architecture Pipeline:**
```
Input Image → CLIP Processor → Frozen CLIP Encoder → MLP Mapping Network → Qwen-2.5 LLM → Generated Caption
```

**Components Details:**
- **CLIP Vision Encoder**: `openai/clip-vit-large-patch14` (frozen)
  - Extracts semantic image features
  - Outputs [CLS] embedding (1×1024)
  
- **Mapping Network (MLP)**: 
  - Projects CLIP embeddings to LLM embedding space
  - Generates visual prefix sequence (length: 10 tokens)
  - Two-layer MLP with GELU activation
  
- **Language Model**: `Qwen/Qwen2.5-3B`
  - Fine-tuned using LoRA for parameter efficiency
  - LoRA configuration:
    - Rank (r): 32
    - Alpha: 64
    - Target modules: ["q_proj", "v_proj"]
    - Dropout: 0.05
  
- **Training Input Format**: `[visual_prefix_sequence | caption_embeddings]`

## Dataset

**Flickr8k Dataset**
- 8,000 images with 5 captions each
- Split ratio:
  - Training: 90% (including 90% train, 10% validation from train split)
  - Testing: 10%
- Average caption length: ~55 tokens
- Preprocessing: Images resized to 224×224, normalized using ImageNet statistics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for evaluation)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('punkt_tab')"

# Additional dependencies for specific components
pip install timm transformers peft
pip install git+https://github.com/tylin/coco-caption.git
pip install openai  # For LLM evaluation
```

## Project Structure

```
.
├── datasets/
│   ├── captions.json          # Processed captions
│   ├── captions.txt           # Original captions
│   └── prepare_dataset.py     # Dataset preparation utilities
├── evaluate/
│   ├── cider.py               # CIDEr metric implementation
│   ├── cider_scorer.py        # CIDEr scoring utilities
│   ├── evaluate_Qwen.py       # Evaluation for Qwen model
│   └── evaluate_vanilla.py    # Evaluation for vanilla model
├── Qwen_based/
│   ├── model.py               # Qwen model architecture
│   ├── config.py              # Configuration parameters
│   ├── dataset.py             # Dataset loader
│   ├── train.py               # Training script
│   ├── inference.py           # Single image inference
│   └── batch_inference.py     # Batch inference
└── vanilla/
    ├── vitdecoder.py          # ViT+Decoder model
    ├── generate_demo_img.py   # Generate demo visualizations
    └── generate_test_infer.py # Test set inference
```

## Usage

### Training

#### Vanilla Model (ViT + Transformer Decoder)
```bash
python vanilla/vitdecoder.py \
    --dataset_dir ./datasets/Flickr8k \
    --epochs 10 \
    --batch_size 16 \
    --lr 1e-4 \
    --n_heads 12 \
    --num_layers 4 \
    --d_feedforward 2048
```

#### Qwen-based Model
```bash
# Configure parameters in config.py, then run:
python Qwen_based/train.py
```

Key configuration parameters in `config.py`:
- `BATCH_SIZE`: 8
- `NUM_EPOCHS`: 5
- `LEARNING_RATE`: 5e-5
- `PREFIX_LENGTH`: 10 (visual tokens)
- `LORA_R`: 32
- `NUM_BEAMS`: 4 (for beam search)

### Inference
The checkpoints can be found in the Hugging Face repository: [winfredtai/image_synth](https://huggingface.co/winfredtai/image_synth)

#### Single Image Caption Generation
```bash
# Vanilla model
python vanilla/generate_test_infer.py \
    --dataset_dir ./datasets/Flickr8k \
    --model_path best_captioning_model.pth \
    --batch_size 64

# Qwen model
python Qwen_based/inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --beam_size 4 \
    --max_length 50
```

#### Batch Inference
```bash
python Qwen_based/batch_inference.py \
    --checkpoint ./checkpoints/best_model.pt \
    --input_dir ./test_images \
    --output_file results.json \
    --batch_size 8
```

## Evaluation

The evaluation pipeline computes multiple metrics:

### Traditional Metrics
- **BLEU-4**: Measures n-gram precision
- **METEOR**: Considers synonyms and stemming
- **ROUGE-L**: Longest common subsequence
- **CIDEr**: Consensus-based metric for image descriptions

### LLM-based Evaluation
- **GPT-4.1-mini**: Evaluates with image context

### Running Evaluation
```bash
# For Qwen model
python evaluate/evaluate_Qwen.py

# For vanilla model
python evaluate/evaluate_vanilla.py
```

The evaluation scripts will:
1. Load test set predictions
2. Calculate all metrics
3. Generate visualization with sample results
4. Save detailed results to JSON

## Results

### Model Performance Comparison

| Model | B-1 | B-2 | B-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr | LLM (GPT-4.1) | 
|-------|-----|-----|-----|--------|--------|---------|-------|----------------|
| Vanilla (ViT+Decoder) | 0.184 | 0.042 | 0.025 | 0.021 | 0.134 | 0.230 | 0.144 | 1.601 |
| CLIP+Qwen (LoRA full - 1.90% trainable) | 0.339 | **0.207** | **0.135** | **0.094** | 0.329 | 0.397 | 0.832 | 2.738 |
| CLIP+Qwen (LoRA q,v - 0.24% trainable) | **0.341** | 0.206 | 0.132 | 0.091 | **0.336** | **0.399** | **0.839** | 2.742 |

### Key Findings
- The CLIP+Qwen model with LoRA fine-tuning shows superior performance
- Visual prefix length of 10 tokens provides optimal balance
- LoRA on attention projections (q_proj, v_proj) achieves comparable results with fewer parameters

### Sample Outputs
Sample visualizations are saved in `./samples/` directory showing:
- Original images
- Ground truth captions
- Generated captions
- Per-sample metrics

## Configuration Details

### Hyperparameters (Qwen Model)
```python
# Vision Encoder
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# Language Model
LLM_MODEL_ID = "Qwen/Qwen2.5-3B"

# Training
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_EPOCHS = 5

# Architecture
PREFIX_LENGTH = 10  # Visual tokens
LORA_R = 32
LORA_ALPHA = 64

# Generation
NUM_BEAMS = 4
MAX_LENGTH = 50
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{multimodal-captioning-2024,
  title={ImageSynth Captions: A Unified Framework for Visual to Textual Generation},
  author={Zhenghan Tai, Yaqian Xu, Yilin Huai, Xiangyu Liu},
  year={2025},
  publisher={GitHub},
  url={https://github.com/winfredtai/imageSythn}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flickr8k dataset creators
- Hugging Face for model implementations
- OpenAI for CLIP model
- Alibaba Cloud for Qwen models
