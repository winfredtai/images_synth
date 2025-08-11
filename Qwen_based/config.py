# Model Identifiers
CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
LLM_MODEL_ID = "Qwen/Qwen2.5-3B"
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# Dataset
# DATASET_PATH = "/root/autodl-tmp/dir_tzh/ECE1508S25/repo/imageSythn/datasets/captions_subset_100.json"
DATASET_PATH = "/root/autodl-tmp/dir_tzh/ECE1508S25/repo/imageSythn/datasets/captions_train_n_val.json"
IMAGE_DIR = "/root/autodl-tmp/dir_tzh/ECE1508S25/repo/imageSythn/datasets/Images"

# Training Parameters
LEARNING_RATE = 5e-5  
BATCH_SIZE = 8
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01

# Model Architecture
PREFIX_LENGTH = 10  # Length of the visual prefix sequence, we found 10 to be optimal
PROJECTION_DIM = None  # Dynamically set to the LLM's embedding dimension, get through model.embed_tokens.embedding_dim

# LoRA Configuration
USE_LORA = True
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
# LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# Additional Training Parameters
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SAVE_STEPS = 500
LOGGING_STEPS = 10

# Generation Parameters
# In this project, we used beam search generation only
MAX_LENGTH = 50
NUM_BEAMS = 4 # if beam search generation
TEMPERATURE = 1.0 # if sampling based generation
TOP_P = 0.9

# Paths
OUTPUT_DIR = "./outputs"
CHECKPOINT_DIR = "./checkpoints" # q,v proj
