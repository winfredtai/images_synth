import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType


class MappingNetwork(nn.Module):
    """
    MLP Network that projects CLIP embeddings into LLM embedding space
    """
    
    def __init__(self, clip_embedding_dim, llm_embedding_dim, prefix_length):
        super(MappingNetwork, self).__init__()
        
        self.prefix_length = prefix_length
        self.llm_embedding_dim = llm_embedding_dim
        
        # Two-layer MLP
        self.network = nn.Sequential(
            nn.Linear(clip_embedding_dim, llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, prefix_length * llm_embedding_dim)
        )
    
    def forward(self, x):
        output = self.network(x)
        
        # Reshape to [batch_size, prefix_length, llm_embedding_dim]
        visual_prefix = output.view(-1, self.prefix_length, self.llm_embedding_dim)
        
        return visual_prefix


class ImageCaptioningModel(nn.Module):
    """
    Main image captioning model
    """
    
    def __init__(self, config):
        super(ImageCaptioningModel, self).__init__()
        
        self.config = config
        self.device = config.DEVICE
        
        print("Loading CLIP vision model...")
        self.clip_vision_model = CLIPVisionModel.from_pretrained(config.CLIP_MODEL_ID)
        self.clip_vision_model = self.clip_vision_model.to(self.device)
        self.clip_vision_model.eval()
        
        # Freeze CLIP parameters
        for param in self.clip_vision_model.parameters():
            param.requires_grad = False
        
        print("Loading Qwen language model...")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Get LLM's embedding dimension to set up the mapping network
        llm_embedding_dim = self.llm.model.embed_tokens.embedding_dim
        config.PROJECTION_DIM = llm_embedding_dim
        clip_embedding_dim = self.clip_vision_model.config.hidden_size
        
        # MLP
        self.mapping_network = MappingNetwork(
            clip_embedding_dim=clip_embedding_dim,
            llm_embedding_dim=llm_embedding_dim,
            prefix_length=config.PREFIX_LENGTH
        )

        self.mapping_network = self.mapping_network.to(self.device)
        
        # Apply LoRA
        if config.USE_LORA:
            print("Applying LoRA to language model...")
            lora_config = LoraConfig(
                r=config.LORA_R,
                lora_alpha=config.LORA_ALPHA,
                target_modules=config.LORA_TARGET_MODULES,
                lora_dropout=config.LORA_DROPOUT,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.llm = get_peft_model(self.llm, lora_config)
            self.llm.print_trainable_parameters()
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        batch_size = pixel_values.shape[0]
        
        # Get image embeddings from CLIP image encoder
        with torch.no_grad():
            clip_outputs = self.clip_vision_model(pixel_values=pixel_values)
            # Extract CLS token embedding, instead of pooler_output
            clip_embeddings = clip_outputs.last_hidden_state[:, 0, :]
        
        # Project image embeddings through mapping network
        visual_prefix = self.mapping_network(clip_embeddings)

        # Convert visual_prefix to match LLM dtype - BF16
        visual_prefix = visual_prefix.to(dtype=torch.bfloat16)
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # Combine visual and text embeddings
        inputs_embeds = torch.cat([visual_prefix, text_embeddings], dim=1)
        
        # Create combined attention mask
        visual_attention_mask = torch.ones(
            batch_size, self.config.PREFIX_LENGTH,
            dtype=attention_mask.dtype,
            device=self.device
        )
        combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
        
        # Prepare labels for loss calculation
        if labels is None:
            labels = input_ids.clone()
        
        # Create extended labels with -100 for visual prefix
        extended_labels = torch.full(
            (batch_size, self.config.PREFIX_LENGTH + labels.shape[1]),
            -100,
            dtype=labels.dtype,
            device=self.device
        )
        
        # Copy the caption labels after the visual prefix
        extended_labels[:, self.config.PREFIX_LENGTH:] = labels
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_attention_mask,
            labels=extended_labels,
            return_dict=True
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(self, pixel_values, max_length=50, num_beams=4, 
                 temperature=1.0, top_p=0.9, do_sample=False):

        # in this project, we use beam search generation only
        batch_size = pixel_values.shape[0]
        
        clip_outputs = self.clip_vision_model(pixel_values=pixel_values)
        clip_embeddings = clip_outputs.last_hidden_state[:, 0, :]
        visual_prefix = self.mapping_network(clip_embeddings)
        visual_prefix = visual_prefix.to(dtype=torch.bfloat16)
        
        # Use visual prefix as the only input embeddings
        inputs_embeds = visual_prefix
        
        attention_mask = torch.ones(
            batch_size, 
            visual_prefix.shape[1],
            device=self.device
        )
        
        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            # temperature=temperature,
            # top_p=top_p, # do beam search, not sampling
            do_sample=do_sample,
            pad_token_id=self.llm.tokenizer.pad_token_id,
            eos_token_id=self.llm.tokenizer.eos_token_id
        )
        
        # Decode generated tokens
        generated_tokens = outputs 
        captions = self.llm.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        return captions
