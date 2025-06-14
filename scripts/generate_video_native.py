#!/usr/bin/env python3

import torch
import sys
import os
from pathlib import Path
import imageio
import gc
import yaml

def main():
    print("🔧 Starting inference using ltxv_trainer framework...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    # Add path (same as your training script)
    ltx_path = "/workspace/Vision_NLP/src"
    sys.path.insert(0, ltx_path)
    print(f"Added to Python path: {ltx_path}")
    
    try:
        # Import exactly like your training script
        from ltxv_trainer.config import LtxvTrainerConfig
        from ltxv_trainer.trainer import LtxvTrainer
        print("✅ Successfully imported ltxv_trainer modules")
        
        # Load your training config (same as your training script)
        config_path = "/workspace/Vision_NLP/configs/ltxv_2b_lora.yaml"
        print(f"Loading config from: {config_path}")
        
        with open(config_path, "r") as file:
            config_data = yaml.safe_load(file)
        
        # Override config for inference (use checkpoint)
        config_data["model"]["load_checkpoint"] = "/workspace/Vision_NLP/outputs/ltxv_lora/checkpoints/lora_weights_step_00187.safetensors"
        
        # Convert to LtxvTrainerConfig (same as your training script)
        trainer_config = LtxvTrainerConfig(**config_data)
        print("✅ Config loaded successfully")
        
        # Initialize trainer (same as your training script)
        print("Initializing trainer...")
        trainer = LtxvTrainer(trainer_config)
        print("✅ Trainer initialized with LoRA checkpoint")
        
        print(f"GPU memory after trainer init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Check if trainer has any inference-related attributes
        print("\n🔍 Checking trainer components...")
        
        # Look for pipeline or model components
        components_found = []
        for attr_name in ['pipeline', 'model', 'transformer', 'vae', 'text_encoder', 'tokenizer', 'scheduler']:
            if hasattr(trainer, attr_name):
                attr = getattr(trainer, attr_name)
                if attr is not None:
                    components_found.append(attr_name)
                    print(f"✅ Found {attr_name}: {type(attr)}")
        
        if not components_found:
            print("❌ No usable components found in trainer")
            return
        
        # Try to create an inference pipeline from trainer components
        print("\n🚀 Creating inference pipeline from trainer components...")
        
        # Get the models from trainer
        transformer = getattr(trainer, 'transformer', None) or getattr(trainer, 'model', None)
        vae = getattr(trainer, 'vae', None)
        text_encoder = getattr(trainer, 'text_encoder', None)
        tokenizer = getattr(trainer, 'tokenizer', None)
        scheduler = getattr(trainer, 'scheduler', None)
        
        if transformer is None:
            print("❌ No transformer found in trainer")
            return
        
        print(f"✅ Using transformer: {type(transformer)}")
        
        # Now use DiffusionPipeline but replace components
        from diffusers import DiffusionPipeline
        
        print("Loading base pipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            "Lightricks/LTX-Video", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print("Replacing pipeline components with trainer components...")
        
        # Replace transformer with the trained one (with LoRA)
        pipe.transformer = transformer
        print("✅ Replaced transformer")
        
        # Replace other components if available from trainer
        if vae is not None:
            pipe.vae = vae
            print("✅ Replaced VAE")
        
        if text_encoder is not None:
            pipe.text_encoder = text_encoder
            print("✅ Replaced text_encoder")
        
        if tokenizer is not None:
            pipe.tokenizer = tokenizer
            print("✅ Replaced tokenizer")
        
        if scheduler is not None:
            pipe.scheduler = scheduler
            print("✅ Replaced scheduler")
        
        # Enable memory optimizations
        print("Enabling memory optimizations...")
        try:
            pipe.enable_model_cpu_offload()
            print("✅ CPU offload enabled")
        except Exception as e:
            print(f"⚠️ CPU offload failed: {e}")
        
        # Generate video
        print("\n🎬 Generating video with trained model...")
        test_prompt = "a news anchor discussing current events"
        print(f"Prompt: {test_prompt}")
        
        # Very conservative parameters
        generation_params = {
            "prompt": test_prompt,
            "height": 256,
            "width": 256,
            "num_frames": 9,
            "num_inference_steps": 10,
            "guidance_scale": 2.5,
        }
        
        print(f"Generation parameters: {generation_params}")
        
        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory before generation: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # Generate
        with torch.no_grad():
            result = pipe(**generation_params)
        
        print("✅ Generation completed!")
        
        # Extract video
        if hasattr(result, 'frames'):
            video = result.frames[0]
        elif hasattr(result, 'videos'):
            video = result.videos[0]
        else:
            video = result
        
        print(f"Video shape: {len(video) if hasattr(video, '__len__') else 'unknown'} frames")
        
        # Save video
        output_path = "ltxv_trainer_inference.mp4"
        imageio.mimsave(output_path, video, fps=8)
        print(f"✅ Video saved as '{output_path}'")
        
        if torch.cuda.is_available():
            print(f"Final GPU memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try just base model as last resort
        print("\n�� Trying base model as fallback...")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            from diffusers import DiffusionPipeline
            
            pipe = DiffusionPipeline.from_pretrained(
                "Lightricks/LTX-Video", 
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            
            pipe.enable_model_cpu_offload()
            
            with torch.no_grad():
                result = pipe(
                    prompt="news anchor",
                    height=128,
                    width=128,
                    num_frames=5,
                    num_inference_steps=5,
                    guidance_scale=1.5
                )
            
            video = result.frames[0] if hasattr(result, 'frames') else result.videos[0]
            imageio.mimsave("base_model_only.mp4", video, fps=4)
            print("✅ Base model video generated!")
            
        except Exception as final_e:
            print(f"❌ Even base model failed: {final_e}")
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print("🧹 Memory cleaned")
    
    print("\n✅ Script completed!")

if __name__ == "__main__":
    main()
