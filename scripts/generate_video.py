#!/usr/bin/env python3

import torch
import sys
import os
from pathlib import Path
import logging
import numpy as np
import imageio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("🔧 Starting video generation script...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        # Use diffusers pipeline directly (skip the individual model loading)
        print("\n🚀 Loading pipeline with diffusers...")
        from diffusers import DiffusionPipeline
        from peft import PeftModel
        
        # Load the pipeline without device_map="auto"
        print("Loading DiffusionPipeline...")
        pipe = DiffusionPipeline.from_pretrained(
            "Lightricks/LTX-Video", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        print(f"✅ Pipeline loaded: {type(pipe)}")
        print(f"Pipeline components: {list(pipe.components.keys()) if hasattr(pipe, 'components') else 'No components'}")
        
        # Check what model components are available
        model_attrs = []
        for attr in ['transformer', 'unet', 'denoiser']:
            if hasattr(pipe, attr):
                component = getattr(pipe, attr)
                if hasattr(component, 'parameters'):
                    model_attrs.append(attr)
                    print(f"Found model component: {attr} ({type(component)})")
        
        if not model_attrs:
            print("❌ No model components found for LoRA application")
            return
        
        # Apply LoRA to the first available model component
        model_attr = model_attrs[0]
        print(f"\n🎯 Applying LoRA to {model_attr}...")
        
        original_model = getattr(pipe, model_attr)
        
        try:
            # Try to load LoRA with strict=False to ignore missing keys
            lora_model = PeftModel.from_pretrained(
                original_model, 
                "Maorb23/ltxv-news-lora",
                is_trainable=False
            )
            setattr(pipe, model_attr, lora_model)
            print("✅ LoRA applied successfully!")
            
        except Exception as lora_e:
            print(f"❌ LoRA application failed: {lora_e}")
            print("Continuing without LoRA...")
        
        # Move pipeline to device manually
        print(f"\n🔧 Moving pipeline to {device}...")
        try:
            pipe = pipe.to(device)
            print(f"✅ Pipeline moved to {device}")
        except Exception as e:
            print(f"⚠️ Error moving pipeline: {e}")
            # Try moving components individually
            for component_name in pipe.components.keys():
                component = getattr(pipe, component_name)
                if hasattr(component, 'to') and hasattr(component, 'parameters'):
                    try:
                        setattr(pipe, component_name, component.to(device))
                        print(f"✅ Moved {component_name} to {device}")
                    except Exception as comp_e:
                        print(f"⚠️ Could not move {component_name}: {comp_e}")
        
        # Enable memory optimizations
        print("\n🔧 Enabling memory optimizations...")
        try:
            pipe.enable_model_cpu_offload()
            print("✅ CPU offload enabled")
        except Exception as e:
            print(f"⚠️ CPU offload failed: {e}")
        
        try:
            pipe.enable_vae_tiling()
            print("✅ VAE tiling enabled")
        except Exception as e:
            print(f"⚠️ VAE tiling failed: {e}")
        
        # Check memory status
        print("\n🧠 Memory status before generation:")
        if torch.cuda.is_available():
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"GPU memory free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9:.2f} GB")
        
        # Generate with very conservative parameters
        print("\n🎬 Generating test video...")
        test_prompt = "a news anchor discussing current events in an outdoor setting"
        print(f"Prompt: {test_prompt}")
        
        generation_params = {
            "prompt": test_prompt,
            "height": 256,  # Small size
            "width": 256,
            "num_frames": 9,   # Very few frames
            "guidance_scale": 2.5,  # Lower guidance
            "num_inference_steps": 10,  # Very few steps
        }
        
        if torch.cuda.is_available():
            generation_params["generator"] = torch.Generator(device=device).manual_seed(42)
        
        print(f"Generation parameters: {generation_params}")
        
        try:
            with torch.no_grad():
                result = pipe(**generation_params)
            
            print(f"✅ Generation complete! Result type: {type(result)}")
            
            # Extract video frames
            if hasattr(result, 'frames'):
                video = result.frames[0]
                print(f"✅ Extracted frames: {len(video)} frames")
            elif hasattr(result, 'videos'):
                video = result.videos[0]
                print(f"✅ Extracted videos: {len(video)} frames")
            else:
                video = result
                print(f"Using direct result: {type(video)}")
            
            # Save video
            output_path = "generated_news_video_test.mp4"
            imageio.mimsave(output_path, video, fps=8)
            print(f"✅ Video saved as '{output_path}'")
            
            print("\n🎉 Video generation successful!")
            
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()
            
            # Try with absolute minimal parameters
            print("\n🔄 Trying with absolute minimal parameters...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                minimal_params = {
                    "prompt": "news anchor",
                    "height": 128,
                    "width": 128,
                    "num_frames": 5,
                    "guidance_scale": 1.5,
                    "num_inference_steps": 5,
                }
                
                with torch.no_grad():
                    result = pipe(**minimal_params)
                
                if hasattr(result, 'frames'):
                    video = result.frames[0]
                elif hasattr(result, 'videos'):
                    video = result.videos[0]
                else:
                    video = result
                
                imageio.mimsave("minimal_test_video.mp4", video, fps=4)
                print("✅ Minimal test video generated!")
                
            except Exception as minimal_e:
                print(f"❌ Even minimal generation failed: {minimal_e}")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Script completed!")

if __name__ == "__main__":
    main()
