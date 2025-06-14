#!/usr/bin/env python3

import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.hooks import apply_group_offloading
from diffusers.utils import export_to_video

def main():
    print("🔧 Loading LTX-Video with your trained transformer...")
    
    # Load YOUR trained transformer (not the base one)
    print("Loading your trained transformer...")
    
    # Try to load your trained transformer from HuggingFace
    try:
        transformer = AutoModel.from_pretrained(
            "Maorb23/ltxv-news-lora",  # Your trained model
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
        print("✅ Loaded your trained transformer from HuggingFace")
    except:
        # Fallback: Load base transformer
        print("⚠️ Couldn't load from HuggingFace, using base transformer...")
        transformer = AutoModel.from_pretrained(
            "Lightricks/LTX-Video",
            subfolder="transformer",
            torch_dtype=torch.bfloat16
        )
    
    # Enable layerwise casting (same as their script)
    transformer.enable_layerwise_casting(
        storage_dtype=torch.float8_e4m3fn, 
        compute_dtype=torch.bfloat16
    )
    
    # Create pipeline with your transformer (same as their script)
    pipeline = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", 
        transformer=transformer, 
        torch_dtype=torch.bfloat16
    )
    
    # Group offloading (same as their script)
    print("Enabling memory optimizations...")
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    
    pipeline.transformer.enable_group_offload(
        onload_device=onload_device, 
        offload_device=offload_device, 
        offload_type="leaf_level", 
        use_stream=True
    )
    
    apply_group_offloading(
        pipeline.text_encoder, 
        onload_device=onload_device, 
        offload_type="block_level", 
        num_blocks_per_group=2
    )
    
    apply_group_offloading(
        pipeline.vae, 
        onload_device=onload_device, 
        offload_type="leaf_level"
    )
    
    print("✅ Memory optimizations enabled")
    
    # Generate video with news prompt
    print("\n🎬 Generating video...")
    
    prompt = "a news anchor discussing current events in an outdoor setting with greenery in the background"
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    
    print(f"Prompt: {prompt}")
    
    # Conservative parameters for your GPU
    video = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=512,          # Reduced from 768
        height=384,         # Reduced from 512  
        num_frames=25,      # Reduced from 161
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        num_inference_steps=30,  # Reduced from 50
    ).frames[0]
    
    # Save video (same as their script)
    export_to_video(video, "news_video_output.mp4", fps=24)
    print("✅ Video saved as 'news_video_output.mp4'")

if __name__ == "__main__":
    main()
