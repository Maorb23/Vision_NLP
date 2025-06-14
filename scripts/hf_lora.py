#!/usr/bin/env python3
# filepath: scripts/hf_lora_debug_confirm.py

import os
import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.utils import export_to_video
from peft import PeftModel, PeftConfig
from huggingface_hub import snapshot_download

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1) Load base transformer
    print("\n🔧 Loading base transformer...")
    base_transformer = AutoModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # 2) Snapshot your LoRA adapter repo locally
    adapter_repo = "Maorb23/ltxv-news-lora"
    print("\n🔍 Downloading adapter repo from HF...")
    snapshot_dir = snapshot_download(repo_id=adapter_repo)
    print(f"• snapshot_download → {snapshot_dir}")
    print("  contains:", os.listdir(snapshot_dir))

    # 3) Inspect adapter config
    print("\n🔍 Inspecting adapter_config.json:")
    cfg = PeftConfig.from_pretrained(snapshot_dir)
    print("• Adapter config:", cfg)

    # 4) Load the LoRA adapter from that local snapshot
    print("\n🔧 Loading LoRA adapter into PEFT model…")
    peft_model = PeftModel.from_pretrained(
        base_transformer,
        snapshot_dir,
        torch_dtype=torch.bfloat16
    )

    # 5) Confirm the adapter keys are actually in the state dict
    sd = peft_model.state_dict()
    lora_keys = [k for k in sd if "lora_" in k]
    print(f"• Found {len(lora_keys)} LoRA keys, e.g.")
    for k in lora_keys[:5]:
        print("   ", k)

    # 6) Merge & unload
    print("\n🔧 Merging LoRA into base weights…")
    merged = peft_model.merge_and_unload().to(device)

    # 7) Build pipeline & offload
    print("\n🔧 Building LTXPipeline…")
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        transformer=merged,
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.enable_model_cpu_offload()

    # 8) Generate a longer video
    print("\n🎬 Generating longer video (50f/50s)...")
    video = pipe(
        prompt="A news anchor giving live updates outdoors",
        negative_prompt="worst quality, blurry, jittery",
        width=768, height=512,
        num_frames=60,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        num_inference_steps=40,
        guidance_scale= 2.5
    ).frames[0]

    out_path = "lora_50_3.5.mp4"
    export_to_video(video, out_path, fps=24)
    print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
