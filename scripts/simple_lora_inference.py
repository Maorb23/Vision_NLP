#!/usr/bin/env python3
import torch
from diffusers import LTXPipeline, AutoModel
from peft import PeftModel
from diffusers.utils import export_to_video

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) load base transformer
    base_transformer = AutoModel.from_pretrained(
        "Lightricks/LTX-Video",
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    # 2) load your LoRA adapter on top of it
    peft_model = PeftModel.from_pretrained(
        base_transformer,
        "outputs/ltxv_lora",
        torch_dtype=torch.bfloat16
    )

    # 3) merge the adapter into the base weights (no more mismatches!)
    merged = peft_model.merge_and_unload()
    merged = merged.to(device)

    # 4) build the pipeline with the merged transformer
    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video",
        transformer=merged,
        torch_dtype=torch.bfloat16
    ).to(device)
    pipe.enable_model_cpu_offload()

    # 5) generate a tiny test video
    out = pipe(
      prompt="a news anchor discussing current events in an outdoor setting",
      negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
      width=256, height=256,
      num_frames=9, num_inference_steps=20,
      guidance_scale=2.5
    ).frames[0]

    export_to_video(out, "lora_news_video.mp4", fps=8)
    print("✅ done!")

if __name__=="__main__":
    main()
