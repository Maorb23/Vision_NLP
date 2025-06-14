import os
import torch
from diffusers import LTXPipeline, AutoModel
from diffusers.utils import export_to_video
from peft import PeftModel, PeftConfig
from huggingface_hub import snapshot_download

def load_and_merge_adapter(base_model: str, subfolder: str, adapter_dir: str, device: str):
    print("🔧 Loading base transformer...")
    base_transformer = AutoModel.from_pretrained(
        base_model, subfolder=subfolder, torch_dtype=torch.bfloat16
    )

    print("🔍 Inspecting adapter config:")
    cfg = PeftConfig.from_pretrained(adapter_dir)
    print("•", cfg)

    print("🔧 Loading LoRA adapter…")
    peft_model = PeftModel.from_pretrained(
        base_transformer, adapter_dir, torch_dtype=torch.bfloat16
    )

    sd = peft_model.state_dict()
    lora_keys = [k for k in sd if "lora_" in k]
    print(f"• Found {len(lora_keys)} LoRA keys (e.g. {lora_keys[:3]})")

    print("�� Merging LoRA into base weights…")
    merged = peft_model.merge_and_unload().to(device)
    return merged

def build_pipeline(base_model: str, merged_transformer, device: str):
    print("🔧 Building LTXPipeline…")
    pipe = LTXPipeline.from_pretrained(
        base_model, transformer=merged_transformer, torch_dtype=torch.bfloat16
    ).to(device)
    pipe.enable_model_cpu_offload()
    return pipe

def generate_and_save(pipe, prompt: str, neg_prompt: str, width: int, height: int,
                      num_frames: int, num_steps: int, guidance: float,
                      decode_timestep: float, decode_noise_scale: float,
                      output_path: str, fps: int):
    print(f"🎬 Generating {num_frames} frames @ {num_steps} steps…")
    video = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        width=width, height=height,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        decode_timestep=decode_timestep,
        decode_noise_scale=decode_noise_scale
    ).frames[0]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    export_to_video(video, output_path, fps=fps)
    print(f"✅ Saved video: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge HF LoRA adapter into LTX-Video and generate a video."
    )
    parser.add_argument(
        "--base_model",
        default="Lightricks/LTX-Video",
        help="Base LTX-Video model repo ID"
    )
    parser.add_argument(
        "--subfolder",
        default="transformer",
        help="Subfolder in base model for transformer"
    )
    parser.add_argument(
        "--adapter_repo",
        default="Maorb23/ltxv-news-lora",
        help="HF repo or local path containing adapter_config.json"
    )
    parser.add_argument(
        "--prompt",
        default=(
            "A professional news anchor delivering live updates on current events "
            "in an indoor studio setting with realism and smooth motion"
        ),
        help="Positive text prompt"
    )
    parser.add_argument(
        "--negative_prompt",
        default="worst quality, blurry, jittery, distorted",
        help="Negative prompt"
    )
    parser.add_argument("--width", type=int, default=768, help="Frame width")
    parser.add_argument("--height", type=int, default=512, help="Frame height")
    parser.add_argument("--num_frames", type=int, default=50, help="Number of frames")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--decode_timestep",
        type=float,
        default=0.03,
        help="Decode timestep"
    )
    parser.add_argument(
        "--decode_noise_scale",
        type=float,
        default=0.025,
        help="Decode noise scale"
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Output video path"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Output frames per second"
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Download adapter if not a local dir
    if not os.path.isdir(args.adapter_repo):
        print("🔍 Downloading adapter from HF...")
        args.adapter_repo = snapshot_download(repo_id=args.adapter_repo)

    merged = load_and_merge_adapter(
        args.base_model, args.subfolder, args.adapter_repo, device
    )
    pipe = build_pipeline(args.base_model, merged, device)
    generate_and_save(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.width,
        args.height,
        args.num_frames,
        args.num_inference_steps,
        args.guidance_scale,
        args.decode_timestep,
        args.decode_noise_scale,
        args.output,
        args.fps
    )
