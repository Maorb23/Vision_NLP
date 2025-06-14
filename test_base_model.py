import torch
from diffusers import DiffusionPipeline
import imageio

# Clear memory
torch.cuda.empty_cache()

print("Loading pipeline with CPU offload...")
pipe = DiffusionPipeline.from_pretrained(
    "Lightricks/LTX-Video", 
    torch_dtype=torch.bfloat16
)

# Enable CPU offload immediately
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()

print("Generating minimal video...")
result = pipe(
    prompt="news anchor",
    height=128,
    width=128, 
    num_frames=5,
    num_inference_steps=5,
    guidance_scale=1.5
)

video = result.frames[0] if hasattr(result, 'frames') else result.videos[0]
imageio.mimsave("base_test.mp4", video, fps=4)
print("✅ Done!")
