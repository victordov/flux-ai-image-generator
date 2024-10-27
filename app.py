import torch
from diffusers import FluxPipeline

# Check if MPS is available, else fall back to CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

# Move the model to MPS or CPU depending on availability
pipe.to(device)

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)  # CPU seed is fine for reproducibility
).images[0]

image.save("flux-dev.png")
