import os

import psutil
import requests
import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast


def log_memory_usage(stage):
    """Logs system memory usage."""
    memory = psutil.virtual_memory()
    print(
        f"{stage} - Memory usage: {memory.used / 1e9:.2f} GB (used), {memory.available / 1e9:.2f} GB (available)"
    )


def download_file(url, local_path):
    """
    Downloads a file from a URL to a local path if it doesn't already exist.
    """
    if not os.path.exists(local_path):
        print(f"File not found at {local_path}. Downloading...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(local_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded file to {local_path}")
    else:
        print(f"File already exists at {local_path}")


# Define the URL and local path
model_url = "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8-e4m3fn.safetensors"
local_model_path = "./models/kijai-flux-schnell-fp8.safetensors"

# Ensure the directory exists
os.makedirs(os.path.dirname(local_model_path), exist_ok=True)

# Check and download the file
download_file(model_url, local_model_path)

# Load components from the original model
model_id = "black-forest-labs/FLUX.1-schnell"  # Original model ID

# Load scheduler
log_memory_usage("Before loading scheduler")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler"
)
log_memory_usage("After loading scheduler")

# Load the VAE
log_memory_usage("Before loading VAE")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
log_memory_usage("After loading VAE")

# Load text encoders and tokenizers
log_memory_usage("Before loading text encoders")
text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder", low_cpu_mem_usage=True, device_map="auto"
)
tokenizer = CLIPTokenizer.from_pretrained(
    model_id, subfolder="tokenizer", add_prefix_space=False
)
text_encoder_2 = T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder_2", low_cpu_mem_usage=True, device_map="auto"
)
tokenizer_2 = T5TokenizerFast.from_pretrained(
    model_id, subfolder="tokenizer_2", add_prefix_space=False
)
log_memory_usage("After loading text encoders")

# Load the transformer model manually
print("Loading the transformer model...")
try:
    transformer = FluxTransformer2DModel.from_single_file(
        local_model_path,  # Path to your model
        torch_dtype=torch.float,
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        device_map="auto",  # Automatically map to CPU/GPU based on availability
    )
    print("Transformer loaded successfully!")
except Exception as e:
    print(f"Transformer loading error: {e}")
log_memory_usage("After loading transformer")

# Assemble the pipeline with the modified components
print("Assembling the pipeline...")
pipe = FluxPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    transformer=transformer,
)

# Enable optimizations for memory
pipe.enable_sequential_cpu_offload()  # Sequential offloading


def read_xml_to_string(file_path):
    with open(file_path, "r") as file:
        xml_string = file.read()
    return xml_string


# Generate an image
prompt = """
Make a top down image of a 10 feet by 12 feet bedroom in the Japandi style.
"""
seed = 42
generator = torch.manual_seed(seed)

print("Generating image...")
image = pipe(
    prompt,
    output_type="pil",
    num_inference_steps=4,  # Optimize for speed
    generator=generator,
).images[0]

# Save the output
output_path = "flux-quantized-output.png"
image.save(output_path)
print(f"Image saved to {output_path}")
