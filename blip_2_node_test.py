import subprocess
import sys

import numpy as np
import torch
from PIL import Image  # Add this line


# Freeze PIP modules
def packages(versions=False):
    import subprocess
    import sys
    return [( r.decode().split('==')[0] if not versions else r.decode() ) for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class BlipConcat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "max_length": ("INT", {
                    "default": 0, 
                    "min": 0, # minimum value
                    "max": 500, # maximum value
                    "step": 1 # slider's step
                }),
                "string_field": ("STRING", {
                    "multiline": True, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": "{{BLIP_TEXT}}"
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "blip"

    CATEGORY = "conditioning"

    def blip(self, clip, image, max_length, string_field):
        # Check if Transformers is installed and update it to the latest version from GitHub
        if 'transformers' not in packages():
            print("\033[34mBLIP-2:\033[0m Installing transformers...")
            subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', '--upgrade', 'git+https://github.com/huggingface/transformers.git'])
        
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        image = tensor2pil(image).resize((596, 437))
        
        # Load model
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
        
        # Move model to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Encode text
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)

        print("\033[34mBLIP-2:\033[0m Generating text...")

        # Generate text
        generated_ids = model.generate(**inputs, max_new_tokens=max_length)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        print(generated_text)

        prompt = 'photo of a dining room with a table and chairs'

        return ([[clip.encode(prompt), {}]], )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeBLIP-2": BlipConcat
}