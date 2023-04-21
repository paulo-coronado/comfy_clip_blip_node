import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

# Add the ComfyUI directory to the system path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
sys.path.append('..'+os.sep+'ComfyUI')

NODE_FILE = os.path.abspath(__file__)
BLIP_NODE_ROOT = os.path.dirname(NODE_FILE)
MODELS_DIR = os.path.join(( os.getcwd()+os.sep+'ComfyUI' if not os.getcwd().startswith(('/content', '/workspace')) else os.getcwd() ), 'models')

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

def transformImage_legacy(input_image, image_size, device):
    raw_image = input_image.convert('RGB')   
    raw_image = raw_image.resize((image_size, image_size))
    transform = transforms.Compose([
        transforms.Resize(raw_image.size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image
            
def transformImage(input_image, image_size, device):
    raw_image = input_image.convert('RGB')   
    raw_image = raw_image.resize((image_size, image_size))
    transform = transforms.Compose([
        transforms.Resize(raw_image.size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image.view(1, -1, image_size, image_size)  # Change the shape of the output tensor

class BlipConcat:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "min_length": ("INT", {
                    "default": 5, 
                    "min": 0, # minimum value
                    "max": 200, # maximum value
                    "step": 1 # slider's step
                }),
                "max_length": ("INT", {
                    "default": 20, 
                    "min": 0, # minimum value
                    "max": 200, # maximum value
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

    def blip(self, clip, image, min_length, max_length, string_field):
        print(f"\033[34mStarting BLIP...\033[0m")


        # Change the current working directory to BLIP_NODE_ROOT
        os.chdir(BLIP_NODE_ROOT)

        # Add BLIP_NODE_ROOT to the Python path
        sys.path.insert(0, BLIP_NODE_ROOT)

        from models.blip import blip_decoder
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        image = tensor2pil(image)
        size = 384
        
        if 'transformers==4.26.1' in packages(True):
            print("Using Legacy `transformImaage()`")
            tensor = transformImage_legacy(image, size, device)
        else:
            tensor = transformImage(image, size, device)
                
        blip_dir = os.path.join(MODELS_DIR, 'blip')
        if not os.path.exists(blip_dir):
            os.mkdir(blip_dir)
            
        torch.hub.set_dir(blip_dir)
    
        model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'

        model = blip_decoder(pretrained=model_url, image_size=size, vit='base')
        model.eval()
        model = model.to(device)
        
        with torch.no_grad():
            caption = model.generate(tensor, sample=False, num_beams=1, min_length=min_length, max_length=max_length) 
            text = string_field.replace('BLIP_TEXT', caption[0])
            print(f"\033[34mPrompt:\033[0m", text)
            return ([[clip.encode(text), {}]], )

NODE_CLASS_MAPPINGS = { "CLIPTextEncodeBLIP": BlipConcat }