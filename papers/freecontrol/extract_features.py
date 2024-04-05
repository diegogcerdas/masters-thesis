import torch
from PIL import Image
import os
from papers.freecontrol.freecontrol_pipeline import FreeControlSDPipeline

pretrained_model_name_or_path = "stabilityai/stable-diffusion-2"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
image_folder = 'subset'
output_path = 'subset'

num_inference_steps = 50
num_save_steps = 50
seed = 0
feature_blocks = ['up_blocks.1']

pipeline = FreeControlSDPipeline.from_pretrained(pretrained_model_name_or_path).to(device)
generator = torch.Generator(device=device).manual_seed(seed)

filenames = [
    int(f.replace(".png", ""))
    for f in os.listdir(image_folder)
    if f.endswith(".png")
]
filenames = [
    os.path.join(image_folder, f"{f}.png") for f in sorted(filenames)
]
images = [
     Image.open(f).convert('RGB').resize((768, 768)) for f in filenames
]

features = pipeline.get_features(
    images,
    num_inference_steps,
    num_save_steps,
    feature_blocks,
    generator,
)

os.makedirs(output_path, exist_ok=True)
torch.save(features, f"{output_path}/features.pt")