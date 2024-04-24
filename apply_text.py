from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
import torch
from PIL import Image
import numpy as np
import torch
import numpy as np
from PIL import Image
from utils.img_utils import save_images
import open_clip

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
seed = 0

img_f = 'burger.png'
output_dir = './outputs/burger'
mults = np.linspace(-2,2,11)
prompt = 'a photo of a burger'

pipe = StableUnCLIPImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip").to(device)
pipe.safety_checker = None  
dtype = next(pipe.image_encoder.parameters()).dtype

clip, _, _ = open_clip.create_model_and_transforms(model_name="ViT-H-14", pretrained="laion2b_s32b_b79k")
tokenizer = open_clip.get_tokenizer("ViT-H-14")

source_img = Image.open(img_f).convert('RGB').resize((768,768))

for word1, word2 in [('bright', 'dark'), ('big', 'small'), ('sweet', 'salty'), ('indoor', 'outdoor'), ('close', 'far'), ('less', 'more'), ('young', 'old')]:

    text = tokenizer([word1, word2])
    x = clip.encode_text(text)
    shift_vector = (x[0] - x[1]).to(device=device, dtype=dtype)

    # Get CLIP vision embeddings
    img_test = pipe.feature_extractor(images=source_img, return_tensors="pt").pixel_values
    img_test = img_test.to(device=device, dtype=dtype)
    img_test_embeds = pipe.image_encoder(img_test).image_embeds

    images = []
    for i, mult in enumerate(mults):
        generator = torch.Generator(device=device).manual_seed(seed)
        emb = img_test_embeds + mult * shift_vector
        img = pipe(
            prompt=prompt,
            generator=generator,
            image_embeds=emb,
            noise_level=0
        ).images[0]
        images.append(img)

    save_dir = f'{output_dir}/{word1}_{word2}'
    save_images(images, save_dir)
