from diffusers import StableUnCLIPImg2ImgPipeline
import torch
import numpy as np
from PIL import Image
from methods.img_utils import save_images
import argparse
from methods.ddim_inversion import ddim_inversion

def main(args):

    # Since we use stabilityai/stable-diffusion-2-1-unclip, we fix the clip model to ViT-H-14 (laion2b_s32b_b79k)
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-unclip"
    ddim_pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1"
    resolution = (768,768)

    # Load unCLIP pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path).to(args.device)
    dtype = next(pipe.image_encoder.parameters()).dtype
    pipe.safety_checker = None  

    # Load source image and perform DDIM inversion
    source_img = Image.open(args.img_path).convert('RGB').resize(resolution)
    inverted_latents, _ = None if args.ddim_inversion_num_steps == 0 else ddim_inversion(
        pretrained_model_name_or_path=ddim_pretrained_model_name_or_path,
        image=source_img,
        num_inference_steps=args.ddim_inversion_num_steps,
        prompt=args.prompt,
        guidance_scale=1,
        seed=args.seed,
        device=args.device,
    )

    # Get CLIP vision embeddings
    source_img = pipe.feature_extractor(images=source_img, return_tensors="pt").pixel_values.to(device=args.device, dtype=dtype)
    source_img_embeds = pipe.image_encoder(source_img).image_embeds

    # Load shift vector
    direction_vector = torch.from_numpy(np.load(args.shift_vector_path)).to(args.device, dtype=dtype)
    mults = np.linspace(args.low_multiplier, args.high_multiplier, args.num_frames)

    images = []
    for mult in mults:

        # Modify image embeddings
        emb = source_img_embeds + mult * direction_vector

        # Generate image
        img = pipe(
            latents=inverted_latents,
            prompt=args.prompt,
            generator=torch.Generator(device=args.device).manual_seed(args.seed),
            image_embeds=emb,
            noise_level=args.unclip_noise_level,
        ).images[0]
        images.append(img)

    save_images(images, args.output_dir, names=[f"{mult:.2f}" for mult in mults])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default='.data/source_images/person.png')
    parser.add_argument("--output_dir", type=str, default='.data/outputs/person/indoor-outdoor')
    parser.add_argument("--prompt", type=str, default='a photo of a person')
    parser.add_argument("--shift_vector_path", type=str, default='./data/shift_vectors/attribute_pairs/indoor-outdoor.npy')
    parser.add_argument("--low_multiplier", type=float, default=0)
    parser.add_argument("--high_multiplier", type=float, default=20)
    parser.add_argument("--num_frames", type=int, default=100)
    parser.add_argument("--unclip_noise_level", type=float, default=0)
    parser.add_argument("--ddim_inversion_num_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )

    # Parse arguments
    args = parser.parse_args()
    main(args)
