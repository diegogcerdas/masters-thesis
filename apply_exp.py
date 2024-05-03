from diffusers import StableUnCLIPImg2ImgPipeline
import torch
from PIL import Image
import numpy as np
import torch
import numpy as np
from PIL import Image
from utils.img_utils import save_images
import argparse
from methods.synthesis import ddim_inversion
from methods.brain_encoder import get_encoder

def main(args):

    # Since we use stabilityai/stable-diffusion-2-1-unclip, we fix the clip model to ViT-H-14 (laion2b_s32b_b79k)
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-unclip"
    feature_extractor_type = "clip_2_0"
    metric = 'cosine'

    # Load unCLIP pipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(pretrained_model_name_or_path).to(args.device)
    dtype = next(pipe.image_encoder.parameters()).dtype
    pipe.safety_checker = None  

    # Get CLIP vision embeddings
    source_img = pipe.feature_extractor(images=source_img, return_tensors="pt").pixel_values.to(device=args.device, dtype=dtype)
    source_img_embeds = pipe.image_encoder(source_img).image_embeds

    # Get brain encoder
    encoder = get_encoder(args.data_root, args.subject, args.roi, args.hemisphere, feature_extractor_type, metric, False, args.seed, args.device)

    # Load direction vector
    direction_vector_filename = f'./direction_vectors/{args.direction_vector_name}.npy'
    direction_vector = torch.from_numpy(np.load(direction_vector_filename)).to(args.device, dtype=dtype)
    mults = np.linspace(args.low_multiplier, args.high_multiplier, args.num_frames)

    for seed in np.random.randint(0,1000,args.num_variations):

        acts = []
        for mult in mults:

            # Modify image embeddings
            emb = source_img_embeds + mult * direction_vector

            # Generate image
            img = pipe(
                prompt=args.prompt,
                generator=torch.Generator(device=args.device).manual_seed(seed),
                image_embeds=emb,
                noise_level=args.unclip_noise_level,
            ).images[0]

            # Predict brain activation
            acts.append(encoder(img))

        acts = np.array(acts)

        save_dir = f'{args.output_dir}/{args.subject}_{args.roi}_{args.hemisphere}/{args.direction_vector_name}/exp'
        np.save(f'{save_dir}/{seed}_acts.npy', acts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model and Data Parameters
    parser.add_argument("--data_root", type=str, default="./data/")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--roi", default="PPA")
    parser.add_argument("--hemisphere", type=str, default="right")

    parser.add_argument("--img_filename", type=str, default='./source_images/person.png')
    parser.add_argument("--output_dir", type=str, default='./outputs/person')
    parser.add_argument("--prompt", type=str, default='a photo of a person')
    parser.add_argument("--num_variations", type=int, default=100)

    parser.add_argument("--direction_vector_name", type=str, default='1_lightness')
    parser.add_argument("--low_multiplier", type=float, default=-20)
    parser.add_argument("--high_multiplier", type=float, default=20)
    parser.add_argument("--num_frames", type=int, default=50)
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

    # WandB Parameters
    parser.add_argument("--wandb-project", type=str, default="masters-thesis")
    parser.add_argument("--wandb-entity", type=str, default="diego-gcerdas")
    parser.add_argument("--wandb-mode", type=str, default="online")

    # Parse arguments
    args = parser.parse_args()
    main(args)
