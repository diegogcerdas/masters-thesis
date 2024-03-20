import os
import torch
from freecontrol_scheduler import CustomDDIMScheduler
from freecontrol_pipeline import FreeControlSDPipeline

def main(
    args_pretrained_model_name_or_path: str,
    args_prompt: str,
    args_negative_prompt: str,
    args_num_inference_steps: int,
    args_num_batch: int,
    args_num_save_basis: int,
    args_num_save_steps: int,
    args_output_path: str,
    args_seed: int,
    args_device: str,
):
    pipeline = FreeControlSDPipeline.from_pretrained(args_pretrained_model_name_or_path).to(args_device)
    pipeline.scheduler = CustomDDIMScheduler.from_pretrained(args_pretrained_model_name_or_path, subfolder="scheduler")

    generator = torch.Generator(device=args_device).manual_seed(args_seed)
    pipeline.sample_semantic_bases(
        prompt=args_prompt,
        negative_prompt=args_negative_prompt,
        num_inference_steps=args_num_inference_steps,
        num_batch=args_num_batch,
        num_save_basis=args_num_save_basis,
        num_save_steps=args_num_save_steps,
        generator=generator,
    )

    os.makedirs(args_output_path, exist_ok=True)
    torch.save(pipeline.pca_info, f"{args_output_path}/pca_info.pt")

if __name__ == "__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    main(
        args_pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        args_prompt="a photo of a dog",
        args_negative_prompt="",
        args_num_inference_steps=100,
        args_num_batch=1,
        args_num_save_basis=64,
        args_num_save_steps=120,
        args_output_path="freecontrol_output",
        args_seed=42,
        args_device=device,
    )