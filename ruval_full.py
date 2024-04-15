from rival_run import run
import torch


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        num_ddim_steps=50,
        invert_steps=50,
        guidance_scale=7,
        t_early=600,
        t_align=600,
        atten_frames=2,
        image_path="plane.png",
        prompt_original="a photo of a plane",
        prompt_edit=" a photo of a cat",
        num_images=3,
        editing_early_steps=900,
        outputs_dir='./',
        seed=0,
        device=device,
    )