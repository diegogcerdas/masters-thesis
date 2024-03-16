from diffmorpher import run
import torch, os

if __name__ == "__main__":

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    images, alpha_list = run(
        args_pretrained_model_name_or_path='runwayml/stable-diffusion-v1-5',
        args_img_path_0='./diffmorpher/data/0.png',
        args_img_path_1='./diffmorpher/data/1.png',
        args_load_lora_path_0='./diffmorpher/data/lora1.safetensors',
        args_load_lora_path_1='./diffmorpher/data/lora1.safetensors',
        args_prompt_0='an oil painting',
        args_prompt_1='an oil painting',
        args_num_inference_steps=50,
        args_num_frames=50,
        args_attn_beta=0,
        args_lamd=0.6,
        args_use_adain=True,
        args_use_reschedule=True,
        args_resolution=512,
        args_device=device,
    )

    out_dir = './diffmorpher/out'
    os.makedirs(out_dir, exist_ok=True)
    for img, alpha in zip(images, alpha_list):
        img.save(f'{out_dir}/{alpha:.2f}.png')