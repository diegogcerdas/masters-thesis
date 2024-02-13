import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPProcessor, CLIPModel

"""
Adapted from Github repo: webis-de/arxiv23-prompt-embedding-manipulation
From Deckers, N., Peters, J. and Potthast, M. (2023). Manipulating Embeddings of Stable Diffusion Prompts. arxiv:2308.12059.
"""


class StableDiffusion:
    def __init__(self, batch_size: int, device="cpu"):
        self.device = device
        self.dtype = torch.float32
        self.scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", torch_dtype=self.dtype
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=self.dtype
        ).to(self.device)
        self.clip = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=self.dtype
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=self.dtype
        )
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=self.dtype
        )

        self.name = "openai-clip-vit-large-patch14"
        self.latent_shape = (batch_size, self.unet.in_channels, 64, 64)

    def text_enc(self, prompts, maxlen=None):
        """
        A function to take a texual prompt and convert it into embeddings
        """
        if maxlen is None:
            maxlen = self.tokenizer.model_max_length
        inp = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=maxlen,
            truncation=True,
            return_tensors="pt",
        )

        text_encoded = self.clip.text_model(inp.input_ids.to(self.device))[0].float()
        return text_encoded

    def latents_to_image(self, latents, return_pil=True):
        """
        Function to convert latents to images
        """
        latents = (1 / 0.18215) * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        if not return_pil:
            return image
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def text_emb_to_img(
        self, text_embedding, return_pil, initial_latents=None, g=7.5, seed=0, steps=500
    ):
        """
        Diffusion process to convert input to image
        """

        if seed:
            torch.manual_seed(seed)

        # Setting number of steps in scheduler
        self.scheduler.set_timesteps(steps)

        # Setting initial latents
        latents = (
            torch.randn(self.latent_shape)
            if initial_latents is None
            else torch.clone(initial_latents)
        )

        # Adding noise to the latents
        latents = latents.to(self.device).float() * self.scheduler.init_noise_sigma

        # Iterating through defined steps
        for i, ts in enumerate(tqdm(self.scheduler.timesteps)):
            # We need to scale the i/p latents to match the variance
            inp = self.scheduler.scale_model_input(torch.cat([latents] * 2), ts)

            # Predicting noise residual using U-Net
            if i < steps - 1:
                with torch.no_grad():
                    u, t = self.unet(
                        inp, ts, encoder_hidden_states=text_embedding
                    ).sample.chunk(2)
            else:
                u, t = self.unet(
                    inp, ts, encoder_hidden_states=text_embedding
                ).sample.chunk(2)

            # Performing Guidance
            pred = u + g * (t - u)

            # Conditioning  the latents
            latents = self.scheduler.step(pred, ts, latents).prev_sample

        if return_pil:
            return self.latents_to_image(latents, return_pil=True)
        return latents
