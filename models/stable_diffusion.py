import torch
from diffusers import (AutoencoderKL, DDPMScheduler,
                       DPMSolverMultistepScheduler, UNet2DConditionModel)
from transformers import AutoTokenizer, CLIPTextModel


class StableDiffusion:
    def __init__(self, pretrained_model_name_or_path: str, device="cpu"):
        self.device = device
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        ).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet"
        ).to(self.device)

    def tokenize_prompt(self, prompt, tokenizer_max_length=None):
        if tokenizer_max_length is not None:
            max_length = tokenizer_max_length
        else:
            max_length = self.tokenizer.model_max_length
        text_inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        return text_inputs

    def encode_prompt(self, input_ids):
        text_input_ids = input_ids.to(self.text_encoder.device)
        prompt_embeds = self.text_encoder(
            text_input_ids,
            return_dict=False,
        )
        prompt_embeds = prompt_embeds[0]
        return prompt_embeds

    def sample(
        self,
        num_samples,
        pipeline,
        pipeline_args,
        device,
        seed,
    ):
        # If we were previously predicting a variance, we need the scheduler to ignore it
        scheduler_args = {}
        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type
            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"
            scheduler_args["variance_type"] = variance_type
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, **scheduler_args
        )
        pipeline = pipeline.to(device)
        pipeline.set_progress_bar_config(disable=True)

        # Run inference
        generator = torch.Generator(device=device).manual_seed(seed)
        images = []
        for _ in range(num_samples):
            with torch.cuda.amp.autocast():
                image = pipeline(**pipeline_args, generator=generator).images[0]
                images.append(image)
        return images
