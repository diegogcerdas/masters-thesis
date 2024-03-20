from typing import Any, Dict, List, Optional, Union
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, DDIMScheduler
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from freecontrol_utils import prep_unet_conv, prep_unet_attention, get_self_attn_feat


class FreeControlSDPipeline(StableDiffusionPipeline):

    def prepare_image_latents(self, image, dtype, device, generator=None):
        image = image.to(device=device, dtype=dtype)
        latents = self.vae.encode(image).latent_dist.sample(generator)
        latents = self.vae.config.scaling_factor * latents
        latents = torch.cat([latents], dim=0)
        return latents

    @torch.no_grad()
    def ddim_inversion(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        # 1. Define call parameters
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0
        self.inverse_scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)

        # 2. Preprocess image
        image = self.image_processor.preprocess(image)

        # 3. Prepare latent variables
        latents = self.prepare_image_latents(image, self.vae.dtype, device, generator)

        # 4. Encode input prompt
        num_images_per_prompt = 1
        prompt_embeds, _ = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
        )

        # 6. Prepare timesteps
        self.inverse_scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.inverse_scheduler.timesteps

        # 7. Denoising loop
        for t in tqdm(timesteps):

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.inverse_scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inverse_scheduler.step(noise_pred, t, latents).prev_sample

        inverted_latents = latents.detach().clone()

        return inverted_latents, prompt_embeds
    
    @torch.no_grad()
    def ddim_sample(
        self,
        latents: torch.Tensor,
        num_inference_steps: int,
        text_embeddings: torch.Tensor,
    ):
        device = self._execution_device
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in tqdm(self.scheduler.timesteps):
            # predict the noise
            noise_pred = self.unet(latents, t, encoder_hidden_states=text_embeddings).sample
            # compute the previous noise sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        image = self.latent2image(latents)
        image = Image.fromarray(image)
        return image
    
    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        elif return_type == "pt":
            image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def sample_semantic_bases(
        self,
        prompt: str = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,  
        num_batch: int = 1,
        num_save_basis: int = 64,
        num_save_steps: int = 120,     
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        pca_guidance_blocks: List[str] = ['up_blocks.1'],
    ):
        
        with torch.no_grad():

            # 0. Prepare the UNet
            self.unet = prep_unet_attention(self.unet)
            self.unet = prep_unet_conv(self.unet)
            self.pca_info: Dict = dict()

            # 0. Default height and width to unet
            height = self.unet.config.sample_size * self.vae_scale_factor
            width = self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(prompt, height, width, None, negative_prompt)

            # 2. Define call parameters
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            text_encoder_lora_scale = (
                cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
            )
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                1,
                do_classifier_free_guidance,
                negative_prompt,
                lora_scale=text_encoder_lora_scale,
            )
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps

            # 5. Prepare latent variables
            num_channels_latents = self.unet.config.in_channels
            all_latents = self.prepare_latents(
                num_batch,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
            )
            latent_list = list(all_latents.chunk(num_batch, dim=0))

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
            
            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps * num_batch) as progress_bar:
                for i, t in enumerate(timesteps):
                    if i >= num_save_steps:
                        break
                    # create dict to store the hidden features
                    attn_key_dict = dict()

                    for latent_id, latents in enumerate(latent_list):
                        # expand the latents if we are doing classifier free guidance
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        # predict the noise residual
                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                            return_dict=False,
                        )[0]

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                        # call the callback, if provided
                        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                            progress_bar.update()
                        latent_list[latent_id] = latents

                        # 8. Post-processing the pca features
                        hidden_state_dict, query_dict, key_dict, value_dict = get_self_attn_feat(self.unet, pca_guidance_blocks)
                        for name in hidden_state_dict.keys():
                            def log_to_dict(feat, selected_dict, name):
                                feat = feat.chunk(2)[1]
                                if name in selected_dict.keys():
                                    selected_dict[name].append(feat)
                                else:
                                    selected_dict[name] = [feat]

                            log_to_dict(key_dict[name], attn_key_dict, name)

                    def apply_pca(feat):
                        with torch.autocast(device_type='cuda', dtype=torch.float32):
                            feat = feat.contiguous().to(torch.float32)
                            # feat shape in [bs,channels,16,16]
                            bs, channels, h, w = feat.shape
                            if feat.ndim == 4:
                                X = feat.permute(0, 2, 3, 1).reshape(-1, channels).to('cuda')
                            else:
                                X = feat.permute(0, 2, 1).reshape(-1, channels).to('cuda')
                            # Computing PCA
                            mean = X.mean(dim=0)
                            tensor_centered = X - mean
                            U, S, V = torch.svd(tensor_centered)
                            n_egv = V.shape[-1]

                            if n_egv > num_save_basis and num_save_basis > 0:
                                V = V[:, :num_save_basis]
                            basis = V.T
                        assert mean.shape[-1] == basis.shape[-1]

                        return {
                            'mean': mean.cpu(),
                            'basis': basis.cpu(),
                        }

                    def process_feat_dict(feat_dict):
                        for name in feat_dict.keys():
                            feat_dict[name] = torch.cat(feat_dict[name], dim=0)
                            feat_dict[name] = apply_pca(feat_dict[name])

                    # Only process for the first num_save_steps
                    if i < num_save_steps:
                        process_feat_dict(attn_key_dict)
                        self.pca_info[i] = {'attn_key': attn_key_dict}