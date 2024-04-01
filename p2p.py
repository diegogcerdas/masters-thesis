

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2").to(device)
ldm_stable.load_lora_weights(
      'lora_min',
      weight_name="pytorch_lora_weights.safetensors",
      adapter_name="lora_min",
  )
ldm_stable.load_lora_weights(
      'lora_max',
      weight_name="pytorch_lora_weights.safetensors",
      adapter_name="lora_max",
  )
tokenizer = ldm_stable.tokenizer

prompts = [
    "a cat sitting next to a mirror",
    "a silver cat sculpture sitting next to a mirror"
]

cross_replace_steps = {'default_': .8, }
self_replace_steps = .6
blend_word = ((('cat',), ("cat",))) # for local edit
eq_params = {"words": ("silver", 'sculpture', ), "values": (2,2,)}  # amplify attention to the words "silver" and "sculpture" by *2 
 
controller = make_controller(prompts, False, cross_replace_steps, self_replace_steps, blend_word, eq_params)
images, _ = run_and_display(prompts, controller, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings)