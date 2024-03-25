import torch
from diffusers import StableDiffusionPipeline
from xmap_utils import (
    cross_attn_init,
    register_cross_attention_hook,
    attn_maps,
    get_net_attn_map,
    resize_net_attn_map,
    save_net_attn_map,
)

cross_attn_init()

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.load_lora_weights('./outputs/cat_max_lora/final', weight_name="pytorch_lora_weights.safetensors")
pipe.unet = register_cross_attention_hook(pipe.unet)
pipe = pipe.to("cuda")

prompt = "a picture of a cat"
image = pipe(prompt).images[0]
image.save('test.png')

dir_name = "attn_maps"
net_attn_maps = get_net_attn_map(image.size)
net_attn_maps = resize_net_attn_map(net_attn_maps, image.size)
save_net_attn_map(net_attn_maps, dir_name, pipe.tokenizer, prompt)