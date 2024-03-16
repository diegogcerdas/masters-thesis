import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch.nn.functional as F
import bisect
import lpips

  
@torch.no_grad()
def get_text_embeddings(tokenizer, text_encoder, prompt):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids.cuda())[0]
    return text_embeddings

@torch.no_grad()
def image2latent(vae, image, resolution, device):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    image = transform(image).unsqueeze(0).to(device)
    latents = vae.encode(image)['latent_dist'].mean
    latents = latents * 0.18215
    return latents

@torch.no_grad()
def ddim_inversion(unet, scheduler, latent, cond):
    timesteps = reversed(scheduler.timesteps)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        for i, t in enumerate(tqdm(timesteps, desc="DDIM inversion")):
            cond_batch = cond.repeat(latent.shape[0], 1, 1)
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                scheduler.alphas_cumprod[timesteps[i - 1]]
                if i > 0 else scheduler.final_alpha_cumprod
            )
            mu = alpha_prod_t ** 0.5
            mu_prev = alpha_prod_t_prev ** 0.5
            sigma = (1 - alpha_prod_t) ** 0.5
            sigma_prev = (1 - alpha_prod_t_prev) ** 0.5
            eps = unet(latent, t, encoder_hidden_states=cond_batch).sample
            pred_x0 = (latent - sigma_prev * eps) / mu_prev
            latent = mu * pred_x0 + sigma * eps
    return latent

@torch.no_grad()
def cal_image(
    pipeline,
    num_inference_steps, 
    img_noise_0, 
    img_noise_1, 
    text_embeddings_0, 
    text_embeddings_1, 
    alpha, 
    use_adain,
):
    latents = slerp(img_noise_0, img_noise_1, alpha, use_adain)
    text_embeddings = (1 - alpha) * text_embeddings_0 + alpha * text_embeddings_1
    pipeline.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(pipeline.scheduler.timesteps, desc=f"DDIM Sampler, alpha={alpha}"):
        # predict the noise
        noise_pred = pipeline.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        # compute the previous noise sample x_t -> x_t-1
        latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    image = latent2image(latents)
    image = Image.fromarray(image)
    return image

@torch.no_grad()
def latent2image(self, latents, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = self.vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    elif return_type == "pt":
        image = (image / 2 + 0.5).clamp(0, 1)
    return image

@torch.no_grad()
def slerp(p0, p1, fract_mixing: float, adain=True):
    r""" Copied from lunarring/latentblending
    Helper function to correctly mix two random variables using spherical interpolation.
    The function will always cast up to float64 for sake of extra 4.
    Args:
        p0: 
            First tensor for interpolation
        p1: 
            Second tensor for interpolation
        fract_mixing: float 
            Mixing coefficient of interval [0, 1]. 
            0 will return in p0
            1 will return in p1
            0.x will return a mix between both preserving angular velocity.
    """
    if p0.dtype == torch.float16:
        recast_to = 'fp16'
    else:
        recast_to = 'fp32'
    p0 = p0.double()
    p1 = p1.double()
    if adain:
        mean1, std1 = calc_mean_std(p0)
        mean2, std2 = calc_mean_std(p1)
        mean = mean1 * (1 - fract_mixing) + mean2 * fract_mixing
        std = std1 * (1 - fract_mixing) + std2 * fract_mixing
    norm = torch.linalg.norm(p0) * torch.linalg.norm(p1)
    epsilon = 1e-7
    dot = torch.sum(p0 * p1) / norm
    dot = dot.clamp(-1+epsilon, 1-epsilon)
    theta_0 = torch.arccos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta_t = theta_0 * fract_mixing
    s0 = torch.sin(theta_0 - theta_t) / sin_theta_0
    s1 = torch.sin(theta_t) / sin_theta_0
    interp = p0*s0 + p1*s1
    if adain:
        interp = F.instance_norm(interp) * std + mean
    if recast_to == 'fp16':
        interp = interp.half()
    elif recast_to == 'fp32':
        interp = interp.float()
    return interp

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    if len(size) == 3:
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    else:
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def distance(img_a, img_b):
    return lpips.LPIPS()(img_a, img_b).item()

class AlphaScheduler:
    def __init__(self):
        ...

    def from_imgs(self, imgs):
        self.__num_values = len(imgs)
        self.__values = [0]
        for i in range(self.__num_values - 1):
            dis = distance(imgs[i], imgs[i + 1])
            self.__values.append(dis)
            self.__values[i + 1] += self.__values[i]
        for i in range(self.__num_values):
            self.__values[i] /= self.__values[-1]

    def save(self, filename):
        torch.save(torch.tensor(self.__values), filename)

    def load(self, filename):
        self.__values = torch.load(filename).tolist()
        self.__num_values = len(self.__values)

    def get_x(self, y):
        assert y >= 0 and y <= 1
        id = bisect.bisect_left(self.__values, y)
        id -= 1
        if id < 0:
            id = 0
        yl = self.__values[id]
        yr = self.__values[id + 1]
        xl = id * (1 / (self.__num_values - 1))
        xr = (id + 1) * (1 / (self.__num_values - 1))
        x = (y - yl) / (yr - yl) * (xr - xl) + xl
        return x

    def get_list(self, len=None):
        if len is None:
            len = self.__num_values

        ys = torch.linspace(0, 1, len)
        res = [self.get_x(y) for y in ys]
        return res