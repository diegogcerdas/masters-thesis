import torch

from model.brain_encoder import EncoderModule
from model.diffusion.stable_diffusion import StableDiffusion


class GradientDescent(torch.nn.Module):
    def __init__(
        self,
        ldm: StableDiffusion,
        brain_encoder: EncoderModule,
        condition: torch.Tensor,
        loss_scale: float,
        g: float,
        seed: int,
        steps: int,
        learning_rate: float,
    ):
        super().__init__()
        self.ldm = ldm
        self.brain_encoder = brain_encoder
        self.loss_scale = loss_scale
        self.g = g
        self.seed = seed
        self.steps = steps
        self.learning_rate = learning_rate

        # Construct conditional and unconditional embeddings
        self.max_length = condition.shape[1]
        self.set_parameters(condition)

        # Construct optimizer
        self.optimizer = torch.optim.Adam([self.condition], lr=learning_rate)

    def set_parameters(self, condition):
        self.condition = torch.nn.Parameter(condition)
        self.condition.requires_grad = True

    def forward(self):
        uncondition = self.ldm.text_enc([""], self.max_length)
        text_embedding = torch.cat([uncondition, self.condition])

        latents = self.ldm.text_emb_to_img(
            text_embedding=text_embedding,
            return_pil=False,
            g=self.g,
            seed=self.seed,
            steps=self.steps,
        )

        image = self.ldm.latents_to_image(latents, return_pil=False)
        score = self.brain_encoder(image, no_grad=False).mean()
        return score * self.loss_scale, latents
