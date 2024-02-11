import torch
from model.diffusion.stable_diffusion import StableDiffusion
from model.brain_encoder import EncoderModule

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
        self.condition_row = condition[:, 0, :]
        self.condition = torch.nn.Parameter(condition[:, 1:, :])
        self.uncondition = torch.nn.Parameter(ldm.text_enc([""], condition.shape[1])[:, 1:, :])
        self.condition.requires_grad = True
        self.uncondition.requires_grad = True

        # Construct optimizer
        self.optimizer = torch.optim.Adam([self.condition, self.uncondition], lr=learning_rate)

    def get_text_embedding(self):
        cond = torch.cat((self.condition_row.unsqueeze(dim=1), self.condition), dim=1)
        uncond = torch.cat((self.condition_row.unsqueeze(dim=1), self.uncondition), dim=1)
        return torch.cat([uncond, cond])

    def forward(self):
        latents = self.ldm.text_emb_to_img(
            text_embedding=self.get_text_embedding(),
            return_pil=False,
            g=self.g,
            seed=self.seed,
            steps=self.steps,
        )

        image = self.ldm.latents_to_image(latents, return_pil=False)
        score = self.brain_encoder(image, no_grad=False).mean()
        return score * self.loss_scale, latents

    