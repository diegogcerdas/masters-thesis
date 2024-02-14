import torch

from model.diffusion.stable_diffusion import StableDiffusion


class GradientDescent(torch.nn.Module):
    def __init__(
        self,
        ldm: StableDiffusion,
        condition: torch.Tensor,
        num_tokens: int,
        learning_rate: float,
    ):
        super().__init__()
        self.ldm = ldm
        self.num_tokens = num_tokens
        self.learning_rate = learning_rate
        self.max_length = condition.shape[1]
        self.condition = torch.nn.Parameter(condition)
        self.condition.requires_grad = True
        self.optimizer = torch.optim.Adam([self.condition], lr=learning_rate)

    def get_text_embeddings(self):
        uncondition = self.ldm.text_enc([""], self.max_length)
        text_embedding = torch.cat([uncondition, self.condition])
        return text_embedding

    def forward(self):
        last_hidden_state = self.condition
        pooled_output = last_hidden_state.mean(dim=1)
        text_features = self.ldm.clip.text_projection(pooled_output)
        return text_features
