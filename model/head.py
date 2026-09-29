from torch import Tensor, nn


class PointerHead(nn.Module):
    def __init__(self, d_embed: int, d_latent: int = 256):
        super().__init__()
        self.q = nn.Linear(d_embed, d_latent)
        self.k = nn.Linear(d_embed, d_latent)
        self.scale = d_latent**-0.5
        self.temperature = 1.0

    def forward(
        self, 
        questions: Tensor, # (B, K, D) 
        decide: Tensor # (B, D)
    ) -> Tensor:
        options = self.k(questions.to(self.k.weight.dtype))
        query = self.q(decide.to(self.q.weight.dtype))
        logits = (options @ query.unsqueeze(-1)).squeeze(-1) * self.scale
        return logits if self.training or self.temperature == 1.0 else logits / self.temperature
