import torch


class FGM:
    """
    Fast Gradient Method adversarial training.

    Tạo perturbation trên embedding parameters dựa trên gradient,
    sau đó khôi phục lại parameters ban đầu sau adversarial backward.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        epsilon: float = 1.0,
        emb_name: str = "embeddings.word_embeddings",
    ):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self) -> None:
        """Thêm perturbation vào embedding parameters."""
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.emb_name in name
            ):
                self.backup[name] = param.data.clone()

                grad_norm = torch.norm(param.grad)

                if torch.isfinite(grad_norm) and grad_norm.item() > 0:
                    perturbation = (
                        self.epsilon * param.grad / (grad_norm + 1e-12)
                    )
                    param.data.add_(perturbation)

    def restore(self) -> None:
        """Khôi phục embedding parameters trước perturbation."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])

        self.backup.clear()
