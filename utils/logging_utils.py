"""
utils/logging_utils.py
Experiment logging with WandB and TensorBoard support.
"""

import os
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Unified logger supporting both WandB and TensorBoard.
    Gracefully falls back to console-only if libraries not available.
    """

    def __init__(
        self,
        project_name: str = "TinyPhoBERT",
        run_name: str = "experiment",
        log_dir: str = "logs",
        use_wandb: bool = False,
        use_tensorboard: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        self.run_name = run_name
        self.log_dir = os.path.join(log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # WandB setup
        self._wandb = None
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(project=project_name, name=run_name, config=config or {})
                print(f"[Logger] WandB initialized: {project_name}/{run_name}")
            except ImportError:
                print("[Logger] WandB not installed — skipping.")
                self.use_wandb = False

        # TensorBoard setup
        self._tb_writer = None
        if use_tensorboard:
            try:
                self._tb_writer = SummaryWriter(log_dir=self.log_dir)
                print(f"[Logger] TensorBoard writer at: {self.log_dir}")
            except Exception as e:
                print(f"[Logger] TensorBoard failed: {e}")
                self.use_tensorboard = False

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log metrics at a given step."""
        # Console
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  Step {step:5d} | {metric_str}")

        # WandB
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)

        # TensorBoard
        if self.use_tensorboard and self._tb_writer is not None:
            for key, val in metrics.items():
                self._tb_writer.add_scalar(key, val, global_step=step)

    def log_hyperparams(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.config.update(hparams)

    def finish(self) -> None:
        """Clean up loggers."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()
        if self._tb_writer is not None:
            self._tb_writer.close()
