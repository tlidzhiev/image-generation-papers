from pathlib import Path
from typing import Any, Literal

import numpy as np
import wandb


class WandbWriter:
    def __init__(
        self,
        project_config: dict[str, Any],
        project_name: str,
        entity: str | None = None,
        run_id: str | None = None,
        run_name: str | None = None,
        mode: str = 'online',
        save_code: bool = False,
        **kwargs,
    ):
        wandb.login()

        wandb.init(
            project=project_name,
            entity=entity,
            config=project_config,
            name=run_name,
            resume='allow',
            id=run_id,
            mode=mode,
            save_code=save_code,
        )

        self.wandb = wandb
        self.run_id = self.wandb.run.id
        self.run_name = self.wandb.run.name
        self.mode = ''
        self.step = 0

    def set_step(self, step: int, mode: Literal['train', 'val', 'test'] = 'train'):
        self.step = step
        self.mode = mode

    def add_checkpoint(self, checkpoint_path: str, save_dir: str):
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, name: str, value: float):
        self.wandb.log(
            {self._object_name(name): value},
            step=self.step,
        )

    def add_scalars(self, values: dict[str, float]):
        self.wandb.log(
            {self._object_name(k): v for k, v in values.items()},
            step=self.step,
        )

    def add_image(self, name: str, image: np.ndarray | Path | str):
        self.wandb.log(
            {self._object_name(name): self.wandb.Image(image)},
            step=self.step,
        )

    def finish(self):
        self.wandb.finish()

    def _object_name(self, name: str) -> str:
        return f'{self.mode}_{name}'
