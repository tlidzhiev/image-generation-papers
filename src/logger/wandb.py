from pathlib import Path
from typing import Any, Literal

import numpy as np
import wandb


class WandbWriter:
    """
    Wrapper for Weights & Biases logging functionality.

    Parameters
    ----------
    project_config : dict[str, Any]
        Configuration dictionary for the project.
    project_name : str
        Name of the W&B project.
    entity : str or None, optional
        W&B entity (username or team name).
    run_id : str or None, optional
        Unique ID for resuming a run.
    run_name : str or None, optional
        Human-readable name for the run.
    mode : str, optional
        W&B mode ('online', 'offline', or 'disabled'), default 'online'.
    save_code : bool, optional
        Whether to save code to W&B, default False.
    **kwargs : dict
        Additional keyword arguments.
    """

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
        """
        Set current step and mode for logging.

        Parameters
        ----------
        step : int
            Current training/evaluation step.
        mode : {'train', 'val', 'test'}, optional
            Current mode, default 'train'.
        """
        self.step = step
        self.mode = mode

    def add_checkpoint(self, checkpoint_path: str, save_dir: str):
        """
        Save a checkpoint file to W&B.

        Parameters
        ----------
        checkpoint_path : str
            Path to the checkpoint file.
        save_dir : str
            Base directory for saving.
        """
        self.wandb.save(checkpoint_path, base_path=save_dir)

    def add_scalar(self, name: str, value: float):
        """
        Log a scalar value to W&B.

        Parameters
        ----------
        name : str
            Name of the scalar metric.
        value : float
            Value to log.
        """
        self.wandb.log(
            {self._object_name(name): value},
            step=self.step,
        )

    def add_scalars(self, values: dict[str, float]):
        """
        Log multiple scalar values to W&B.

        Parameters
        ----------
        values : dict[str, float]
            Dictionary of metric names to values.
        """
        self.wandb.log(
            {self._object_name(k): v for k, v in values.items()},
            step=self.step,
        )

    def add_image(self, name: str, image: np.ndarray | Path | str):
        """
        Log an image to W&B.

        Parameters
        ----------
        name : str
            Name for the image.
        image : np.ndarray or Path or str
            Image as array or path to image file.
        """
        self.wandb.log(
            {self._object_name(name): self.wandb.Image(image)},
            step=self.step,
        )

    def finish(self):
        """Finish the W&B run and upload pending data."""
        self.wandb.finish()

    def _object_name(self, name: str) -> str:
        """
        Create prefixed object name with mode.

        Parameters
        ----------
        name : str
            Base name of the object.

        Returns
        -------
        str
            Prefixed name with mode.
        """
        return f'{self.mode}_{name}'
