import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


def _parse_act(activation: str) -> tuple[str, float | None]:
    """
    Parse activation string into name and optional parameter.

    Parameters
    ----------
    activation : str
        Activation string, optionally with parameter (e.g., 'leaky_relu:0.2').

    Returns
    -------
    tuple[str, float or None]
        Activation name and parameter (None if not specified).

    Raises
    ------
    ValueError
        If parameter is invalid.
    """
    if ':' in activation:
        act_name, param_str = activation.split(':', 1)
        try:
            param = float(param_str)
        except ValueError:
            raise ValueError(
                f'Invalid activation parameter "{param_str}" in "{activation}". '
                f'Parameter must be a number.'
            )
        return act_name.lower(), param
    return activation.lower(), None


def get_activation(activation: str) -> nn.Module:
    """
    Get activation module from string descriptor.

    Parameters
    ----------
    activation : str
        Activation type ('relu', 'leaky_relu', 'gelu', 'silu').

    Returns
    -------
    nn.Module
        PyTorch activation module.

    Raises
    ------
    ValueError
        If activation type is unsupported.
    """
    act_name, param = _parse_act(activation)
    match act_name:
        case 'relu':
            return nn.ReLU()
        case 'leaky_relu':
            slope = param if param is not None else 0.01
            return nn.LeakyReLU(slope)
        case 'gelu':
            return nn.GELU()
        case 'silu':
            return nn.SiLU()
        case _:
            raise ValueError(
                f'Unknown activation type: "{act_name}". '
                f'Supported types: "relu", "leaky_relu", "gelu", "silu"'
            )


def get_norm_layer(norm_type: str, channels: int, num_groups: int = 32) -> nn.Module:
    """
    Get normalization layer from string descriptor.

    Parameters
    ----------
    norm_type : str
        Normalization type ('batchnorm' or 'groupnorm').
    channels : int
        Number of channels.
    num_groups : int, optional
        Number of groups for GroupNorm (default: 32).

    Returns
    -------
    nn.Module
        PyTorch normalization module.

    Raises
    ------
    ValueError
        If normalization type is unsupported.
    """
    match norm_type:
        case 'batchnorm':
            return nn.BatchNorm2d(channels)
        case 'groupnorm':
            num_groups = min(num_groups, channels)
            return nn.GroupNorm(num_groups, channels)
        case _:
            raise ValueError(
                f'Unknown normalization type: "{norm_type}". '
                'Supported types: "batchnorm", "groupnorm"'
            )


def initialize_weights(module: nn.Module, activation: str = 'relu', mode: str = 'normal'):
    """
    Initialize module weights using Kaiming initialization.

    Parameters
    ----------
    module : nn.Module
        PyTorch module to initialize.
    activation : str, optional
        Activation function type (default: 'relu').
    mode : str, optional
        Initialization mode ('normal' or 'uniform'), default 'normal'.

    Raises
    ------
    ValueError
        If mode or activation type is unsupported.
    """
    if mode not in ['normal', 'uniform']:
        raise ValueError(
            f'Unknown initialization mode: "{mode}". Supported modes: "normal", "uniform".'
        )

    act_name, param = _parse_act(activation)
    param = param if param is not None else 0.0
    activation_map = {
        'relu': 'relu',
        'leaky_relu': 'leaky_relu',
        'silu': 'relu',
        'gelu': 'relu',
    }

    if act_name not in activation_map:
        raise ValueError(
            f"Unknown activation type for initialization: '{act_name}'. "
            f'Supported types: {", ".join(activation_map.keys())}'
        )
    nonlinearity = activation_map[act_name]
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            init_fn = nn.init.kaiming_normal_ if mode == 'normal' else nn.init.kaiming_uniform_
            init_fn(m.weight, a=param, mode='fan_out', nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            init_fn = nn.init.kaiming_normal_ if mode == 'normal' else nn.init.kaiming_uniform_
            init_fn(m.weight, a=param, mode='fan_in', nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)


def get_num_params(model: nn.Module, trainable_only: bool = True) -> str:
    """
    Get formatted count of model parameters.

    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    trainable_only : bool, optional
        Count only trainable parameters (default: True).

    Returns
    -------
    str
        Formatted parameter count (e.g., '1.23M', '456K').
    """

    def format_number(num: int) -> str:
        if abs(num) >= 1_000_000:
            return f'{num / 1_000_000:.2f}M'
        elif abs(num) >= 1_000:
            return f'{num / 1_000:.2f}K'
        else:
            return str(num)

    if trainable_only:
        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    else:
        num_params = sum([p.numel() for p in model.parameters()])
    return format_number(num_params)


def get_lr_scheduler(
    cfg: DictConfig,
    optimizer: Optimizer,
    train_loader: DataLoader,
) -> LRScheduler:
    """
    Create learning rate scheduler from config.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    optimizer : Optimizer
        PyTorch optimizer.
    train_loader : DataLoader
        Training dataloader for computing total steps.

    Returns
    -------
    LRScheduler
        Configured learning rate scheduler.
    """
    if cfg.lr_scheduler.scheduler.name == 'constant':
        num_training_steps, num_warmup_steps = None, None
    else:
        num_training_steps = cfg.trainer.num_epochs * len(train_loader)
        num_warmup_steps = int(
            round(num_training_steps * cfg.lr_scheduler.get('warmup_ratio', 0.03))
        )
    return instantiate(
        cfg.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )


def get_dtype(dtype: str = 'float32') -> torch.dtype:
    """
    Convert string to PyTorch dtype.

    Used for Hydra configuration compatibility, as Hydra cannot directly
    instantiate torch.dtype objects.

    Parameters
    ----------
    dtype : str, optional
        Data type string ('float32' or 'float64'), default 'float32'.

    Returns
    -------
    torch.dtype
        PyTorch data type.

    Raises
    ------
    ValueError
        If dtype string is unsupported.
    """
    match dtype:
        case 'float32':
            return torch.float32
        case 'float64':
            return torch.float64
        case _:
            raise ValueError(f'Unknown dtype: {dtype}. Supported dtypes: "float32" or "float64"')
