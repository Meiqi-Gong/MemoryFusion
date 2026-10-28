"""Strict checkpoint loading shared by training and evaluation."""
from collections.abc import Mapping
from pathlib import Path
import torch


def load_state_dict_strict(model, checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f'Checkpoint must be a mapping: {path}')
    state_dict = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError(f'Checkpoint state_dict must be a mapping: {path}')
    model.load_state_dict(state_dict, strict=True)
    print(f'Loaded all model parameters from {path}')


def load_network(model, checkpoint_path, gpu):
    load_state_dict_strict(model, checkpoint_path)
    return model.cuda(gpu), []
