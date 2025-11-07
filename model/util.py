from torch import nn as nn
from torchinfo import summary

from options.dataset_resolution import Resolutions, shape_by_resolution


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_encoder(encoder: nn.Module) -> nn.Module:
    encoder_children = encoder.children()

    for child in encoder_children:
        for param in child.parameters():
            param.requires_grad = False

    return encoder


def unfreeze(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    return


def show_model_summary(model: nn.Module, input_size: Resolutions) -> None:
    input_size = shape_by_resolution[input_size]
    summary(
        model=model,
        input_size=(1, 3, input_size[0], input_size[1]),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
