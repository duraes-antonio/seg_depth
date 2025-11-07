import os
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from lightning import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from depth.losses import DepthLoss
from depth.model_loader import ModelLoader
from depth.runners import run_test, run_train, run_inference
from model import loader
from model.coatnet.coatnet import register_smp_custom_encoders
from model.util import show_model_summary
from options.dataset_resolution import Resolutions
from options.model import Models
from options.task import Task
from util.config import SEED, DEVICE

torch.set_float32_matmul_precision('medium')
register_smp_custom_encoders()


def save_checkpoint(model, optimizer, lr_scheduler, checkpoint_pth, filename: str):
    if not os.path.isdir(checkpoint_pth):
        os.mkdir(checkpoint_pth)

    # Save checkpoint for training
    checkpoint_dir = os.path.join(
        checkpoint_pth, filename + '.pth'
    )
    to_save = {
        "model": model.cuda().state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
    }
    torch.save(to_save, checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir)
    torch.save(checkpoint["model"], checkpoint_dir.replace('.pth', '_model.pth'))


RunModes = ['train', 'test', 'inference']
RunModesLiteral = Literal[tuple(RunModes)]


@click.command()
@click.option("-e", "--max_epochs", type=click.INT, default=20)
@click.option("-b", "--batch_size", type=click.INT, default=16)
@click.option("-w", "--num_workers", type=click.INT, default=4)
@click.option("-mr", "--mode_run", type=click.Choice(RunModes), default='train')
@click.option(
    "-m",
    "--model",
    type=click.Choice(list(map(str, Models))),
    default=Models.UNetMixedTransformerB2.value
)
@click.option(
    "-s",
    "--size",
    type=click.Choice(list(map(str, Resolutions))),
    default=Resolutions.Mini.value
)
@click.option(
    "-clp",
    "--checkpoint_load_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    default=None
)
@click.option(
    "-csp",
    "--checkpoint_save_path",
    type=click.Path(
        exists=False,
        file_okay=False,
        readable=False,
        path_type=Path,
    ),
    default="checkpoints/checkpoints_depth/",
)
@click.option(
    "-tdp",
    "--train_data_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    default="data/nyu2_train.csv",
)
@click.option(
    "-tedp",
    "--test_data_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    default="NYU_Testset.zip",
)
@click.option(
    "-rd",
    "--run_directory",
    type=click.Path(
        exists=False,
        file_okay=False,
        readable=False,
        path_type=Path,
    ),
)
@click.option(
    "-mf",
    "--model_filename",
    type=click.STRING,
)
@click.option(
    "-du",
    "--dataset_usage",
    type=click.IntRange(1, 100),
    default=100
)
@click.option(
    "-ss",
    "--show_summary",
    is_flag=True,
    show_default=True,
    default=False,
    help="Show model summary"
)
def main(
        max_epochs: int,
        batch_size: int,
        num_workers: int,
        size: Resolutions,
        checkpoint_load_path: Optional[Path],
        checkpoint_save_path: Path,
        train_data_path: Path,
        test_data_path: Path,
        run_directory: Path,
        model_filename: Optional[str],
        mode_run: RunModesLiteral,
        model: str,
        dataset_usage: int = 100,
        show_summary: bool = False,
):
    seed_everything(seed=SEED, workers=True)
    percent_to_train = 90
    max_depth = 10.0

    if run_directory is None:
        raise ValueError("run_directory must be provided")

    run_directory = str(run_directory)
    checkpoint_load_path = str(checkpoint_load_path) if checkpoint_load_path else ''
    from_task = Task.SEGMENTATION if 'segmentation' in checkpoint_load_path else Task.DEPTH
    size = Resolutions(size)
    loss_function = DepthLoss(0.1, 1, 1, max_depth=max_depth)
    model_type = Models(model)

    load_model_hardcoded = False

    if load_model_hardcoded:
        print('Loading HARDCODED model')
        scheduler_step_size = 15
        learning_rate = 1e-4
        model = loader.load_model(model_type, size, num_classes=19)
        model.load_state_dict(torch.load(checkpoint_load_path))

        new_last_layer = nn.Sequential(
            nn.Conv2d(16, 1, (3, 3), (1, 1), (1, 1)),
        ).to(DEVICE)
        model.segmentation_head[0] = new_last_layer

        optimizer = Adam(model.parameters(), learning_rate)
        scheduler = StepLR(optimizer, scheduler_step_size, 0.1)

    else:
        print('Loading model using LOADER')
        model_loader = ModelLoader(
            target_model=model_type,
            resolution=size,
            checkpoint_load_path=checkpoint_load_path,
            from_task=from_task,
            to_task=Task.DEPTH
        )
        model = model_loader.model
        optimizer = model_loader.optimizer
        scheduler = model_loader.scheduler

    if show_summary:
        show_model_summary(model, size)

    optimizer_used = len(optimizer.state) > 0
    scheduler_used = scheduler.last_epoch > 0

    model_filename_params = [
        f'model-{model_type.value}',
        f'size-{size.value}',
        f'ds-usage-{dataset_usage}',
        f'batch-{batch_size}',
        f'opt-{int(optimizer_used)}',
        f'scheduler-{int(scheduler_used)}',
    ]

    if not checkpoint_load_path:
        model_filename_params = ['base'] + model_filename_params

    log_root_path = os.path.join('depth', 'logs', run_directory)
    log_last_dir = '_'.join(model_filename_params)

    auto_model_filename = '_'.join(model_filename_params + ['{epoch}'])
    model_filename = model_filename or auto_model_filename
    loggers = [TensorBoardLogger(log_root_path, name=log_last_dir)]
    original_run_directory = run_directory
    run_directory = os.path.join('depth', run_directory)

    if mode_run == 'train':
        run_train(
            model, optimizer, scheduler, loss_function, loggers,
            train_data_path, size, batch_size, num_workers, max_epochs, percent_to_train,
            dataset_usage, run_directory, model_filename
        )
        run_test(model, optimizer, scheduler, loggers, test_data_path, size)
        model_filename = model_filename.replace('{epoch}', str(max_epochs))
        print(f'\n--> Path to save MODEL: {os.path.join(run_directory, model_filename)}\n')
        save_checkpoint(model, optimizer, scheduler, run_directory, model_filename)

    if mode_run == 'test':
        run_test(model, optimizer, scheduler, loggers, test_data_path, size)

    if mode_run == 'inference':
        if not os.path.exists(original_run_directory):
            os.makedirs(original_run_directory)
        run_inference(model, optimizer, scheduler, test_data_path, size, original_run_directory)

    return None


if __name__ == "__main__":
    main()
