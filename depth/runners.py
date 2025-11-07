import os
from os import path
from pathlib import Path
from random import shuffle
from typing import List, Callable

import cv2
import numpy as np
import torch
import torchvision
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import Logger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, random_split

from dataset.nyu import test_transform, NyuNumpyZipDataset, read_nyu_csv, train_transform, NyuDataset
from depth.metrics import get_calc_test_metrics_fn, calculate_train_metrics
from options.dataset_resolution import Resolutions, shape_by_resolution
from segmentation.runner import LightningSegmentationRunner
from util.config import SEED


def run_inference(
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        test_data_path: Path,
        resolution: Resolutions,
        output_path: str,
):
    model.eval()
    size = shape_by_resolution[resolution]
    transform_test = test_transform(size)
    test_dataset = NyuNumpyZipDataset(
        zip_path=str(test_data_path),
        transform=transform_test,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )
    runner: LightningSegmentationRunner = LightningSegmentationRunner(
        model=model, get_metrics=get_calc_test_metrics_fn(resolution),
        optimizer=optimizer, lr_scheduler=scheduler
    )
    test_trainer = Trainer(enable_checkpointing=False)
    max_depth = 10.0
    upscale_depth = torchvision.transforms.Resize(
        shape_by_resolution[Resolutions.Full]
    )
    predictions = test_trainer.predict(model=runner, dataloaders=test_loader)

    def inverse_depth_norm(depth: torch.Tensor):
        depth = max_depth / depth
        depth = torch.clamp(depth, max_depth / 100, max_depth)
        return depth

    def unpack_and_move(data):
        device = 'cuda'
        if isinstance(data, (tuple, list)):
            image = data[0].to(device, non_blocking=True)
            gt = data[1].to(device, non_blocking=True)
            return image, gt
        if isinstance(data, dict):
            keys = data.keys()
            image = data['image'].to(device, non_blocking=True)
            gt = data['depth'].to(device, non_blocking=True)
            return image, gt
        print('Type not supported')

    def save_image_results_cv(image, gt, prediction, image_id, output_path, max_depth):
        # Transforma tensores PyTorch para NumPy
        img = image[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        gt = gt[0, 0]
        prediction = prediction[0, 0].detach().cpu().numpy()

        # Normaliza as imagens para aplicar o colormap corretamente
        vmax_error = max_depth / 10.0
        vmin_error = 0.0
        cmap = cv2.COLORMAP_PLASMA

        vmax = torch.max(gt[gt != 0.0]).item()
        vmin = torch.min(gt[gt != 0.0]).item()

        gt = gt.cpu().numpy()
        error_map = gt - prediction

        # Salvar a imagem original
        save_to_dir = os.path.join(output_path, f'image_{image_id}.png')
        cv2.imwrite(save_to_dir, img)

        # Salvar o mapa de erro
        error_map_normalized = np.clip((error_map - vmin_error) / (vmax_error - vmin_error), 0, 1) * 255
        error_map_colored = cv2.applyColorMap(error_map_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        save_to_dir = os.path.join(output_path, f'errors_{image_id}.png')
        cv2.imwrite(save_to_dir, error_map_colored)

        # Salvar o ground truth
        gt_normalized = np.clip((gt - vmin) / (vmax - vmin), 0, 1) * 255
        gt_colored = cv2.applyColorMap(gt_normalized.astype(np.uint8), cmap)
        save_to_dir = os.path.join(output_path, f'gt_{image_id}.png')
        cv2.imwrite(save_to_dir, gt_colored)

        # Salvar a predição
        prediction_normalized = np.clip((prediction - vmin) / (vmax - vmin), 0, 1) * 255
        prediction_colored = cv2.applyColorMap(prediction_normalized.astype(np.uint8), cmap)
        save_to_dir = os.path.join(output_path, f'depth_{image_id}.png')
        cv2.imwrite(save_to_dir, prediction_colored)

    for i, ((input_image, ground_truth), original_prediction) in enumerate(zip(test_loader, predictions)):
        image, gt = unpack_and_move((input_image, ground_truth))

        inv_prediction = original_prediction
        prediction = inverse_depth_norm(inv_prediction)

        predict_size = prediction.shape[-2:]
        gt_size = gt.shape[-2:]

        if predict_size != gt_size:
            prediction = upscale_depth(prediction)

        # save_image_results(image, gt, prediction, i)
        save_image_results_cv(image, gt, prediction, i, output_path=output_path, max_depth=max_depth)


def run_test(
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loggers: List[Logger],
        test_data_path: Path,
        resolution: Resolutions,
):
    model.eval()
    size = shape_by_resolution[resolution]
    transform_test = test_transform(size)
    test_dataset = NyuNumpyZipDataset(
        zip_path=str(test_data_path),
        transform=transform_test,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )
    runner: LightningSegmentationRunner = LightningSegmentationRunner(
        model=model, get_metrics=get_calc_test_metrics_fn(resolution),
        optimizer=optimizer, lr_scheduler=scheduler
    )
    test_trainer = Trainer(logger=loggers, enable_checkpointing=False)
    test_trainer.test(model=runner, dataloaders=test_loader)


def run_train(
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        loss_function: Callable,
        loggers: List[Logger],
        train_data_path: Path,
        resolution: Resolutions,
        batch_size: int,
        num_workers: int,
        max_epochs: int,
        percent_to_train: int,
        percent_dataset_used: int,
        checkpoint_dir: str,
        checkpoint_filename: str,
):
    final_size = shape_by_resolution[resolution]
    train_csv_path = str(train_data_path)
    train_paths = read_nyu_csv(train_csv_path, repository_path=train_csv_path.split('data')[0])
    shuffle(train_paths)
    train_paths = train_paths[:int(len(train_paths) * percent_dataset_used / 100)]
    transform_train = train_transform(final_size)
    train_dataset = NyuDataset(
        pairs_path=train_paths,
        transform=transform_train,
        split='train',
    )

    train_val_set_size = len(train_dataset)
    train_set_size = int(train_val_set_size * percent_to_train / 100)
    valid_set_size = train_val_set_size - train_set_size

    seed = torch.Generator().manual_seed(SEED)
    train_subset, valid_subset = random_split(
        train_dataset,
        [train_set_size, valid_set_size],
        generator=seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    lightning_train_model = LightningSegmentationRunner(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        criterion=loss_function,
        get_metrics=calculate_train_metrics,
    )
    print(f'\n--> Path to save checkpoint: {path.join(checkpoint_dir, checkpoint_filename)}\n')
    callbacks = [
        EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=True),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath=checkpoint_dir,
            filename=checkpoint_filename,
            mode='min',
        )
    ]
    train_trainer = Trainer(
        callbacks=callbacks,
        max_epochs=max_epochs,
        log_every_n_steps=5,
        logger=loggers,
    )
    train_trainer.fit(lightning_train_model, train_loader, valid_loader)
