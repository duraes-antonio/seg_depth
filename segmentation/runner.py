from typing import Iterable, Dict, Callable, Literal, TypedDict, Optional, Union

from lightning import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT, LRSchedulerPLType
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

ExecutionType = Literal["train", "test", "val"]


class CalculateMetricsParams(TypedDict):
    image: Tensor
    prediction: Tensor
    ground_truth: Tensor
    model: nn.Module


GetMetrics = Callable[[CalculateMetricsParams], Dict[str, float]]


class LightningSegmentationRunner(LightningModule):

    def __init__(
            self,
            model: nn.Module,
            get_metrics: GetMetrics,
            optimizer: Optimizer,
            lr_scheduler: Optional[LRScheduler] = None,
            criterion: Optional[Callable[[Tensor, Tensor], Union[Tensor, float]]] = None,
    ):
        """
        :param model: nn.Module - Modelo a ser executado
        :param criterion: loss function
        :param get_metrics: Função que calcula as métricas, recebe (input, prediction, target) e retorna um dicionário
        """
        super(LightningSegmentationRunner, self).__init__()
        self.model = model
        self.criterion = criterion
        self.get_metrics = get_metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def calculate_metrics(self, batch: Iterable[Tensor]) -> Dict[str, float]:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        prediction = self.model(images)
        metrics = self.get_metrics({
            "image": images,
            "prediction": prediction,
            "ground_truth": labels,
            "model": self.model
        })

        if self.criterion is None:
            return metrics

        loss: Tensor = self.criterion(prediction, labels)
        return {
            "loss": loss.item(),
            **metrics
        }

    def calculate_loss(self, batch: Iterable[Tensor]) -> Tensor:
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        prediction = self.model(images)
        return self.criterion(prediction, labels)

    def training_step(self, batch: Iterable[Tensor], batch_idx: int) -> Tensor:
        values = self.calculate_metrics(batch)
        values = {f'train_{k}': v for k, v in values.items()}
        self.log_dict(values, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        return self.calculate_loss(batch)

    def test_step(self, batch: Iterable[Tensor], batch_idx: int) -> STEP_OUTPUT:
        values = self.calculate_metrics(batch)
        values = {f'test_{k}': v for k, v in values.items()}
        self.log_dict(values, on_step=False, on_epoch=True, logger=True)
        return values

    def predict_step(self, batch: Iterable[Tensor], batch_idx: int) -> STEP_OUTPUT:
        images, _ = batch
        images = images.to(self.device)
        return self.model(images)

    def validation_step(self, batch: Iterable[Tensor], batch_idx: int) -> STEP_OUTPUT:
        values = self.calculate_metrics(batch)
        values = {f'val_{k}': v for k, v in values.items()}
        self.log_dict(values, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return self.calculate_loss(batch)

    def configure_optimizers(self) -> Dict[str, Union[Optimizer, LRSchedulerPLType]]:
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler
        }
