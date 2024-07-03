import torch
from torch import nn

from utils.errors.exceptions import WrongTensorShapeException


class IoU(nn.Module):
    """ Intersection over union evaluation metric.
    """
    def __init__(self, *args, **kwargs) -> None:
        """ Intersection over union evaluation metric.
        """
        super().__init__(*args, **kwargs)
        self.intersec = None
        self.union = None

    def forward(self, x: torch.TensorType, y: torch.TensorType) -> torch.Tensor:
        """ Calculates IoU of two masks of one channel.

        args:
            :param x: Model output
            :type x: torch.Tensor
            :param y: Target
            :type y: torch.Tensor
        Returns:
            :return iou: Intersection over union
            :type iou: torch.Tensor
        """
        if x.shape != y.shape:
            raise WrongTensorShapeException("Both tensors must have same shape")
        
        union = torch.sum(torch.logical_or(x, y))
        self.union = union

        # For ZeroDivisionError avoidance
        if union == 0:
            y = torch.logical_not(y)
            x = torch.logical_not(x)
            union = torch.sum(torch.logical_or(x, y))

        intersection = torch.sum(torch.logical_and(x, y))
        self.intersec = 0 if self.union == 0 else intersection

        iou = intersection / union

        return iou

if __name__ == "__main__":
    x = torch.ones((1, 128, 128, 128))
    y = torch.ones((1, 128, 128, 128))
    iou = IoU()
    print(iou(x, y))
