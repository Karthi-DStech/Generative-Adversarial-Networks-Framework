import argparse
import os
import sys

from torchvision import transforms
from torchvision.datasets import MNIST
from data.datasets import BaseDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MNISTDataset(BaseDataset):
    """
    MNIST dataset class
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the MNISTDataset class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "MNIST"
        self._print_dataset_info()

    def _create_dataset(self) -> None:
        """
        Creates the dataset
        """
        self._dataset = MNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(self._opt.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((self.mean,), (self.std,)),
                ]
            ),
        )


class MNISTTest(MNISTDataset):
    """
    MNIST test dataset class
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the MNISTTest class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "MNISTTest"

    def _create_dataset(self) -> None:
        """
        Creates the dataset
        """
        self._dataset = MNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(self._opt.img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((self.mean,), (self.std,)),
                ]
            ),
        )
