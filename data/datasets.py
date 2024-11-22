import argparse
import os
import sys

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseDataset(torch.utils.data.Dataset):
    """
    This class is an abstract class for datasets
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the BaseDataset class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__()
        self._name = "BaseDataset"
        self._opt = opt
        self._dataset: torch.utils.data.Dataset = None
        self.dataloader: torch.utils.data.DataLoader = None
        self.mean = opt.dataset_params["mean"]
        self.std = opt.dataset_params["std"]
        self._img_type = opt.img_type
        self._create_dataset()

    @property
    def name(self) -> str:
        """
        Returns the name of the dataset
        """
        return self._name

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self._dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        """
        Returns the item at the given index
        """
        return self._dataset[index]

    def _create_dataset(self) -> None:
        """
        Creates the dataset

        Raises
        ------
        NotImplementedError
            If the dataset is not created
        """
        raise NotImplementedError

    def _print_dataset_info(self) -> None:
        """
        Prints the dataset info
        """
        print(f"{self._name} dataset has {len(self)} samples")

    def print_dataloader_info(self) -> None:
        """
        Prints the dataloader info

        Raises
        ------
        NotImplementedError
            If the dataloader is not created
        """
        if self.dataloader is None:
            raise NotImplementedError
        print(f"{self._name} dataloader has {len(self.dataloader)} batches")

    def __str__(self) -> str:
        """
        Returns the name of the dataset
        """
        return self._name
