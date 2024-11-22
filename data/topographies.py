import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image

from data.datasets import BaseDataset
from utils import images_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BiologicalObservation(BaseDataset):
    """
    Biological Observation dataset class
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the BiologicalObservation class
        """
        super().__init__(opt)
        self._name = "BiologicalObservation"
        self._print_dataset_info()

    def _create_dataset(self) -> None:
        """
        Creates the dataset
        """
        self.images_path = self._opt.images_folder
        label_path = self._opt.label_path
        self._transform = images_utils.get_transform(
            self._opt.img_size, self.mean, self.std
        )

        self._annotations = pd.read_csv(label_path)
        self._images_names = [
            Path(image).stem
            for image in glob.glob(
                os.path.join(self.images_path, f"*.{self._img_type}")
            )
        ]
        self._featids = [image.split("_")[2] for image in self._images_names]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the data at the given index

        Parameters
        ----------
        index: int
            The index of the data to return

        Returns
        -------
        image: torch.Tensor
            The image at the given index
        label: torch.Tensor
            The label at the given index
        """
        featid, label = self._annotations.iloc[index]
        image_name = self._images_names[self._featids.index(str(featid))]
        img_path = os.path.join(self.images_path, f"{image_name}.{self._img_type}")

        image = Image.open(img_path)
        image = self._transform(image)
        label = torch.tensor(label)
        return image, label  # type: ignore

    def __len__(self) -> int:
        """
        Returns the length of the dataset
        """
        return len(self._annotations)
