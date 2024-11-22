import os
import sys
import torch

import torch.nn as nn
from utils.weight_init import normal_init, xavier_init, kaiming_init
from typing import Tuple, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseNetwork(nn.Module):
    """
    This class is an abstract class for networks
    """

    def __init__(self) -> None:
        """
        Initializes the BaseNetwork class
        """
        super().__init__()
        self._name = "BaseNetwork"

    @property
    def name(self) -> str:
        """
        Returns the name of the network
        """
        return self._name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the network

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """
        Returns the string representation of the network
        """
        return self._name

    def init_weights(self, init_type: str = "normal") -> None:
        """
        Initializes the weights of the network

        Parameters
        ----------
        init_type: str
            The type of initialization

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        if init_type == "normal":
            self.apply(normal_init)
        elif init_type == "xavier_normal":
            self.apply(xavier_init)
        elif init_type == "kaiming_normal":
            self.apply(kaiming_init)
        else:
            raise NotImplementedError(f"Invalid init type: {init_type}")

    def get_num_params(self) -> Tuple[int, int]:
        """
        Returns the number of parameters in the network

        Returns
        -------
        all_params: int
            The total number of parameters in the network
        trainable_params: int
            The total number of trainable parameters in the network
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return all_params, trainable_params

    def save(self, path: str, epoch: Union[int, str]) -> None:
        """
        Saves the network to the specified path

        Parameters
        ----------
        path: str
            The path to save the network
        epoch: Union[int, str]
            The epoch number

        Returns
        -------
        None
        """
        network_file_name = f"net_{self.name}_{epoch}.pth"
        netwrok_file_path = os.path.join(path, network_file_name)
        torch.save(self.state_dict(), netwrok_file_path)
        print(f"Saved {network_file_name} to {path}")

        optimizer_file_name = f"optimizer_{self.name}_{epoch}.pth"
        optimizer_file_path = os.path.join(path, optimizer_file_name)
        torch.save(self._optimizer.state_dict(), optimizer_file_path)
        print(f"Saved {optimizer_file_name} to {path}")
