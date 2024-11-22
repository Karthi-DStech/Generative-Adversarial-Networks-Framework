import argparse
import os
import sys
from typing import Dict, Tuple, Union

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseModel(object):
    """
    This class is an abstract class for models
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the BaseModel class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__()
        self._name = "BaseModel"
        self._opt = opt
        self._is_train = self._opt.is_train

        self._discriminator = torch.nn.Module()
        self._generator = torch.nn.Module()
        self._networks = [self._discriminator, self._generator]
        self._fake_images = torch.Tensor()
        self.performance: Dict = {}

        self._get_device()

    def _create_networks(self) -> None:
        """
        Creates the networks

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _make_loss(self) -> None:
        """
        Creates the loss functions

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _forward_discriminator(self):
        """
        Forward pass for the discriminator

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _forward_generator(self):
        """
        Forward pass for the generator

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Gets the current performance of the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the performance

        Raises
        ------
        NotImplementedError
            if the method is not implemented
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Returns the name of the model
        """
        return self._name

    def load_trained_generator(self, generator_path: str) -> None:
        """
        Loads a trained generator

        Parameters
        ----------
        generator_path: str
            The path to the generator

        Returns
        -------
        None
        """
        self._generator.load_state_dict(
            torch.load(generator_path, map_location=self._device)
        )
        print("Trained generator loaded")

    def train(self, train_generator: bool = True, do_visualization: bool = False):
        """
        Trains the model

        Parameters
        ----------
        train_generator: bool
            Whether to train the generator
        do_visualization: bool
            Whether to visualize the performance

        Returns
        -------
        None
        """
        self._discriminator.train()
        self._generator.train()

        if train_generator:
            self._generator.zero_grad()
            self._gen_loss = self._forward_generator()
            self._gen_loss.backward()
            self._generator._optimizer.step()

        self._discriminator.zero_grad()
        self._discriminator_loss = self._forward_discriminator()
        self._discriminator_loss.backward()
        self._discriminator._optimizer.step()
        if train_generator:
            self._get_current_performance(do_visualization)

    def test(self, do_visualization: bool = False):
        """
        Tests the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the performance

        Returns
        -------
        None
        """
        self._generator.eval()
        self._discriminator.eval()
        self._generator.zero_grad()
        self._discriminator.zero_grad()
        self._discriminator_loss = self._forward_discriminator()
        self._gen_loss = self._forward_generator()
        self._get_current_performance(do_visualization)

    def predict(self):
        """
        Generate new fake images
        """
        num_images = self._opt.num_images
        batch_size = self._opt.batch_size
        is_conditional = self._opt.is_conditional
        label = self._opt.label

        self._generator.eval()
        generated_images = []
        with torch.no_grad():
            for i in range(0, num_images, batch_size):
                current_batch_size = min(batch_size, num_images - i)
                self._noise = (
                    torch.randn(current_batch_size, self._opt.latent_dim)
                    .to(self._device)
                    .float()
                )
                self._gen_labels = (
                    torch.tensor([label] * current_batch_size).to(self._device).long()
                )
                if is_conditional:
                    batch_images = self._generator(self._noise, self._gen_labels)
                else:
                    batch_images = self._generator(self._noise)
                generated_images.append(batch_images)
                self._fake_images = torch.cat(generated_images, dim=0)
                self._vis_images, _ = self._get_generated_image(num_images)
                self._vis_images_names = [f"{i}" for i in range(num_images)]
                self.vis_data = (self._vis_images, self._vis_images_names)

        print(f"{num_images} images generated")

    def set_label(self, label: int) -> None:
        """
        Sets the label for the model

        Parameters
        ----------
        label: int
            The label to set

        Returns
        -------
        None
        """
        self._opt.label = label

    def __str__(self) -> str:
        """
        Returns the name of the model
        """
        return self._name

    def _print_num_params(self) -> None:
        """
        Prints the number of parameters of the model

        Raises
        ------
        ValueError
            If the networks are not created yet
        """
        if self._networks is None:
            raise ValueError("Networks are not created yet")
        else:
            for network in self._networks:
                all_params, trainable_params = network.get_num_params()
                print(
                    f"{network.name} has {all_params/1e3:.1f}K parameters ({trainable_params/1e3:.1f}K trainable)"
                )

    def _make_optimizer(self) -> None:
        """
        Creates the optimizers

        Raises
        ------
        NotImplementedError
            If the optimizer is not implemented
        """
        if self._opt.optimizer == "adam":
            self._generator._optimizer = torch.optim.Adam(  # type: ignore
                self._generator.parameters(),
                lr=self._opt.g_lr,
                betas=(self._opt.g_adam_beta1, self._opt.g_adam_beta2),
            )
            self._discriminator._optimizer = torch.optim.Adam(  # type: ignore
                self._discriminator.parameters(),
                lr=self._opt.d_lr,
                betas=(self._opt.d_adam_beta1, self._opt.d_adam_beta2),
            )
        elif self._opt.optimizer == "rmsprop":
            self._generator._optimizer = torch.optim.RMSprop(  # type: ignore
                self._generator.parameters(), lr=self._opt.g_lr
            )
            self._discriminator._optimizer = torch.optim.RMSprop(  # type: ignore
                self._discriminator.parameters(), lr=self._opt.d_lr
            )
        else:
            raise NotImplementedError(f"Invalid optimizer name: {self._opt.optimizer}")
        self._make_scheduler()

    def _make_scheduler(self) -> None:
        """
        Creates the learning rate schedulers

        Raises
        ------
        NotImplementedError
            If the scheduler is not implemented

        ValueError
            If the networks are not created yet
        """
        if self._networks is None:
            raise ValueError("Networks are not created yet")
        else:
            for network in self._networks:
                if self._opt.lr_scheduler == "none":
                    network._scheduler = None  # type: ignore
                elif self._opt.lr_scheduler == "step":
                    network._scheduler = torch.optim.lr_scheduler.StepLR(  # type: ignore
                        network._optimizer,  # type: ignore
                        step_size=self._opt.lr_decay_step,
                        gamma=self._opt.lr_decay_gamma,
                    )
                else:
                    raise NotImplementedError(
                        f"Invalid lr scheduler: {self._opt.lr_scheduler}"
                    )

    def update_learning_rate(self) -> None:
        """
        Updates the learning rate

        Raises
        ------
        ValueError
            If the networks are not created yet
        """
        if self._networks is None:
            raise ValueError("Networks are not created yet")
        else:
            for network in self._networks:
                if network._scheduler is not None:
                    network._scheduler.step()

    def set_input(self, data: torch.Tensor) -> None:
        """
        Sets the input of the model

        Parameters
        ----------
        data: torch.Tensor
            The input data

        Returns
        -------
        None
        """
        # Real Data
        self._real_images, self._real_labels = data
        self._real_images = self._real_images.to(self._device).float()
        self._real_labels = self._real_labels.to(self._device).long()

        # Adversarial ground truths
        self._valid = torch.ones(self._real_images.size(0), 1).to(self._device).float()
        self._fake = torch.zeros(self._real_images.size(0), 1).to(self._device).float()

        # Noise and labels for generator
        self._noise = (
            torch.randn(self._real_images.size(0), self._opt.latent_dim)
            .to(self._device)
            .float()
        )
        self._gen_labels = (
            torch.randint(0, self._opt.n_classes, (self._real_images.size(0),))
            .to(self._device)
            .long()
        )

    def _get_device(self) -> None:
        """
        Gets the device to train the model
        """
        self._device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        print(f"Using device: {self._device}")

    def _send_to_device(
        self, data: Union[torch.Tensor, list]
    ) -> Union[torch.Tensor, list]:
        """
        Sends the data to the device

        Parameters
        ----------
        data: torch.Tensor
            The data to send to the device

        Returns
        -------
        torch.Tensor
            The data in the device
        """
        if isinstance(data, list):
            return [x.to(self._device) for x in data]
        else:
            return data.to(self._device)

    def save_networks(self, path: str, epoch: Union[int, str]) -> None:
        """
        Saves the networks

        Parameters
        ----------
        path: str
            The path to save the networks
        epoch: Union[int, str]
            The current epoch

        Returns
        -------
        None
        """
        for network in self._networks:
            network.save(path, epoch)

    def _get_generated_image(self, n_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the generated images

        Parameters
        ----------
        n_samples: int
            The number of samples to save

        Returns
        -------
        torch.Tensor
            The generated images
        """

        self._vis_images = self._fake_images[:n_samples]
        self._vis_images = self._vis_images.view(
            self._vis_images.size(0),
            self._opt.out_channels,
            self._opt.img_size,
            self._opt.img_size,
        )
        self._vis_labels = self._gen_labels[:n_samples]

        mean = self._opt.dataset_params["mean"]
        std = self._opt.dataset_params["std"]
        self._vis_images = self._vis_images * std + mean
        self.vis_data = (self._vis_images, self._vis_labels)
        return self.vis_data
