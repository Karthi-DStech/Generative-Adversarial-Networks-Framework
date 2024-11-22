import argparse
import os
import sys

import torch
import torch.nn as nn
from torch._tensor import Tensor

from call_methods import make_network
from model.models import BaseModel
from utils.losses import wasserstein_loss

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WGAN_GP(BaseModel):
    """
    This class implements the WGAN model with gradient penalty
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the WGANGP class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "WGANGP"
        self._networks = None
        self._create_networks()
        self._print_num_params()

        if self._is_train:
            self._make_loss()
            self._make_optimizer()

    def _create_networks(self) -> None:
        """
        Creates the networks
        """
        self._generator = make_network(
            network_name="vanillagenerator",
            opt=self._opt,
            g_neurons=self._opt.vanilla_g_neurons,
            latent_dim=self._opt.latent_dim,
            out_channels=self._opt.out_channels,
        )
        self._generator.init_weights(self._opt.init_type)

        self._discriminator = make_network(
            network_name="wgancritic",
            opt=self._opt,
            d_neurons=self._opt.vanilla_d_neurons,
            out_channels=self._opt.out_channels,
        )
        self._discriminator.init_weights(self._opt.init_type)
        self._networks = [self._generator, self._discriminator]
        self._send_to_device(self._networks)

    def _make_loss(self) -> None:
        """
        Creates the loss functions
        """
        self._w_loss = wasserstein_loss

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

        self._gp = self._gradient_penalty()
        self._discriminator.zero_grad()
        self._gp.backward()
        self._discriminator._optimizer.step()

        if train_generator:
            self._get_current_performance(do_visualization)

    def _forward_discriminator(self) -> torch.Tensor:
        """
        Forward pass for the discriminator

        Returns
        -------
        torch.Tensor
            The discriminator loss
        """
        self._real_validity = self._discriminator(self._real_images)
        self._real_loss_w = (
            self._w_loss(self._real_validity, True) * self._opt.d_lambda_w
        )

        self._fake_images = self._generator(self._noise)
        self._fake_validity = self._discriminator(self._fake_images.detach())
        self._fake_loss_w = (
            self._w_loss(self._fake_validity, False) * self._opt.d_lambda_w
        )

        discriminator_loss = self._real_loss_w + self._fake_loss_w

        return discriminator_loss

    def _forward_generator(self) -> torch.Tensor:
        """
        Forward pass for the generator

        Returns
        -------
        torch.Tensor
            The generator loss
        """
        self._fake_images = self._generator(self._noise)
        self._gen_validity = self._discriminator(self._fake_images)
        generator_loss = self._w_loss(self._gen_validity, True) * self._opt.g_lambda_w

        return generator_loss

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Get the current performance of the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the images

        Returns
        -------
        None
        """
        self._current_disc_performance = {
            "D_real_loss_w": self._real_loss_w.item(),
            "D_fake_loss_w": self._fake_loss_w.item(),
            "D_gp": self._gp.item(),
            "D_loss": self._discriminator_loss.item(),
        }

        self._current_gen_performance = {
            "G_loss": self._gen_loss.item(),
        }
        self._current_performance = {
            **self._current_disc_performance,
            **self._current_gen_performance,
        }

        if do_visualization:
            self.vis_data = self._get_generated_image(n_samples=self._opt.n_vis_samples)

        gen_lr = self._generator._optimizer.param_groups[0]["lr"]
        disc_lr = self._discriminator._optimizer.param_groups[0]["lr"]

        self.performance = {
            "losses": self._current_performance,
            "lr": {"G_lr": gen_lr, "D_lr": disc_lr},
        }

    def _gradient_penalty(self) -> torch.Tensor:
        """
        Calculates the gradient penalty loss

        Returns
        -------
        torch.Tensor
            The calculated gradient penalty
        """
        batch_size = self._real_images.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1, device=self._device).expand_as(
            self._real_images
        )
        interpolates = (
            alpha * self._real_images.data
            + ((1 - alpha) * self._fake_images.view_as(self._real_images).data)
        ).requires_grad_(True)
        disc_interpolates = self._discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates, device=self._device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = (
            torch.mean((gradients.norm(2, dim=1) - 1) ** 2) * self._opt.d_lambda_gp
        )
        return gradient_penalty


class WGAN_WC(WGAN_GP):
    """
    This class implements the WGAN model with weight clipping
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the WGANWC class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "WGANGP"

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

    def _forward_discriminator(self) -> Tensor:
        """
        Forward pass for the discriminator with weight clipping

        Returns
        -------
        torch.Tensor
            The discriminator loss
        """
        for p in self._discriminator.parameters():
            p.data.clamp_(-self._opt.clip_value, self._opt.clip_value)

        self._real_validity = self._discriminator(self._real_images)
        self._real_loss_w = (
            self._w_loss(self._real_validity, True) * self._opt.d_lambda_w
        )

        self._fake_images = self._generator(self._noise)
        self._fake_validity = self._discriminator(self._fake_images.detach())
        self._fake_loss_w = (
            self._w_loss(self._fake_validity, False) * self._opt.d_lambda_w
        )

        discriminator_loss = self._real_loss_w + self._fake_loss_w

        return discriminator_loss

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Get the current performance of the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the images

        Returns
        -------
        None
        """
        self._current_disc_performance = {
            "D_real_loss_w": self._real_loss_w.item(),
            "D_fake_loss_w": self._fake_loss_w.item(),
            "D_loss": self._discriminator_loss.item(),
        }

        self._current_gen_performance = {
            "G_loss": self._gen_loss.item(),
        }
        self._current_performance = {
            **self._current_disc_performance,
            **self._current_gen_performance,
        }

        if do_visualization:
            self.vis_data = self._get_generated_image(n_samples=self._opt.n_vis_samples)

        gen_lr = self._generator._optimizer.param_groups[0]["lr"]
        disc_lr = self._discriminator._optimizer.param_groups[0]["lr"]

        self.performance = {
            "losses": self._current_performance,
            "lr": {"G_lr": gen_lr, "D_lr": disc_lr},
        }


class WCGAN_GP(WGAN_GP):
    """
    This class implements the WCGAN model with gradient penalty. It is based on the
    WGAN model with gradient penalty but uses the Convolution Layers instead of the
    Fully Connected Layers.
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the WCGANGP class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "WCGANGP"

    def _create_networks(self) -> None:
        """
        Creates the networks
        """
        self._generator = make_network(
            network_name="convgangenerator",
            opt=self._opt,
            latent_dim=self._opt.latent_dim,
            out_channels=self._opt.out_channels,
        )
        self._generator.init_weights(self._opt.init_type)

        self._discriminator = make_network(
            network_name="convgancritic",
            opt=self._opt,
            in_channels=self._opt.in_channels,
        )
        self._discriminator.init_weights(self._opt.init_type)
        self._networks = [self._generator, self._discriminator]
        self._send_to_device(self._networks)


class ACWGAN_GP(WGAN_GP):
    """
    This class implements the ACWGAN model with gradient penalty. It is based on the
    WGAN model with gradient penalty but uses the Auxiliary Classifier in the discriminator.
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the ACWGANGP class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "ACWGANGP"

    def _create_networks(self) -> None:
        """
        Creates the networks
        """
        self._generator = make_network(
            network_name="acvanillagenerator",
            opt=self._opt,
            n_classes=self._opt.n_classes,
            embedding_dim=self._opt.embedding_dim,
            g_neurons=self._opt.vanilla_g_neurons,
            latent_dim=self._opt.latent_dim,
            out_channels=self._opt.out_channels,
        )
        self._generator.init_weights(self._opt.init_type)

        self._discriminator = make_network(
            network_name="acwgancritic",
            opt=self._opt,
            n_classes=self._opt.n_classes,
            d_neurons=self._opt.vanilla_d_neurons,
            out_channels=self._opt.out_channels,
        )
        self._discriminator.init_weights(self._opt.init_type)
        self._networks = [self._generator, self._discriminator]
        self._send_to_device(self._networks)

    def _make_loss(self) -> None:
        """
        Creates the loss functions
        """
        self._w_loss = wasserstein_loss
        self._criterion_cls = nn.CrossEntropyLoss()

    def _forward_discriminator(self) -> torch.Tensor:
        """
        Forward pass for the discriminator

        Returns
        -------
        torch.Tensor
            The discriminator loss
        """
        self._real_validity, self._real_labels_pred = self._discriminator(
            self._real_images
        )
        self._real_loss_w = (
            self._w_loss(self._real_validity, True) * self._opt.d_lambda_w
        )
        self._real_loss_cls = (
            self._criterion_cls(self._real_labels_pred, self._real_labels)
            * self._opt.d_lambda_cls
        )

        self._fake_images = self._generator(self._noise, self._gen_labels)
        self._fake_validity, self._fake_labels_pred = self._discriminator(
            self._fake_images.detach()
        )
        self._fake_loss_w = (
            self._w_loss(self._fake_validity, False) * self._opt.d_lambda_w
        )
        self._fake_loss_cls = (
            self._criterion_cls(self._fake_labels_pred, self._gen_labels)
            * self._opt.d_lambda_cls
        )

        discriminator_loss = (
            self._real_loss_w
            + self._real_loss_cls
            + self._fake_loss_w
            + self._fake_loss_cls
        ) / 2

        # Calculate discriminator label accuracy, note that predicted labels are logits
        self._real_label_acc = torch.mean(
            (
                torch.softmax(self._real_labels_pred, dim=1).argmax(1)
                == self._real_labels
            ).float()
        )
        self._fake_label_acc = torch.mean(
            (
                torch.softmax(self._fake_labels_pred, dim=1).argmax(1)
                == self._gen_labels
            ).float()
        )
        self._discriminator_label_acc = (
            self._real_label_acc + self._fake_label_acc
        ) / 2

        return discriminator_loss

    def _forward_generator(self) -> torch.Tensor:
        """
        Forward pass for the generator

        Returns
        -------
        torch.Tensor
            The generator loss
        """
        self._fake_images = self._generator(self._noise, self._gen_labels)
        self._gen_validity, self._gen_labels_pred = self._discriminator(
            self._fake_images
        )
        self._gen_loss_w = self._w_loss(self._gen_validity, True) * self._opt.g_lambda_w
        self._gen_loss_cls = (
            self._criterion_cls(self._gen_labels_pred, self._gen_labels)
            * self._opt.g_lambda_cls
        )
        generator_loss = (self._gen_loss_w + self._gen_loss_cls) / 2

        return generator_loss

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Get the current performance of the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the images

        Returns
        -------
        None
        """
        self._current_disc_performance = {
            "D_real_loss_w": self._real_loss_w.item(),
            "D_real_loss_cls": self._real_loss_cls.item(),
            "D_fake_loss_w": self._fake_loss_w.item(),
            "D_fake_loss_cls": self._fake_loss_cls.item(),
            "D_gp": self._gp.item(),
            "D_loss": self._discriminator_loss.item(),
            "D_real_label_acc": self._real_label_acc.item(),
            "D_fake_label_acc": self._fake_label_acc.item(),
            "D_label_acc": self._discriminator_label_acc.item(),
        }

        self._current_gen_performance = {
            "G_loss_w": self._gen_loss_w.item(),
            "G_loss_cls": self._gen_loss_cls.item(),
            "G_loss": self._gen_loss.item(),
        }
        self._current_performance = {
            **self._current_disc_performance,
            **self._current_gen_performance,
        }

        if do_visualization:
            self.vis_data = self._get_generated_image(n_samples=self._opt.n_vis_samples)

        gen_lr = self._generator._optimizer.param_groups[0]["lr"]
        disc_lr = self._discriminator._optimizer.param_groups[0]["lr"]

        self.performance = {
            "losses": self._current_performance,
            "lr": {"G_lr": gen_lr, "D_lr": disc_lr},
        }

    def _gradient_penalty(self) -> torch.Tensor:
        """
        Calculates the gradient penalty loss

        Returns
        -------
        torch.Tensor
            The calculated gradient penalty
        """
        batch_size = self._real_images.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1, device=self._device).expand_as(
            self._real_images
        )
        interpolates = (
            alpha * self._real_images.data
            + ((1 - alpha) * self._fake_images.view_as(self._real_images).data)
        ).requires_grad_(True)
        disc_interpolates = self._discriminator(interpolates)[0]
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates, device=self._device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = (
            torch.mean((gradients.norm(2, dim=1) - 1) ** 2) * self._opt.d_lambda_gp
        )
        return gradient_penalty
