import argparse
import os
import sys

import torch
import torch.nn as nn
from model.models import BaseModel
from call_methods import make_network

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ACGAN(BaseModel):
    """
    This class implements the ACGAN model
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the ACGAN class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "ACGAN"
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
            network_name="acgangenerator",
            opt=self._opt,
            n_classes=self._opt.n_classes,
            embedding_dim=self._opt.embedding_dim,
            latent_dim=self._opt.latent_dim,
            out_channels=self._opt.out_channels,
        )
        self._generator.init_weights(self._opt.init_type)

        self._discriminator = make_network(
            network_name="acgandiscriminator",
            opt=self._opt,
            n_classes=self._opt.n_classes,
            in_channels=self._opt.in_channels,
        )
        self._discriminator.init_weights(self._opt.init_type)
        self._networks = [self._generator, self._discriminator]
        self._send_to_device(self._networks)

    def _make_loss(self) -> None:
        """
        Creates the loss functions
        """
        self._criterion_adv = nn.BCELoss()
        self._criterion_cls = nn.CrossEntropyLoss()
        self._loss = [self._criterion_adv, self._criterion_cls]
        self._send_to_device(self._loss)

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

        self._real_loss_adv = self._criterion_adv(self._real_validity, self._valid)
        self._real_loss_cls = self._criterion_cls(
            self._real_labels_pred, self._real_labels
        )

        self._fake_images = self._generator(self._noise, self._gen_labels)
        self._fake_validity, self._fake_labels_pred = self._discriminator(
            self._fake_images.detach()
        )

        self._fake_loss_adv = self._criterion_adv(self._fake_validity, self._fake)
        self._fake_loss_cls = self._criterion_cls(
            self._fake_labels_pred, self._gen_labels
        )
        discriminator_loss = (
            self._real_loss_adv
            + self._real_loss_cls
            + self._fake_loss_adv
            + self._fake_loss_cls
        ) / 4

        # Calculate discriminator real/fake accuracy
        self._real_acc = torch.mean((self._real_validity > 0.5).float())
        self._fake_acc = torch.mean((self._fake_validity < 0.5).float())
        self._discriminator_acc = (self._real_acc + self._fake_acc) / 2

        # Calculate discriminator label accuracy
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

        self._gen_loss_adv = self._criterion_adv(self._gen_validity, self._valid)
        self._gen_loss_cls = self._criterion_cls(
            self._gen_labels_pred, self._gen_labels
        )
        gen_loss = (self._gen_loss_adv + self._gen_loss_cls) / 2

        return gen_loss

    def _get_current_performance(self, do_visualization: bool = False) -> None:
        """
        Gets the current performance of the model

        Parameters
        ----------
        do_visualization: bool
            Whether to visualize the performance

        Returns
        -------
        None
        """
        self._current_disc_performance = {
            "D_real_loss_adv": self._real_loss_adv.item(),
            "D_real_loss_cls": self._real_loss_cls.item(),
            "D_fake_loss_adv": self._fake_loss_adv.item(),
            "D_fake_loss_cls": self._fake_loss_cls.item(),
            "D_loss": self._discriminator_loss.item(),
            "D_real_acc": self._real_acc.item(),
            "D_fake_acc": self._fake_acc.item(),
            "D_acc": self._discriminator_acc.item(),
            "D_real_label_acc": self._real_label_acc.item(),
            "D_fake_label_acc": self._fake_label_acc.item(),
            "D_label_acc": self._discriminator_label_acc.item(),
        }

        self._current_gen_performance = {
            "G_loss_adv": self._gen_loss_adv.item(),
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
