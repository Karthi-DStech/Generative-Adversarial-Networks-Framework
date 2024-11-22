import argparse
import copy
import os
import sys
from typing import Tuple
import numpy as np

import torch
from torch._tensor import Tensor
import torch.nn as nn
from torch.nn import AvgPool2d
from torch.nn.functional import interpolate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.models import BaseModel
from call_methods import make_network
from utils.losses import wasserstein_loss
from utils.utils import update_average

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class StyleGAN(BaseModel):
    """
    This class implements the StyleGAN model (https://arxiv.org/abs/1812.04948). The code is adapted from
    https://github.com/huangzh13/StyleGAN.pytorch by Zhonghao
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the StyleGAN class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "StyleGAN"
        self._networks = None
        self.depth = int(np.log2(self._opt.img_size)) - 1
        self.alpha = 1
        self._structure = self._opt.structure
        if self._structure == "fixed":
            self.start_depth = self.depth - 1
        else:
            self.start_depth = self._opt.start_depth
        self.current_depth = self.start_depth
        self._use_ema = self._opt.use_ema
        self._ema_decay = self._opt.ema_decay
        self._create_networks()
        self._print_num_params()

        if self._is_train:
            self._make_loss()
            self._make_optimizer()

        if self._use_ema:
            self._create_ema()

    def _create_networks(self) -> None:
        """
        Creates the networks
        """
        self._generator = make_network(
            network_name="stylegenerator",
            num_channels=self._opt.out_channels,
            resolution=self._opt.img_size,
            structure=self._opt.structure,
            latent_dim=self._opt.latent_dim,
            mapping_layers=self._opt.mapping_layers,
            blur_filter=self._opt.blur_filter,
            truncation_psi=self._opt.truncation_psi,
            truncation_cutoff=self._opt.truncation_cutoff,
        )

        self._discriminator = make_network(
            network_name="stylediscriminator",
            num_channels=self._opt.out_channels,
            resolution=self._opt.img_size,
            structure=self._opt.structure,
            use_wscale=self._opt.use_wscale,
            blur_filter=self._opt.blur_filter,
        )
        self._networks = [self._generator, self._discriminator]
        self._send_to_device(self._networks)

    def _make_loss(self) -> None:
        """
        Creates the loss functions
        """
        self._w_loss = wasserstein_loss

    def _create_ema(self):
        """
        Creates the exponential moving average of the generator
        """
        self._generator_shadow = copy.deepcopy(self._generator)
        self._generator_shadow._name = self._generator._name + "_shadow"
        self._ema_updater = update_average
        self._networks.append(self._generator_shadow)  # type: ignore
        self._send_to_device(self._networks)  # type: ignore

        # initialize the gen_shadow weights equal to the weights of gen
        self._ema_updater(self._generator_shadow, self._generator, beta=0)

    def _progressive_downsampling(
        self, real_images: Tensor, depth: int, alpha: float
    ) -> Tensor:
        """
        Downsamples the real images to the current depth

        Parameters
        ----------
        real_images: torch.Tensor
            The real images
        depth: int
            The current depth
        alpha: float
            The current alpha for fade-in

        Returns
        -------
        torch.Tensor
            The downsampled images
        """
        if self._opt.structure == "fixed":
            return real_images

        assert depth < self.depth, "Requested output depth cannot be produced"

        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_images = AvgPool2d(down_sample_factor)(real_images)

        if depth > 0:
            prior_ds_real_images = interpolate(
                AvgPool2d(prior_down_sample_factor)(real_images), scale_factor=2
            )
        else:
            prior_ds_real_images = ds_real_images

        # real images are combination of the downsampled real images and the prior downsampled real images
        real_images = alpha * ds_real_images + (1 - alpha) * prior_ds_real_images

        return real_images

    def _forward_discriminator(self, depth: int, alpha: float) -> Tensor:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        depth: int
            The current depth
        alpha: float
            The current alpha for fade-in

        Returns
        -------
        torch.Tensor
            The discriminator loss
        """
        self._down_real_images = self._progressive_downsampling(
            self._real_images, depth, alpha
        )
        self._real_validity = self._discriminator(self._down_real_images, depth, alpha)
        self._real_loss_w = (
            self._w_loss(self._real_validity, True) * self._opt.d_lambda_w
        )

        self._fake_images = self._generator(self._noise, depth, alpha)
        self._fake_validity = self._discriminator(
            self._fake_images.detach(), depth, alpha
        )
        self._fake_loss_w = (
            self._w_loss(self._fake_validity, False) * self._opt.d_lambda_w
        )

        self._gp = (
            self._gradient_penalty(
                self._down_real_images, self._fake_images, depth, alpha
            )
            * self._opt.d_lambda_gp
        )

        discriminator_loss = self._real_loss_w + self._fake_loss_w + self._gp

        return discriminator_loss

    def _forward_generator(self, depth: int, alpha: float) -> Tensor:
        """
        Forward pass for the generator

        Returns
        -------
        torch.Tensor
            The generator loss
        """
        self._fake_images = self._generator(self._noise, depth, alpha)
        self._gen_validity = self._discriminator(self._fake_images, depth, alpha)
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

    def _gradient_penalty(
        self,
        down_sampled_real_images: Tensor,
        fake_images: Tensor,
        depth: int,
        f_alpha: float,
    ) -> Tensor:
        """
        Calculates the gradient penalty loss

        Parameters
        ----------
        down_sampled_real_images: torch.Tensor
            The downsampled real images
        fake_images: torch.Tensor
            The fake images

        Returns
        -------
        torch.Tensor
            The calculated gradient penalty
        """
        batch_size = down_sampled_real_images.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1, device=self._device).expand_as(
            down_sampled_real_images
        )
        interpolates = (
            alpha * down_sampled_real_images.data
            + ((1 - alpha) * fake_images.view_as(down_sampled_real_images).data)
        ).requires_grad_(True)
        disc_interpolates = self._discriminator(interpolates, depth, f_alpha)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates, device=self._device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

    def train(
        self,
        depth: int,
        alpha: float,
        train_generator: bool = True,
        do_visualization: bool = False,
    ) -> None:
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
        if self._use_ema:
            self._generator_shadow.train()

        if train_generator:
            self._generator.zero_grad()
            self._gen_loss = self._forward_generator(depth, alpha)
            self._gen_loss.backward()
            # Gradient Clipping
            nn.utils.clip_grad_norm_(self._generator.parameters(), max_norm=10.0)  # type: ignore
            self._generator._optimizer.step()
            if self._use_ema:
                self._ema_updater(
                    self._generator_shadow, self._generator, self._ema_decay
                )

        self._discriminator.zero_grad()
        self._discriminator_loss = self._forward_discriminator(depth, alpha)
        self._discriminator_loss.backward()
        self._discriminator._optimizer.step()
        if train_generator:
            self._get_current_performance(do_visualization)

    def test(self, do_visualization: bool = False) -> None:
        pass

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
        with torch.no_grad():
            noise = torch.randn(n_samples, self._opt.latent_dim, device=self._device)
            if not self._use_ema:
                self._vis_images = self._generator(
                    noise, self.current_depth, self.alpha
                ).detach()
            else:
                self._vis_images = self._generator_shadow(
                    noise, self.current_depth, self.alpha
                ).detach()
            scale_factor = (
                int(np.power(2, self.depth - self.current_depth - 1))
                if self._structure == "linear"
                else 1
            )
            if scale_factor > 1:
                self._vis_images = interpolate(
                    self._vis_images, scale_factor=scale_factor
                )
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


# TODO:
