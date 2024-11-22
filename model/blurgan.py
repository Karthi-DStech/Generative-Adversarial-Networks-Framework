import argparse
import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.wgan import ACWGAN_GP, WGAN_GP
from utils.losses import wasserstein_loss
from call_methods import make_network


class BlurGAN(WGAN_GP):
    """
    This class implements the BlurGAN model based on WGAN with gradient penalty
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the BlurGAN class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "BlurGAN"
        self._kernel_size = self._opt.blur_kernel_size
        if self._kernel_size % 2 == 0:
            self._kernel_size += 1
            print(
                f"Kernel size should be odd. Changing the kernel size to {self._kernel_size}"
            )

        if self._is_train:
            self._make_loss()
            self._make_optimizer()

    def _make_loss(self) -> None:
        """
        Creates the loss functions
        """
        self._w_loss = wasserstein_loss
        self._mse_loss = nn.MSELoss()

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

    def _forward_generator(self) -> torch.Tensor:
        """
        Forward pass for the generator

        Returns
        -------
        torch.Tensor
            The generator loss
        """
        self._fake_images = self._generator(self._noise)
        self._reshaped_fakes = self._reshape_fake(self._fake_images)
        self._threshold_fakes = self._apply_blur_and_threshold(
            self._reshaped_fakes, self._kernel_size
        )

        self._gen_validity = self._discriminator(self._fake_images)
        self._wasserstein_loss = (
            self._w_loss(self._gen_validity, True) * self._opt.g_lambda_w
        )
        self._blur_loss = (
            self._mse_loss(self._reshaped_fakes, self._threshold_fakes)
            * self._opt.g_lambda_blur
        )

        generator_loss = (self._wasserstein_loss + self._blur_loss) / 2

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
            "G_wasserstein_loss": self._wasserstein_loss.item(),
            "G_blur_loss": self._blur_loss.item(),
            "G_loss": self._gen_loss.item(),
        }
        self._current_performance = {
            **self._current_disc_performance,
            **self._current_gen_performance,
        }

        if do_visualization:
            fakes, lables = self._get_generated_image(n_samples=self._opt.n_vis_samples)
            threshold_fakes = self._threshold_fakes[: self._opt.n_vis_samples]
            fakes = torch.stack([fakes, threshold_fakes], dim=1).view(
                -1, fakes.shape[1], fakes.shape[2], fakes.shape[3]
            )
            lables = torch.concatenate([lables, lables], dim=0)
            self.vis_data = (fakes, lables)

        gen_lr = self._generator._optimizer.param_groups[0]["lr"]
        disc_lr = self._discriminator._optimizer.param_groups[0]["lr"]

        self.performance = {
            "losses": self._current_performance,
            "lr": {"G_lr": gen_lr, "D_lr": disc_lr},
        }

    def _reshape_fake(self, fake_images: torch.Tensor) -> torch.Tensor:
        """
        Reshape the fake images to perform morphological operations on them

        Parameters
        ----------
        fake_images: Tensor
            The fake images

        Returns
        -------
        reshaped_fakes: Tensor
            The reshaped fake images
        """
        reshaped_fakes = fake_images.view(
            fake_images.size(0),
            self._opt.out_channels,
            self._opt.img_size,
            self._opt.img_size,
        )
        return reshaped_fakes

    def _apply_blur_and_threshold(
        self, images: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        """
        Apply blur and threshold to the images

        Parameters
        ----------
        images: torch.Tensor
            The images to apply blur and threshold
        kernel_size: int
            The kernel size for the blur

        Returns
        -------
        threshold_imgs: torch.Tensor
            The threshold images
        """

        blurred_imgs = TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size])
        threshold_imgs = (
            blurred_imgs > blurred_imgs.mean(dim=(2, 3), keepdim=True)
        ).float()
        threshold_imgs = 2 * threshold_imgs - 1

        return threshold_imgs


class ACBlurGAN(ACWGAN_GP):
    """
    This class implements the ACBlurGAN model based on ACWGAN with gradient penalty
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initializes the ACBlurGAN class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        super().__init__(opt)
        self._name = "ACBlurGAN"
        self._kernel_size = self._opt.blur_kernel_size
        if self._kernel_size % 2 == 0:
            self._kernel_size += 1
            print(
                f"Kernel size should be odd. Changing the kernel size to {self._kernel_size}"
            )

        if self._is_train:
            self._make_loss()
            self._make_optimizer()

    def _make_loss(self) -> None:
        """
        Creates the loss functions
        """
        self._w_loss = wasserstein_loss
        self._criterion_cls = nn.CrossEntropyLoss()
        self._mse_loss = nn.MSELoss()

    def _forward_generator(self) -> torch.Tensor:
        """
        Forward pass for the generator

        Returns
        -------
        torch.Tensor
            The generator loss
        """
        self._fake_images = self._generator(self._noise, self._gen_labels)
        self._reshaped_fakes = self._reshape_fake(self._fake_images)
        self._threshold_fakes = self._apply_blur_and_threshold(
            self._reshaped_fakes, self._kernel_size
        )

        self._gen_validity, self._gen_labels_pred = self._discriminator(
            self._fake_images
        )
        self._wasserstein_loss = (
            self._w_loss(self._gen_validity, True) * self._opt.g_lambda_w
        )
        self._gen_loss_cls = (
            self._criterion_cls(self._gen_labels_pred, self._gen_labels)
            * self._opt.g_lambda_cls
        )
        self._blur_loss = (
            self._mse_loss(self._reshaped_fakes, self._threshold_fakes)
            * self._opt.g_lambda_blur
        )

        generator_loss = (
            self._wasserstein_loss + self._gen_loss_cls + self._blur_loss
        ) / 3

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
            "G_wasserstein_loss": self._wasserstein_loss.item(),
            "G_loss_cls": self._gen_loss_cls.item(),
            "G_blur_loss": self._blur_loss.item(),
            "G_loss": self._gen_loss.item(),
        }
        self._current_performance = {
            **self._current_disc_performance,
            **self._current_gen_performance,
        }

        if do_visualization:
            fakes, lables = self._get_generated_image(n_samples=self._opt.n_vis_samples)
            threshold_fakes = self._threshold_fakes[: self._opt.n_vis_samples]
            fakes = torch.stack([fakes, threshold_fakes], dim=1).view(
                -1, fakes.shape[1], fakes.shape[2], fakes.shape[3]
            )
            lables = torch.concatenate([lables, lables], dim=0)
            self.vis_data = (fakes, lables)

        gen_lr = self._generator._optimizer.param_groups[0]["lr"]
        disc_lr = self._discriminator._optimizer.param_groups[0]["lr"]

        self.performance = {
            "losses": self._current_performance,
            "lr": {"G_lr": gen_lr, "D_lr": disc_lr},
        }

    def _reshape_fake(self, fake_images: torch.Tensor) -> torch.Tensor:
        """
        Reshape the fake images to perform morphological operations on them

        Parameters
        ----------
        fake_images: Tensor
            The fake images

        Returns
        -------
        reshaped_fakes: Tensor
            The reshaped fake images
        """
        reshaped_fakes = fake_images.view(
            fake_images.size(0),
            self._opt.out_channels,
            self._opt.img_size,
            self._opt.img_size,
        )
        return reshaped_fakes

    def _apply_blur_and_threshold(
        self, images: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        """
        Apply blur and threshold to the images

        Parameters
        ----------
        images: torch.Tensor
            The images to apply blur and threshold
        kernel_size: int
            The kernel size for the blur

        Returns
        -------
        threshold_imgs: torch.Tensor
            The threshold images
        """

        blurred_imgs = TF.gaussian_blur(images, kernel_size=[kernel_size, kernel_size])
        threshold_imgs = (
            blurred_imgs > blurred_imgs.mean(dim=(2, 3), keepdim=True)
        ).float()
        threshold_imgs = 2 * threshold_imgs - 1

        return threshold_imgs


class ACCBlurGAN(ACBlurGAN):
    """
    This class implements the ACCBlurGAN model with gradient penalty.
    It is based on the ACBlurGAN model but uses convolution layers as the backbone.
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        super().__init__(opt)
        self._name = "ACCBlurGAN"

    def _create_networks(self) -> None:
        """
        Creates the networks
        """
        self._generator = make_network(
            network_name="acconvgangenerator",
            opt=self._opt,
            n_classes=self._opt.n_classes,
            embedding_dim=self._opt.embedding_dim,
            latent_dim=self._opt.latent_dim,
            out_channels=self._opt.out_channels,
        )
        self._generator.init_weights(self._opt.init_type)

        self._discriminator = make_network(
            network_name="acwcgancritic",
            opt=self._opt,
            n_classes=self._opt.n_classes,
            in_channels=self._opt.in_channels,
        )
        self._discriminator.init_weights(self._opt.init_type)
        self._networks = [self._generator, self._discriminator]
        self._send_to_device(self._networks)


if __name__ == "__main__":
    from options.train_option import TrainOptions

    opt = TrainOptions().parse()
    model = BlurGAN(opt)
    model._noise = torch.randn(32, 112).float()
    model._forward_generator()
