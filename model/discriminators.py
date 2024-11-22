import argparse
import os
import sys
from typing import List, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn

from model.networks import BaseNetwork
from utils.custom_layers import DiscriminatorBlock, EqualizedConv2d, DiscriminatorTop

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ACGANDiscriminator(BaseNetwork):
    """
    This class implements the ACGANDiscriminator
    """

    def __init__(
        self, opt: argparse.Namespace, n_classes: int, in_channels: int
    ) -> None:
        """
        Initializes the ACGANDiscriminator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        in_channels: int
            The number of input channels
        """
        super().__init__()
        self._name = "ACGANDiscriminator"
        self._opt = opt

        self.conv_blocks = nn.Sequential(
            *self._discriminator_block(in_channels, 16, bn=False),
            *self._discriminator_block(16, 32),
            *self._discriminator_block(32, 64),
            *self._discriminator_block(64, 128),
        )

        # Dimension of output feature map after conv_blocks
        self.dim = self._opt.img_size // 2**4

        self.adv_layer = nn.Linear(128 * self.dim**2, 1)
        self.aux_layer = nn.Linear(128 * self.dim**2, n_classes)

    def _discriminator_block(
        self, in_filters: int, out_filters: int, bn: bool = True
    ) -> List:
        """
        Creates a discriminator block

        Parameters
        ----------
        in_filters: int
            The number of input filters
        out_filters: int
            The number of output filters
        bn: bool
            Whether to use batch normalization or not

        Returns
        -------
        list
            The discriminator block
        """
        block = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        ]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        img: torch.Tensor
            The input image

        Returns
        -------
        validity: torch.Tensor
            The validity of the image (real or fake)
        label: torch.Tensor
            The label of the image
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        validity = torch.sigmoid(validity)
        label = self.aux_layer(out)
        # label = torch.softmax(label, dim=1)
        return validity, label


class VanillaDiscriminator(BaseNetwork):
    """
    This class implements the VanillaDiscriminator
    """

    def __init__(
        self, opt: argparse.Namespace, d_neurons: int, out_channels: int
    ) -> None:
        """
        Initializes the VanillaDiscriminator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        d_neurons: int
            The number of neurons in the discriminator
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "VanillaDiscriminator"
        self._opt = opt
        self.in_size = self._opt.img_size * self._opt.img_size * out_channels

        self.linear1 = nn.Linear(self.in_size, d_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        torch.Tensor
            The validity of the image (real or fake)
        """
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear4(out)
        validity = torch.sigmoid(out)
        return validity


class ACVanillaDiscriminator(BaseNetwork):
    """
    This class implements the ACVanillaDiscriminator
    """

    def __init__(
        self, opt: argparse.Namespace, n_classes: int, d_neurons: int, out_channels: int
    ) -> None:
        """
        Initializes the ACVanillaDiscriminator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        d_neurons: int
            The number of neurons in the discriminator
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACVanillaDiscriminator"
        self._opt = opt
        self.in_size = self._opt.img_size * self._opt.img_size * out_channels
        self.n_classes = n_classes

        self.linear1 = nn.Linear(self.in_size, d_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )
        self.adv_layer = nn.Linear(self.linear3.out_features, 1)
        self.aux_layer = nn.Linear(self.linear3.out_features, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        forward pass for the discriminator

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        validity: torch.Tensor
            The validity of the image (real or fake)
        label: torch.Tensor
            The label of the image
        """
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2)(out)
        out = nn.Dropout(0.3)(out)
        validity = self.adv_layer(out)
        validity = torch.sigmoid(validity)
        label = self.aux_layer(out)
        return validity, label


class WGANCritic(BaseNetwork):
    """
    This class implements the WGANCritic
    """

    def __init__(
        self, opt: argparse.Namespace, d_neurons: int, out_channels: int
    ) -> None:
        """
        Initializes the WGANCritic class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        d_neurons: int
            The number of neurons in the critic
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "WGANCritic"
        self._opt = opt
        self.in_size = self._opt.img_size * self._opt.img_size * out_channels

        self.linear1 = nn.Linear(self.in_size, d_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the critic

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        torch.Tensor
            The realness of the image
        """
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear4(out)
        validity = out
        return validity


class ACWGANCritic(BaseNetwork):
    """
    This class implements the ACWGANCritic
    """

    def __init__(
        self, opt: argparse.Namespace, n_classes: int, d_neurons: int, out_channels: int
    ) -> None:
        """
        Initializes the WGANCritic class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        d_neurons: int
            The number of neurons in the critic
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACWGANCritic"
        self._opt = opt
        self.in_size = self._opt.img_size * self._opt.img_size * out_channels
        self.n_classes = n_classes

        self.linear1 = nn.Linear(self.in_size, d_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features // 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features // 2
        )
        self.adv_layer = nn.Linear(self.linear3.out_features, 1)
        self.aux_layer = nn.Linear(self.linear3.out_features, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the critic

        Parameters
        ----------
        x: torch.Tensor
            The input image

        Returns
        -------
        validity: torch.Tensor
            The realness of the image
        label: torch.Tensor
            The label of the image
        """
        x = x.view(x.shape[0], -1)
        out = self.linear1(x)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2)(out)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

class ACWCGANCritic(BaseNetwork):
    """
    This class implements the ACWCGANCritic
    """

    def __init__(
            self, 
            opt: argparse.Namespace, 
            n_classes: int, 
            in_channels: int
            ) -> None:
        """
        Initializes the ACWCGANCritic class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        in_channels: int
            The number of input channels
        """
        super().__init__()
        self._name = "ACWCGANCritic"
        self._opt = opt
        self.n_classes = n_classes

        self.conv_blocks = nn.Sequential(
            *self._discriminator_block(in_channels, 16, bn=False),
            *self._discriminator_block(16, 32),
            *self._discriminator_block(32, 64),
            *self._discriminator_block(64, 128),
        )

        # Dimension of output feature map after conv_blocks
        self.dim = self._opt.img_size // 2**4

        self.adv_layer = nn.Linear(128 * self.dim**2, 1)
        self.aux_layer = nn.Linear(128 * self.dim**2, n_classes)

    def _discriminator_block(
        self, in_filters: int, out_filters: int, bn: bool = True
    ) -> List:
        """
        Creates a discriminator block

        Parameters
        ----------
        in_filters: int
            The number of input filters
        out_filters: int
            The number of output filters
        bn: bool
            Whether to use batch normalization or not

        Returns
        -------
        list
            The discriminator block
        """
        block = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        ]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    def forward(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the critic

        Parameters
        ----------
        img: torch.Tensor
            The input image

        Returns
        -------
        validity: torch.Tensor
            The realness of the image
        label: torch.Tensor
            The label of the image
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        return validity, label

class ConvGANCritic(BaseNetwork):
    """
    This class implements the ConvGANCritic
    """

    def __init__(self, opt: argparse.Namespace, in_channels: int) -> None:
        """
        Initializes the ConvGANCritic class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        in_channels: int
            The number of input channels
        """
        super().__init__()
        self._name = "ConvGANCritic"
        self._opt = opt

        self.conv_blocks = nn.Sequential(
            *self._discriminator_block(in_channels, 16, bn=False),
            *self._discriminator_block(16, 32),
            *self._discriminator_block(32, 64),
            *self._discriminator_block(64, 128),
        )

        # Dimension of output feature map after conv_blocks
        self.dim = self._opt.img_size // 2**4

        self.real_layer = nn.Linear(128 * self.dim**2, 1)

    def _discriminator_block(
        self, in_filters: int, out_filters: int, bn: bool = True
    ) -> List:
        """
        Creates a discriminator block

        Parameters
        ----------
        in_filters: int
            The number of input filters
        out_filters: int
            The number of output filters
        bn: bool
            Whether to use batch normalization or not

        Returns
        -------
        list
            The discriminator block
        """
        block = [
            nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        ]
        if bn:
            block.append(nn.BatchNorm2d(out_filters, 0.8))
        return block

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        img: torch.Tensor
            The input image

        Returns
        -------
        torch.Tensor
            The realness of the image
        """
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.real_layer(out)
        return validity


class StyleDiscriminator(BaseNetwork):
    """
    This class implements the StyleDiscriminator
    """

    def __init__(
        self,
        fmap_base: int = 8192,
        fmap_decay: float = 1.0,
        fmap_max: int = 512,
        resolution: int = 1024,
        num_channels: int = 3,
        mbstd_num_features: int = 1,
        mbstd_group_size: int = 4,
        structure: str = "fixed",
        use_wscale: bool = True,
        blur_filter=None,
    ) -> None:
        super().__init__()
        """
        Initializes the StyleDiscriminator class

        Parameters
        ----------
        fmap_base: int
            Overall multiplier for the number of feature maps
        fmap_decay: float
            log2 feature map reduction when doubling the resolution
        fmap_max: int
            Maximum number of feature maps in any layer
        resolution: int
            Resolution of the generated images
        num_channels: int
            The number of output channels
        mbstd_num_features: int
            Number of features for the minibatch standard deviation layer
        mbstd_group_size: int
            Group size for the minibatch standard deviation layer. 0 = disable.
        structure: str
            Structure of the generator, 'fixed' = no progressive growing, 'linear' = human-readable
        use_wscale: bool
            Enable equalized learning rate
        blur_filter: list
            Low pass filter to apply when resampling activations. None = disable blurring
        """
        self._name = "StyleDiscriminator"

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2**resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        blocks = []
        color_converters = []
        for res in range(resolution_log2, 2, -1):
            blocks.append(
                DiscriminatorBlock(
                    in_channels=nf(res - 1),
                    num_channels=nf(res - 2),
                    gain=np.sqrt(2),
                    use_wscale=use_wscale,
                    blur_filter=blur_filter,
                )
            )
            color_converters.append(
                EqualizedConv2d(
                    input_channels=num_channels,
                    output_channels=nf(res - 1),
                    kernel_size=1,
                    gain=np.sqrt(2),
                    use_wscale=use_wscale,
                )
            )

        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(
            mbstd_num_features=self.mbstd_num_features,
            mbstd_group_size=self.mbstd_group_size,
            in_channels=nf(2),
            intermediate_channels=nf(2),
            gain=np.sqrt(2),
            use_wscale=use_wscale,
        )
        color_converters.append(
            EqualizedConv2d(
                input_channels=num_channels,
                output_channels=nf(2),
                kernel_size=1,
                gain=np.sqrt(2),
                use_wscale=use_wscale,
            )
        )

        self.color_converters = nn.ModuleList(color_converters)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(
        self, images_in: torch.Tensor, depth: int, alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass for the discriminator

        Parameters
        ----------
        images_in: torch.Tensor
            The input image [mini_batch, channel, height, width]
        depth: int
            current height of operation (Progressive GAN)
        alpha: float
            The alpha value for the fade-in

        Returns
        -------
        torch.Tensor
            The validity of the image (real or fake)
        """
        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == "fixed":
            x = self.color_converters[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)

        elif self.structure == "linear":
            if depth > 0:
                residual = self.color_converters[self.depth - depth](
                    self.temporaryDownsampler(images_in)
                )
                straight = self.blocks[self.depth - depth - 1](
                    self.color_converters[self.depth - depth - 1](images_in)
                )
                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[(self.depth - depth) :]:  # type: ignore
                    x = block(x)

            else:
                x = self.color_converters[-1](images_in)

            scores_out = self.final_block(x)

        else:
            raise ValueError(f"Unknown structure: '{self.structure}'")

        return scores_out


if __name__ == "__main__":
    disc = StyleDiscriminator(
        fmap_base=8192, resolution=1024, num_channels=3, structure="linear"
    )
    print(f"Number of parameters: {disc.get_num_params()}")
    images = torch.randn(1, 3, 16, 16)
    scores = disc(images, depth=2)
    print(scores)
