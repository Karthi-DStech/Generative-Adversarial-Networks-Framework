import argparse
from collections import OrderedDict
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import numpy as np
from model.networks import BaseNetwork
from utils.custom_layers import PixelNormLayer, EqualizedLinear, EqualizedConv2d, Truncation, InputBlock, GSynthesisBlock

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ACGANGenerator(BaseNetwork):
    """
    This class implements the ACGAN generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        n_classes: int,
        embedding_dim: int,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the ACGANGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        embedding_dim: int
            The embedding dimension of the labels
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACGANGenerator"
        self._opt = opt
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels

        self.label_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.init_size = self._opt.img_size // 4
        # self.init_linear = nn.Linear(self.embedding_dim+self.latent_dim, 128*self.init_size**2)
        self.init_linear = nn.Linear(self.latent_dim, 128 * self.init_size**2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor
        labels: torch.Tensor
            The labels tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        # gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        gen_input = torch.mul(self.label_embedding(labels), noise)
        out = self.init_linear(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ConvGANGenerator(BaseNetwork):
    """
    This class implements the ConvGAN generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the ACGANGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ConvGANGenerator"
        self._opt = opt
        self.latent_dim = latent_dim
        self.out_channels = out_channels

        self.init_size = self._opt.img_size // 4
        self.init_linear = nn.Linear(self.latent_dim, 128 * self.init_size**2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        out = self.init_linear(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ACConvGANGenerator(BaseNetwork):
    """
    This class implements the Conditional ConvGAN generator
    """

    def __init__(
            self,
            opt: argparse.Namespace, 
            n_classes: int,
            embedding_dim: int, 
            latent_dim: int, 
            out_channels: int,
        ) -> None:
        """
        Initializes the ConditionalConvGANGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        embedding_dim: int
            The embedding dimension of the labels           
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACConvGANGenerator"
        self._opt = opt
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels

        self.init_size = self._opt.img_size // 4
        self.label_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.init_linear = nn.Linear(self.latent_dim, 128 * self.init_size**2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.out_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor
        labels: torch.Tensor
            The labels tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        label_embedding = self.label_embedding(labels)
        gen_input = torch.mul(label_embedding, noise)
        out = self.init_linear(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class VanillaGenerator(BaseNetwork):
    """
    This class implements the vanilla generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        g_neurons: int,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the VanillaGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        g_neurons: int
            The number of neurons in the generator
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "VanillaGenerator"
        self._opt = opt
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.out_size = self._opt.img_size * self._opt.img_size * self.out_channels

        self.linear1 = nn.Linear(self.latent_dim, g_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features * 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features * 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, self.out_size)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        out = self.linear1(noise)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear4(out)
        out = nn.Tanh()(out)
        return out


class ACVanillaGenerator(BaseNetwork):
    """
    This class implements the ACVanillaGAN generator
    """

    def __init__(
        self,
        opt: argparse.Namespace,
        n_classes: int,
        embedding_dim: int,
        g_neurons: int,
        latent_dim: int,
        out_channels: int,
    ) -> None:
        """
        Initializes the ACVanillaGenerator class

        Parameters
        ----------
        opt: argparse.Namespace
            The training options
        n_classes: int
            The number of classes
        embedding_dim: int
            The embedding dimension of the labels
        g_neurons: int
            The number of neurons in the generator
        latent_dim: int
            The latent dimension of the noise
        out_channels: int
            The number of output channels
        """
        super().__init__()
        self._name = "ACVanillaGenerator"
        self._opt = opt
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.out_size = self._opt.img_size * self._opt.img_size * self.out_channels

        self.label_embedding = nn.Embedding(self.n_classes, self.embedding_dim)
        self.linear1 = nn.Linear(self.latent_dim, g_neurons)
        self.linear2 = nn.Linear(
            self.linear1.out_features, self.linear1.out_features * 2
        )
        self.linear3 = nn.Linear(
            self.linear2.out_features, self.linear2.out_features * 2
        )
        self.linear4 = nn.Linear(self.linear3.out_features, self.out_size)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        noise: torch.Tensor
            The noise tensor
        labels: torch.Tensor
            The labels tensor

        Returns
        -------
        torch.Tensor
            The generated images
        """
        gen_input = torch.mul(self.label_embedding(labels), noise)
        out = self.linear1(gen_input)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear2(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear3(out)
        out = nn.LeakyReLU(0.2, inplace=True)(out)
        out = self.linear4(out)
        out = nn.Tanh()(out)
        return out


class GMapping(BaseNetwork):
    """
    This class implements the StyleGAN mapping network (https://arxiv.org/pdf/1812.04948.pdf)
    """

    def __init__(
        self,
        latent_dim: int = 512,
        mapping_fmaps: int = 512,
        dlatent_dim: int = 512,
        mapping_layers: int = 8,
        mapping_lrmul: float = 0.01,
        dlatent_broadcast=None,
        normalize_latents: bool = True,
        use_wscale: bool = True,
    ) -> None:
        super().__init__()
        """
        Initializes the GMapping class

        Parameters
        ----------
        latent_dim: int
            The latent dimension
        mapping_fmaps: int
            The number of feature maps in the mapping network
        dlatent_dim: int
            The dimension of the dlatent vector
        dlatent_broadcast: int
            The number of times to broadcast the dlatent vector
        mapping_layers: int
            The number of layers in the mapping network
        mapping_lrmul: float
            The learning rate multiplier to use for the mapping network
        normalize_latents: bool
            Whether to normalize the latents
        use_wscale: bool
            Whether to use the weight scaling trick
        """
        self._name = "GMapping"
        self.latent_dim = latent_dim
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_dim = dlatent_dim
        self.dlatent_broadcast = dlatent_broadcast

        layers = []
        # Normalize latent
        if normalize_latents:
            layers.append(("pixel_norm", PixelNormLayer()))

        # Mapping layers
        layers.append(
            (
                "dense0",
                EqualizedLinear(
                    input_size=self.latent_dim,
                    output_size=self.mapping_fmaps,
                    gain=np.sqrt(2),
                    use_wscale=use_wscale,
                    lrmul=mapping_lrmul,
                ),
            )
        )
        layers.append(("dense0_act", nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = (
                self.dlatent_dim
                if layer_idx == mapping_layers - 1
                else self.mapping_fmaps
            )
            layers.append(
                (
                    "dense{:d}".format(layer_idx),
                    EqualizedLinear(
                        input_size=fmaps_in,
                        output_size=fmaps_out,
                        gain=np.sqrt(2),
                        use_wscale=use_wscale,
                        lrmul=mapping_lrmul,
                    ),
                )
            )
            layers.append(
                (
                    "dense{:d}_act".format(layer_idx),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                )
            )

        self.mapping = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the mapping network

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        x = self.mapping(x)
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(BaseNetwork):
    """
    This class implements the StyleGAN synthesis network (https://arxiv.org/pdf/1812.04948.pdf)
    """

    def __init__(
        self,
        fmap_base: int = 8192,  # TODO: Check if we can change to not to be hardcoded (It is 4 * 4 * 512)
        fmap_decay: float = 1.0,
        fmap_max: int = 512,
        resolution: int = 1024,
        dlatent_dim: int = 512,
        num_channels: int = 3,
        structure: str = "fixed",
        use_styles: bool = True,
        const_input_layer: bool = True,
        use_noise: bool = True,
        use_pixel_norm: bool = False,
        use_instance_norm: bool = True,
        use_wscale: bool = True,
        blur_filter=None,
    ) -> None:
        super().__init__()
        """
        Initializes the GSynthesis class

        Parameters
        ----------
        fmap_base: int
            Overall multiplier for the number of feature maps
        fmap_decay: float
            log2 feature map reduction when doubling the resolution
        fmap_max: int
            Maximum number of feature maps in any layer
        resolution: int
            The resolution of the output image
        dlatent_dim: int
            Disentangled latent (W) dimensionality
        num_channels: int
            Number of output color channels
        structure: str
            Structure of the generator, 'fixed' = no progressive growing, 'linear' = human-readable
        use_styles: bool
            Whether to use styles or not
        const_input_layer: bool
            Whether to use a constant input layer
        use_noise: bool
            Whether to use a noise layer
        use_pixel_norm: bool
            Whether to use pixel norm
        use_instance_norm: bool
            Whether to use instance norm
        use_wscale: bool
            Whether to use the weight scaling trick
        blur_filter: list
            Low pass filter to apply when resampling activations. None = no filtering
        """

        self._name = "GSynthesis"

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure
        resolution_log2 = int(np.log2(resolution))
        assert (
            resolution == 2**resolution_log2 and resolution >= 4
        ), "resolution must be a power of 2 and >= 4"
        self.depth = resolution_log2 - 1

        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        # Early layers
        self.init_block = InputBlock(
            nf=nf(1),
            dlatent_dim=dlatent_dim,
            gain=np.sqrt(2),
            const_input_layer=const_input_layer,
            use_noise=use_noise,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_styles=use_styles,
            use_wscale=use_wscale,
        )

        # create the ToRGB layers for various outputs
        color_converters = [
            EqualizedConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)
        ]

        # Building blocks for remaining layers
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            blocks.append(
                GSynthesisBlock(
                    in_channels=last_channels,
                    num_channels=channels,
                    blur_filter=blur_filter,
                    dlatent_dim=dlatent_dim,
                    gain=np.sqrt(2),
                    use_noise=use_noise,
                    use_pixel_norm=use_pixel_norm,
                    use_instance_norm=use_instance_norm,
                    use_styles=use_styles,
                    use_wscale=use_wscale,
                )
            )
            color_converters.append(
                EqualizedConv2d(
                    channels, num_channels, 1, gain=1, use_wscale=use_wscale
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.color_converters = nn.ModuleList(color_converters)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(
        self,
        dlatents_in: torch.Tensor,
        depth: int = 0,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass for the synthesis network

        Parameters
        ----------
        dlatents_in: torch.Tensor
            The disentangled latents (W) [mini_batch, num_layers, dlatent_size].
        depth: int
            Current depth from where output is required
        alpha: float
            Coefficient for fade-in effect

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == "fixed":
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1) : 2 * (i + 2)])
            images_out = self.color_converters[-1](x)

        elif self.structure == "linear":
            x = self.init_block(dlatents_in[:, 0:2])

            if depth > 0:
                for i, block in enumerate(self.blocks[: depth - 1]):  # type: ignore
                    x = block(x, dlatents_in[:, 2 * (i + 1) : 2 * (i + 2)])
                residual = self.color_converters[depth - 1](self.temporaryUpsampler(x))
                straight = self.color_converters[depth](
                    self.blocks[depth - 1](
                        x, dlatents_in[:, 2 * depth : 2 * (depth + 1)]
                    )
                )
                images_out = (alpha * straight) + ((1 - alpha) * residual)

            else:
                images_out = self.color_converters[0](x)

        else:
            raise ValueError(f"Unknown structure: {self.structure}")

        return images_out


class StyleGenerator(BaseNetwork):
    """
    This class implements the StyleGAN generator (https://arxiv.org/pdf/1812.04948.pdf)
    """

    def __init__(
        self,
        fmap_base: int = 8192,
        fmap_decay: float = 1.0,
        resolution: int = 1024,
        fmap_max: int = 512,
        latent_dim: int = 512,
        dlatent_dim: int = 512,
        mapping_fmaps: int = 512,
        num_channels: int = 3,
        structure: str = "fixed",
        mapping_layers: int = 8,
        mapping_lrmul: float = 0.01,
        style_mixing_prob: float = 0.9,
        dlatent_avg_beta: float = 0.995,
        truncation_psi=None,
        truncation_cutoff: int = 8,
        blur_filter=None,
        normalize_latents: bool = True,
        use_wscale: bool = True,
        use_styles: bool = True,
        const_input_layer: bool = True,
        use_noise: bool = True,
        use_pixel_norm: bool = False,
        use_instance_norm: bool = True,
    ) -> None:
        """
        Initializes the StyleGenerator class

        Parameters
        ----------
        fmap_base: int
            Overall multiplier for the number of feature maps
        fmap_decay: float
            log2 feature map reduction when doubling the resolution
        fmap_max: int
            Maximum number of feature maps in any layer
        resolution: int
            The resolution of the output image
        latent_dim: int
            The latent dimension
        dlatent_dim: int
            The dimension of the dlatent vector
        mapping_fmaps: int
            The number of feature maps in the mapping network
        num_channels: int
            Number of output color channels
        structure: str
            Structure of the generator, 'fixed' = no progressive growing, 'linear' = human-readable
        mapping_layers: int
            The number of layers in the mapping network
        mapping_lrmul: float
            The learning rate multiplier to use for the mapping network
        style_mixing_prob: float
            The probability of mixing styles, None = disable
        dlatent_avg_beta: float
            Decay for tracking the moving average of W during training.
        truncation_psi: float
            Style strength multiplier for the truncation trick. None = disable.
        truncation_cutoff: int
            Number of layers for which to apply the truncation trick.
        blur_filter: list
            Low pass filter to apply when resampling activations. None = no filtering
        normalize_latents: bool
            Whether to normalize the latents
        use_wscale: bool
            Whether to use the weight scaling trick
        use_styles: bool
            Whether to use styles or not
        const_input_layer: bool
            Whether to use a constant input layer
        use_noise: bool
            Whether to use a noise layer
        use_pixel_norm: bool
            Whether to use pixel norm
        use_instance_norm: bool
            Whether to use instance norm
        """
        super().__init__()
        self._name = "StyleGenerator"
        self._style_mixing_prob = style_mixing_prob
        self._num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(
            latent_dim=latent_dim,
            mapping_fmaps=mapping_fmaps,
            dlatent_dim=dlatent_dim,
            mapping_layers=mapping_layers,
            mapping_lrmul=mapping_lrmul,
            dlatent_broadcast=self._num_layers,
            normalize_latents=normalize_latents,
            use_wscale=use_wscale,
        )
        self.g_synthesis = GSynthesis(
            fmap_base=fmap_base,
            fmap_decay=fmap_decay,
            fmap_max=fmap_max,
            resolution=resolution,
            dlatent_dim=dlatent_dim,
            num_channels=num_channels,
            structure=structure,
            use_styles=use_styles,
            const_input_layer=const_input_layer,
            use_noise=use_noise,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_wscale=use_wscale,
            blur_filter=blur_filter,
        )

        if truncation_psi is not None:
            self.truncation = Truncation(
                avg_latent=torch.zeros(dlatent_dim),
                max_layer=truncation_cutoff,
                threshold=truncation_psi,
                beta=dlatent_avg_beta,
            )
        else:
            self.truncation = None

    def forward(
        self, latents_in: torch.Tensor, depth: int, alpha: float
    ) -> torch.Tensor:
        """
        Forward pass for the generator

        Parameters
        ----------
        latents_in: torch.Tensor
            The input latent tensor
        depth: int
            The depth of the layer from which output is required
        alpha: float
            Coefficient for fade-in effect

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        dlatents_in = self.g_mapping(latents_in)

        # Update moving average of W(dlatent).
        if self.truncation is not None:
            self.truncation.update(dlatents_in[0, 0].detach())

        # Perform style mixing regularization
        if self._style_mixing_prob is not None and self._style_mixing_prob > 0:
            second_latents_in = torch.randn(latents_in.shape).to(latents_in.device)
            second_dlatents_in = self.g_mapping(second_latents_in)
            layer_idx = torch.from_numpy(
                np.arange(self._num_layers)[np.newaxis, :, np.newaxis]
            ).to(latents_in.device)
            cur_layers = 2 * (depth + 1)
            mixing_cutoff = (
                random.randint(1, cur_layers)
                if random.random() < self._style_mixing_prob
                else cur_layers
            )
            dlatents_in = torch.where(
                layer_idx < mixing_cutoff, dlatents_in, second_dlatents_in
            )

        # Apply truncation trick.
        if self.truncation is not None:
            dlatents_in = self.truncation(dlatents_in)

        # Generate images.
        img = self.g_synthesis(dlatents_in, depth, alpha)
        return img


if __name__ == "__main__":
    from options.train_option import TrainOptions

    opt = TrainOptions().parse()
    gen = StyleGenerator(
        resolution=1024,
        structure="linear",
        style_mixing_prob=opt.style_mixing_prob,
        truncation_psi=opt.truncation_psi,
    )
    # print(gen)
    # print(f'Number of parameters: {gen.get_num_params()}')
    latents = torch.randn(1, 512)
    print(f"latents shape: {latents.shape}")
    out = gen(latents, 7, 1.0)
    print(f"out shape: {out.shape}")
