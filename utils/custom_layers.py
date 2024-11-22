from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn


class PixelNormLayer(nn.Module):
    """
    This class implements the pixelwise feature vector normalization layer (https://arxiv.org/pdf/1710.10196.pdf)

    Formulation:
    PixelNormLayer(x) = x * (1 / sqrt(mean(x**2) + epsilon)
    """

    def __init__(self, epsilon=1e-8) -> None:
        super().__init__()
        """
        Initializes the PixelNormLayer class
        
        Parameters
        ----------
        epsilon: float
            The epsilon value to use for numerical stability
        """
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the pixel norm layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The normalized input tensor
        """
        return x * torch.rsqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class BlurLayer(nn.Module):
    """
    This class implements the blur layer (https://arxiv.org/pdf/1710.10196.pdf)
    """

    def __init__(self, kernel=None, normalize=True, flip=False, stride=1) -> None:
        super(BlurLayer, self).__init__()
        """
        Initializes the BlurLayer class

        Parameters
        ----------
        kernel: list
            The kernel to use for blurring
        normalize: bool
            Whether to use the normalize value
        flip: bool
            Whether to use the flip value
        stride: int
            The stride value
        """
        if kernel is None:
            kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer("kernel", kernel)
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the blur layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The blurred tensor
        """
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = nn.functional.conv2d(
            x,
            kernel,
            stride=self.stride,
            padding=int((self.kernel.size(2) - 1) / 2),
            groups=x.size(1),
        )
        return x


class Upscale2d(nn.Module):
    """
    This class implements upscaling layer
    """

    def __init__(self, factor=2, gain=1) -> None:
        super().__init__()
        """
        Initializes the Upscale2d class
        
        Parameters
        ----------
        factor: int
            The upscaling factor
        gain: float
            The gain value to use
        """
        assert isinstance(factor, int) and factor >= 1
        self.gain = gain
        self.factor = factor

    @staticmethod
    def upscale2d(x: torch.Tensor, factor=2, gain=1):
        """
        Upscales the input tensor by the specified factor

        Parameters
        ----------
        x: torch.Tensor
            The input tensor
        factor: int
            The upscaling factor
        gain: float
            The gain value to use

        Returns
        -------
        torch.Tensor
            The upscaled tensor
        """
        assert x.dim() == 4
        if gain != 1:
            x = x * gain
        if factor != 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(
                -1, -1, -1, factor, -1, factor
            )
            x = x.contiguous().view(
                shape[0], shape[1], factor * shape[2], factor * shape[3]
            )
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the upscaling layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        return self.upscale2d(x, factor=self.factor, gain=self.gain)


class Downscale2d(nn.Module):
    """
    This class implements downscaling layer
    """

    def __init__(self, factor=2, gain=1) -> None:
        super().__init__()
        """
        Initializes the Downscale2d class

        Parameters
        ----------
        factor: int
            The downscaling factor
        gain: float
            The gain value to use
        """
        assert isinstance(factor, int) and factor >= 1
        self.factor = factor
        self.gain = gain
        if factor == 2:
            f = [np.sqrt(gain) / factor] * factor
            self.blur = BlurLayer(kernel=f, normalize=False, stride=factor)
        else:
            self.blur = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the downscaling layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        assert x.dim() == 4
        # 2x2, float32 => downscale using _blur2d().
        if self.blur is not None and x.dtype == torch.float32:
            return self.blur(x)

        # Apply gain.
        if self.gain != 1:
            x = x * self.gain

        # No-op => early exit.
        if self.factor == 1:
            return x

        # Large factor => downscale using tf.nn.avg_pool().
        # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
        return nn.functional.avg_pool2d(x, self.factor)


class EqualizedLinear(nn.Module):
    """
    This class implements the equalized linear layer (https://arxiv.org/pdf/1710.10196.pdf)

    Formulation:
    EqualizedLinear(x) = (x * weight + bias) * scale
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        gain: float = 2 ** (0.5),
        lrmul: float = 1.0,
        bias: bool = True,
        use_wscale: bool = False,
    ) -> None:
        super().__init__()
        """
        Initializes the EqualizedLinear class
        
        Parameters
        ----------
        input_size: int
            The input size
        output_size: int
            The output size
        gain: float
            The gain value to use
        lrmul: float
            The learning rate multiplier value
        bias: bool
            Whether to use the bias value
        use_wscale: bool
            Whether to use the scale value
        """
        he_std = gain * input_size ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.wscale = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.wscale = lrmul
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the equalized linear layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return nn.functional.linear(x, self.weight * self.wscale, bias)


class EqualizedConv2d(nn.Module):
    """
    This class implements the equalized 2D convolution layer (https://arxiv.org/pdf/1710.10196.pdf)
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        gain=2**0.5,
        lrmul=1,
        use_wscale=False,
        bias=True,
        intermediate=None,
        upscale=False,
        downscale=False,
    ) -> None:
        super().__init__()
        """
        Initializes the EqualizedConv2d class

        Parameters
        ----------
        input_channels: int
            The number of input channels
        output_channels: int
            The number of output channels
        kernel_size: int
            The kernel size
        stride: int
            The stride value
        gain: float
            The gain value to use
        lrmul: float
            The learning rate multiplier value
        use_wscale: bool
            Whether to use the scale value
        bias: bool
            Whether to use the bias value
        intermediate: torch.Tensor
            The intermediate tensor
        upscale: bool
            Whether to use the upscale value
        downscale: bool
            Whether to use the downscale value
        """
        if upscale:
            self.upscale = Upscale2d()
        else:
            self.upscale = None
        if downscale:
            self.downscale = Downscale2d()
        else:
            self.downscale = None
        he_std = gain * (input_channels * kernel_size**2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size)
            * init_std
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_channels))
            self.b_mul = lrmul
        else:
            self.bias = None
        self.intermediate = intermediate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the equalized 2D convolution layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul

        have_convolution = False
        if self.upscale is not None and min(x.shape[2:]) * 2 >= 128:
            # this is the fused upscale + conv from StyleGAN, sadly this seems incompatible with the non-fused way
            # TODO: # this really needs to be cleaned up and go into the conv...
            w = self.weight * self.w_mul
            w = w.permute(1, 0, 2, 3)
            # probably applying a conv on w would be more efficient. also this quadruples the weight (average)?!
            w = torch.nn.functional.pad(w, (1, 1, 1, 1))
            w = (
                w[:, :, 1:, 1:]
                + w[:, :, :-1, 1:]
                + w[:, :, 1:, :-1]
                + w[:, :, :-1, :-1]
            )
            x = torch.nn.functional.conv_transpose2d(
                x, w, stride=2, padding=(w.size(-1) - 1) // 2
            )
            have_convolution = True
        elif self.upscale is not None:
            x = self.upscale(x)

        downscale = self.downscale
        intermediate = self.intermediate
        if downscale is not None and min(x.shape[2:]) >= 128:
            w = self.weight * self.w_mul
            w = torch.nn.functional.pad(w, [1, 1, 1, 1])
            # in contrast to upscale, this is a mean...
            w = (
                w[:, :, 1:, 1:]
                + w[:, :, :-1, 1:]
                + w[:, :, 1:, :-1]
                + w[:, :, :-1, :-1]
            ) * 0.25  # avg_pool?
            x = torch.nn.functional.conv2d(
                x, w, stride=2, padding=(w.size(-1) - 1) // 2
            )
            have_convolution = True
            downscale = None
        elif downscale is not None:
            assert intermediate is None
            intermediate = downscale

        if not have_convolution and intermediate is None:
            return torch.nn.functional.conv2d(
                x, self.weight * self.w_mul, bias, padding=self.kernel_size // 2
            )
        elif not have_convolution:
            x = torch.nn.functional.conv2d(
                x, self.weight * self.w_mul, None, padding=self.kernel_size // 2
            )

        if intermediate is not None:
            x = intermediate(x)

        if bias is not None:
            x = x + bias.view(1, -1, 1, 1)
        return x


class NoiseLayer(nn.Module):
    """
    This class implements the noise layer. noise is per pixel (constant over channels) with per-channel weight.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        """
        Initializes the NoiseLayer class
        
        Parameters
        ----------
        channels: int
            The number of channels
        """
        self.weight = nn.Parameter(torch.zeros(channels))
        self.noise = None

    def forward(self, x: torch.Tensor, noise=None) -> torch.Tensor:
        """
        Forward pass for the noise layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        if noise is None and self.noise is None:
            noise = torch.randn(
                x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype
            )
        if noise is None:
            noise = self.noise
        return x + self.weight.view(1, -1, 1, 1) * noise


class Truncation(nn.Module):
    """
    This class implements the truncation layer
    """

    def __init__(
        self,
        avg_latent: torch.Tensor,
        max_layer: int = 8,
        threshold: float = 0.7,
        beta=0.995,
    ) -> None:
        super().__init__()
        """
        Initializes the Truncation class

        Parameters
        ----------
        avg_latent: torch.Tensor
            The average latent tensor
        max_layer: int
            The maximum layer value
        threshold: float
            The threshold value
        beta: float
            The beta value
        """
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer("avg_latent", avg_latent)

    def update(self, last_avg: torch.Tensor):
        """
        Updates the truncation layer

        Parameters
        ----------
        last_avg: torch.Tensor
            The last average tensor
        """
        self.avg_latent.copy_(
            self.beta * self.avg_latent + (1.0 - self.beta) * last_avg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the truncation layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (
            (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        )
        return torch.where(do_trunc, interp, x)


class StyleMod(nn.Module):
    """
    This class implements the style modulation layer (https://arxiv.org/abs/1812.04948v3)
    Specialize w to styles y = (ys, yb) that control adaptive instance normalization (AdaIN) operations after each convolution
    layer of the synthesis network g. The AdaIN operation is defined as
    AdaIN(xi, y) = ys,i * (xi - mean(xi)) / std(xi) + yb,i
    where each feature map xi is normalized separately, and then scaled and biased using the corresponding scalar components from style y.
    Thus the dimensionality of y is twice the number of feature maps on that layer.
    """

    def __init__(
        self, dlatent_dim: int, channels: int, use_wscale: bool = True
    ) -> None:
        super().__init__()
        """
        Initializes the StyleMod class.
        
        Parameters
        ----------
        dlatent_dim: int
            The dimension of the dlatent vector
        channels: int
            The number of channels
        use_wscale: bool
            Whether to use the scale value
        """
        self.lin = EqualizedLinear(
            dlatent_dim,
            channels * 2,  # 2 for scale and bias
            gain=1.0,
            use_wscale=use_wscale,
        )

    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the style modulation layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor
        latent: torch.Tensor
            The latent tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        style = self.lin(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        return x * (style[:, 0] + 1.0) + style[:, 1]


class LayerEpilogue(nn.Module):
    """
    This class implements the layer epilogue which are the operations that are applied at the end of each layer (https://arxiv.org/pdf/1710.10196.pdf)
    """

    def __init__(
        self,
        channels: int = 512,
        dlatent_dim: int = 512,
        use_noise: bool = True,
        use_pixel_norm: bool = False,
        use_instance_norm: bool = True,
        use_styles: bool = True,
        use_wscale: bool = True,
    ) -> None:
        super().__init__()
        """
        Initializes the LayerEpilogue class

        Parameters
        ----------
        channels: int
            The number of channels
        dlatent_dim: int
            The dimension of the dlatent vector
        use_noise: bool
            Whether to use the noise layer
        use_pixel_norm: bool
            Whether to use the pixel norm layer
        use_instance_norm: bool
            Whether to use the instance norm layer
        use_styles: bool
            Whether to use the style modulation layer
        use_wscale: bool
            Whether to use the scale value
        """
        layers = []
        if use_noise:
            layers.append(("noise", NoiseLayer(channels)))
        layers.append(("activation", nn.LeakyReLU(0.2, inplace=True)))
        if use_pixel_norm:
            layers.append(("pixel_norm", PixelNormLayer()))
        if use_instance_norm:
            layers.append(("instance_norm", nn.InstanceNorm2d(channels)))

        self.top_epi = nn.Sequential(OrderedDict(layers))

        if use_styles:
            self.style_mod = StyleMod(
                dlatent_dim=dlatent_dim, channels=channels, use_wscale=use_wscale
            )
        else:
            self.style_mod = None

    def forward(self, x: torch.Tensor, dlatents_in_slice=None) -> torch.Tensor:
        """
        Forward pass for the layer epilogue

        Parameters
        ----------
        x: torch.Tensor
            The input tensor
        dlatents_in_slice: torch.Tensor
            The latent tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        x = self.top_epi(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        else:
            assert dlatents_in_slice is None
        return x


class StddevLayer(nn.Module):
    """
    This class implements the standard deviation layer

    """

    def __init__(
        self,
        group_size: int = 4,
        num_new_features: int = 1,
    ) -> None:
        super().__init__()
        """
        Initializes the StddevLayer class

        Parameters
        ----------
        group_size: int
            The group size
        num_new_features: int
            The number of new features
        """
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the standard deviation layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        b, c, h, w = x.shape
        group_size = min(self.group_size, b)
        y = x.reshape(
            [group_size, -1, self.num_new_features, c // self.num_new_features, h, w]
        )
        y = y - y.mean(dim=0, keepdim=True)
        y = (y**2).mean(dim=0, keepdim=False)
        y = (y + 1e-8) ** 0.5
        y = y.mean(dim=[2, 3, 4], keepdim=True).squeeze(3)
        y = (
            y.expand(group_size, -1, -1, h, w)
            .clone()
            .reshape(b, self.num_new_features, h, w)
        )
        z = torch.cat([x, y], dim=1)
        return z


class View(nn.Module):
    """
    This class implements the view layer
    """

    def __init__(self, *shape: int) -> None:
        super().__init__()
        """
        Initializes the View class

        Parameters
        ----------
        shape: int
            The shape of the input tensor
        """
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the view layer

        Parameters
        ----------
        x: torch.Tensor
            The input tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        return x.view(x.size(0), *self.shape)


class InputBlock(nn.Module):
    """
    The first block (4x4 "pixels") doesn't have an input.
    The result of the first convolution is just replaced by a (trained) constant.
    """

    def __init__(
        self,
        nf: int = 512,
        dlatent_dim: int = 512,
        gain: float = np.sqrt(2),
        const_input_layer: bool = True,
        use_noise: bool = True,
        use_pixel_norm: bool = False,
        use_instance_norm: bool = True,
        use_styles: bool = True,
        use_wscale: bool = True,
    ) -> None:
        super().__init__()
        """
        Initializes the InputBlock class

        Parameters
        ----------
        nf: int
            The number of filters
        dlatent_dim: int
            The dimension of the dlatent vector
        gain: float
            The gain value to use
        const_input_layer: bool
            Whether to use the constant input layer
        use_noise: bool
            Whether to use the noise layer
        use_pixel_norm: bool
            Whether to use the pixel norm layer
        use_instance_norm: bool
            Whether to use the instance norm layer
        use_styles: bool
            Whether to use the style modulation layer
        use_wscale: bool
            Whether to use the scale value
        """
        self.const_input_layer = const_input_layer
        self.nf = nf

        if self.const_input_layer:  # Learned constant input layer
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = EqualizedLinear(
                input_size=dlatent_dim,
                output_size=nf * 16,
                gain=gain / 4,
                use_wscale=use_wscale,
            )
            # tweak gain to match the official implementation of Progressing GAN

        self.epi1 = LayerEpilogue(
            channels=self.nf,
            dlatent_dim=dlatent_dim,
            use_noise=use_noise,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_styles=use_styles,
            use_wscale=use_wscale,
        )
        self.conv = EqualizedConv2d(
            input_channels=self.nf,
            output_channels=self.nf,
            kernel_size=3,
            gain=gain,
            use_wscale=use_wscale,
        )
        self.epi2 = LayerEpilogue(
            channels=self.nf,
            dlatent_dim=dlatent_dim,
            use_noise=use_noise,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_styles=use_styles,
            use_wscale=use_wscale,
        )

    def forward(self, dlatents_in_range: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the input block

        Parameters
        ----------
        dlatents_in_range: torch.Tensor
            The latent tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        batch_size = dlatents_in_range.size(0)

        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)

        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])

        return x


class GSynthesisBlock(nn.Module):
    """
    This class implements the synthesis block
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        dlatent_dim: int = 512,
        blur_filter=None,
        gain: float = np.sqrt(2),
        use_wscale: bool = True,
        use_noise: bool = True,
        use_pixel_norm: bool = False,
        use_instance_norm: bool = True,
        use_styles: bool = True,
    ) -> None:
        super().__init__()
        """
        Initializes the GSynthesisBlock class

        Parameters
        ----------
        in_channels: int
            The number of input channels
        num_channels: int
            The number of output channels
        dlatent_dim: int
            The dimension of the dlatent vector
        blur_filter: None
            Low-pass filter to apply when resampling activations. None = no filtering.
        gain: float
            The gain value to use
        use_wscale: bool
            Whether to use the scale value
        use_noise: bool
            Whether to use the noise layer
        use_pixel_norm: bool
            Whether to use the pixel norm layer
        use_instance_norm: bool
            Whether to use the instance norm layer
        use_styles: bool
            Whether to use the style modulation layer
        """
        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None

        self.conv0_up = EqualizedConv2d(
            input_channels=in_channels,
            output_channels=num_channels,
            kernel_size=3,
            gain=gain,
            use_wscale=use_wscale,
            intermediate=blur,
            upscale=True,
        )
        self.epi1 = LayerEpilogue(
            channels=num_channels,
            dlatent_dim=dlatent_dim,
            use_noise=use_noise,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_styles=use_styles,
            use_wscale=use_wscale,
        )
        self.conv1 = EqualizedConv2d(
            input_channels=num_channels,
            output_channels=num_channels,
            kernel_size=3,
            gain=gain,
            use_wscale=use_wscale,
        )
        self.epi2 = LayerEpilogue(
            channels=num_channels,
            dlatent_dim=dlatent_dim,
            use_noise=use_noise,
            use_pixel_norm=use_pixel_norm,
            use_instance_norm=use_instance_norm,
            use_styles=use_styles,
            use_wscale=use_wscale,
        )

    def forward(self, x: torch.Tensor, dlatents_in_range: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the synthesis block

        Parameters
        ----------
        x: torch.Tensor
            The input tensor
        dlatents_in_range: torch.Tensor
            The latent tensor

        Returns
        -------
        torch.Tensor
            The output tensor
        """
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


class DiscriminatorBlock(nn.Sequential):
    """
    This class implements the discriminator block
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: int,
        gain: float = np.sqrt(2),
        use_wscale: bool = True,
        blur_filter=None,
    ) -> None:
        super().__init__(
            OrderedDict(
                [
                    (
                        "conv0",
                        EqualizedConv2d(
                            in_channels,
                            in_channels,
                            kernel_size=3,
                            gain=gain,
                            use_wscale=use_wscale,
                        ),
                    ),
                    ("act0", nn.LeakyReLU(0.2, inplace=True)),
                    ("blur", BlurLayer(kernel=blur_filter)),
                    (
                        "conv1_down",
                        EqualizedConv2d(
                            in_channels,
                            num_channels,
                            kernel_size=3,
                            gain=gain,
                            use_wscale=use_wscale,
                            downscale=True,
                        ),
                    ),
                    ("act1", nn.LeakyReLU(0.2, inplace=True)),
                ]
            )
        )


class DiscriminatorTop(nn.Sequential):
    """
    This class implements the discriminator top
    """

    def __init__(
        self,
        mbstd_num_features: int = 1,
        mbstd_group_size: int = 4,
        in_channels: int = 512,
        resolution: int = 4,
        second_in_channels=None,
        intermediate_channels: int = 512,
        output_features: int = 1,
        gain: float = np.sqrt(2),
        last_gain: float = 1,
        use_wscale: bool = True,
    ) -> None:
        """
        Initializes the DiscriminatorTop class

        Parameters
        ----------
        mbstd_num_features: int
            The minibatch standard number of features
        mbstd_group_size: int
            The minibatch standard group size
        in_channels: int
            The input channels
        resolution: int
            The resolution value
        second_in_channels: None
            The second input channels
        intermediate_channels: int
            The intermediate channels
        output_features: int
            The output features
        gain: float
            The gain value to use
        last_gain: float
            The last gain value to use
        use_wscale: bool
            Whether to use the scale value
        """

        layers = []
        if mbstd_group_size > 1:
            layers.append(
                ("stddev_layer", StddevLayer(mbstd_group_size, mbstd_num_features))
            )

        if second_in_channels is None:
            second_in_channels = in_channels

        layers.append(
            (
                "conv",
                EqualizedConv2d(
                    in_channels + mbstd_num_features,
                    second_in_channels,
                    kernel_size=3,
                    gain=gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        layers.append(("act0", nn.LeakyReLU(0.2, inplace=True)))
        layers.append(("view", View(-1)))
        layers.append(
            (
                "dense0",
                EqualizedLinear(
                    second_in_channels * resolution * resolution,
                    intermediate_channels,
                    gain=gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        layers.append(("act1", nn.LeakyReLU(0.2, inplace=True)))
        layers.append(
            (
                "dense1",
                EqualizedLinear(
                    intermediate_channels,
                    output_features,
                    gain=last_gain,
                    use_wscale=use_wscale,
                ),
            )
        )
        super().__init__(OrderedDict(layers))


if __name__ == "__main__":
    pass
