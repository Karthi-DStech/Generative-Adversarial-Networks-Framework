import os
import sys

from options.base_option import BaseOptions

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainOptions(BaseOptions):
    """Train options"""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initialize train options"""
        BaseOptions.initialize(self)

        self._parser.add_argument(
            "--model_name",
            type=str,
            required=False,
            default="ACBlurGAN",
            help="model name",
            choices=[
                "ACGAN",
                "VanillaGAN",
                "ACVanillaGAN",
                "WGANGP",
                "ACWGANGP",
                "WGANWC",
                "MorphGAN",
                "WCGANGP",
                "BlurGAN",
                "ACBlurGAN",
                "STYLEGAN",
                "ACCBlurGAN",
            ],
        )
        self._parser.add_argument(
            "--init_type",
            type=str,
            required=False,
            default="xavier_normal",
            help="initialization type",
            choices=["normal", "xavier_normal", "kaiming_normal"],
        )

        self._parser.add_argument(
            "--optimizer",
            type=str,
            required=False,
            default="rmsprop",
            help="optimizer",
            choices=["adam", "rmsprop"],
        )
        self._parser.add_argument(
            "--lr_scheduler",
            type=str,
            required=False,
            default="none",
            help="learning rate scheduler",
            choices=["none", "step"],
        )
        self._parser.add_argument(
            "--lr_decay_start",
            type=int,
            required=False,
            default=10,
            help="learning rate decay start epoch",
        )
        self._parser.add_argument(
            "--lr_decay_step",
            type=int,
            required=False,
            default=2,
            help="learning rate decay step",
        )
        self._parser.add_argument(
            "--lr_decay_gamma",
            type=float,
            required=False,
            default=0.3,
            help="learning rate decay gamma",
        )
        self._parser.add_argument(
            "--g_lr",
            type=float,
            required=False,
            default=0.0003,
            help="generator learning rate",
        )
        self._parser.add_argument(
            "--g_adam_beta1",
            type=float,
            required=False,
            default=0.5,
            help="generator adam beta1",
        )
        self._parser.add_argument(
            "--g_adam_beta2",
            type=float,
            required=False,
            default=0.999,
            help="generator adam beta2",
        )

        self._parser.add_argument(
            "--d_lr",
            type=float,
            required=False,
            default=0.0003,
            help="discriminator learning rate",
        )
        self._parser.add_argument(
            "--d_adam_beta1",
            type=float,
            required=False,
            default=0.5,
            help="discriminator adam beta1",
        )
        self._parser.add_argument(
            "--d_adam_beta2",
            type=float,
            required=False,
            default=0.999,
            help="discriminator adam beta2",
        )
        self._parser.add_argument(
            "--latent_dim",
            type=int,
            required=False,
            default=112,
            help="latent dimension",
        )
        self._parser.add_argument(
            "--embedding_dim",
            type=int,
            required=False,
            default=112,
            help="embedding dimension",
        )
        self._parser.add_argument(
            "--n_classes",
            type=int,
            required=False,
            default=5,
            help="number of classes for conditional GANs",
        )

        # VanillaGAN and ACVanillaGAN parameters
        self._parser.add_argument(
            "--vanilla_g_neurons",
            type=int,
            required=False,
            default=256,
            help="VanillaGAN generator neurons",
        )
        self._parser.add_argument(
            "--vanilla_d_neurons",
            type=int,
            required=False,
            default=1024,
            help="VanillaGAN discriminator neurons",
        )
        self._parser.add_argument(
            "--d_lambda_adv",
            type=float,
            required=False,
            default=1,
            help="discriminator adversarial loss weight",
        )
        self._parser.add_argument(
            "--d_lambda_cls",
            type=float,
            required=False,
            default=1,
            help="discriminator classification loss weight",
        )
        self._parser.add_argument(
            "--g_lambda_adv",
            type=float,
            required=False,
            default=1,
            help="generator adversarial loss weight",
        )
        self._parser.add_argument(
            "--g_lambda_cls",
            type=float,
            required=False,
            default=1,
            help="generator classification loss weight",
        )

        # WGAN parameters
        self._parser.add_argument(
            "--d_lambda_w",
            type=float,
            required=False,
            default=1,
            help="discriminator wasserstein loss weight",
        )
        self._parser.add_argument(
            "--g_lambda_w",
            type=float,
            required=False,
            default=1,
            help="generator wasserstein loss weight",
        )
        self._parser.add_argument(
            "--d_lambda_gp",
            type=float,
            required=False,
            default=10,
            help="discriminator gradient penalty loss weight",
        )
        self._parser.add_argument(
            "--clip_value",
            type=float,
            required=False,
            default=0.01,
            help="weight clipping value",
        )

        # StyleGAN parameters
        self._parser.add_argument(
            "--batch_list",
            type=list,
            required=False,
            default=[128, 128, 128, 64, 32, 16, 8, 4, 2],
            help="list of batch_sizes for every resolution",
        )
        self._parser.add_argument(
            "--epoch_list",
            type=list,
            required=False,
            default=[4, 4, 4, 4, 8, 16, 32, 64, 64],
            help="list of epochs for every resolution",
        )
        self._parser.add_argument(
            "--fade_in_perc",
            type=list,
            required=False,
            default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            help="list of fade-in percentages for every resolution",
        )
        self._parser.add_argument(
            "--structure",
            type=str,
            required=False,
            default="linear",
            help="Structure of the generator, 'fixed' = no progressive growing, 'linear' = human-readable",
            choices=["fixed", "linear"],
        )
        self._parser.add_argument(
            "--mapping_layers",
            type=int,
            required=False,
            default=8,
            help="Number of mapping layers",
        )
        self._parser.add_argument(
            "--blur_filter",
            type=self._list_or_none,
            required=False,
            default=[1, 2, 1],
            help="Blur filter",
        )
        self._parser.add_argument(
            "--truncation_psi",
            type=self._float_or_none,
            required=False,
            default=0.7,
            help="Style strength multiplier for the truncation trick, None = disable",
        )
        self._parser.add_argument(
            "--truncation_cutoff",
            type=int,
            required=False,
            default=8,
            help="Number of layers for which to apply the truncation trick",
        )
        self._parser.add_argument(
            "--style_mixing_prob",
            type=self._float_or_none,
            required=False,
            default=0.9,
            help="Probability of mixing styles during training, None = disable",
        )
        self._parser.add_argument(
            "--use_ema",
            type=bool,
            required=False,
            default=True,
            help="Whether to use exponential moving average",
        )
        self._parser.add_argument(
            "--ema_decay",
            type=float,
            required=False,
            default=0.999,
            help="Exponential moving average decay",
        )
        self._parser.add_argument(
            "--use_wscale",
            type=bool,
            required=False,
            default=True,
            help="Whether to use weight scaling",
        )
        self._parser.add_argument(
            "--start_depth",
            type=int,
            required=False,
            default=0,
            help="Starting depth for progressive growing",
        )

        # MorphGAN parameters
        self._parser.add_argument(
            "--morph_type",
            type=str,
            required=False,
            default="closing",
            help="Morphological operation",
            choices=["closing", "opening"],
        )
        self._parser.add_argument(
            "--morph_kernel_size",
            type=int,
            required=False,
            default=5,
            help="Morphological kernel size",
        )
        self._parser.add_argument(
            "--g_lambda_morph",
            type=float,
            required=False,
            default=100,
            help="Morphological loss weight",
        )

        # BlurGAN parameters
        self._parser.add_argument(
            "--blur_kernel_size",
            type=int,
            required=False,
            default=11,
            help="Blur kernel size",
        )
        self._parser.add_argument(
            "--g_lambda_blur",
            type=float,
            required=False,
            default=600,
            help="Blur loss weight",
        )

        ######## New Parameters to be added above this line ########
        self._is_train = True
