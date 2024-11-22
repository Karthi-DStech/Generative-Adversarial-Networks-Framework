import argparse
import ast
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from options.base_option import BaseOptions


class PredictOptions(BaseOptions):
    """Predict options"""

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """Initialize predict options"""
        BaseOptions.initialize(self)

        self._parser.add_argument(
            "--is_conditional",
            type=bool,
            required=False,
            default=True,
            help="whether the model is conditional",
        )

        self._parser.add_argument(
            "--label",
            type=int,
            required=False,
            default=4,
            help="label to generate",
        )

        self._parser.add_argument(
            "--model_path",
            type=str,
            required=False,
            default="/Users/rezachi/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Biomaterial/GAN results/logs/experiment30_ACBlurGAN_very_high_weight/net_ACVanillaGenerator_199.pth",
            help="path to the generator model",
        )
        self._parser.add_argument(
            "--num_images",
            type=int,
            required=False,
            default=12,
            help="number of images to generate",
        )

        ######## New Parameters to be added above this line ########
        self._is_train = False

    def parse(self) -> argparse.Namespace:
        """
        Parses the arguments passed to the script

        Parameters
        ----------
        None

        Returns
        -------
        opt: argparse.Namespace
            The parsed arguments
        """
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        self._load_opt_json()
        self._opt.is_train = self._is_train

        args = vars(self._opt)
        # self._print(args)

        return self._opt

    def _load_opt_json(self) -> None:
        """Load options from opt_path and add them as arguments"""
        model_name = self._opt.model_path.split("/")[-1]
        training_opt_path = self._opt.model_path.replace(model_name, "opt.json")
        # check if the file exists
        if not os.path.exists(training_opt_path):
            raise FileNotFoundError(
                f"training options file not found. Expected path: {training_opt_path}"
            )
        with open(training_opt_path, "r") as f:
            opt_data = f.read()
            lines = opt_data.strip().split("\n")
            for line in lines:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key == "is_train":
                    continue
                try:
                    value = eval(value)
                except (NameError, SyntaxError):
                    pass
                try:
                    self._parser.add_argument(
                        f"--{key}", type=type(value), default=value
                    )
                except argparse.ArgumentError:
                    continue
        self._opt = self._parser.parse_args()
