import argparse
import ast
import os
import sys
from typing import Dict, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseOptions:
    """
    This class defines options used during all types of experiments.
    It also implements several helper functions such as parsing, printing, and saving the options.
    """

    def __init__(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._parser = argparse.ArgumentParser()
        self._initialized = False
        self._float_or_none = self.float_or_none
        self._list_or_none = self.list_or_none

    def initialize(self) -> None:
        """
        Initializes the BaseOptions class

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._parser.add_argument(
            "--experiment_name",
            type=str,
            required=False,
            default="train_pipeline_ACBLURGAN",
            help="experiment name",
        )
        self._parser.add_argument(
            "--images_folder",
            type=str,
            required=False,
            default="../Datasets/Topographies/raw/FiguresStacked 8X8_4X4_2X2 Embossed",
            help="path to the images",
        )
        self._parser.add_argument(
            "--label_path",
            type=str,
            required=False,
            default="../Datasets/biology_data/TopoChip/AeruginosaWithClass.csv",
            help="path to the label csv file",
        )
        self._parser.add_argument(
            "--log_dir", type=str, required=False, default="./logs", help="path to log"
        )
        self._parser.add_argument(
            "--dataset_name",
            type=str,
            required=False,
            default="biological",
            help="dataset name",
            choices=["mnist", "biological"],
        )
        self._parser.add_argument(
            "--dataset_params",
            type=lambda x: ast.literal_eval(x),
            required=False,
            default={"mean": 0.5, "std": 0.5},
            help="dataset parameters",
        )
        self._parser.add_argument(
            "--n_epochs", type=int, required=False, default=200, help="number of epochs"
        )
        self._parser.add_argument(
            "--img_type", type=str, required=False, default="png", help="image type"
        )
        self._parser.add_argument(
            "--img_size", type=int, required=False, default=224, help="image size"
        )
        self._parser.add_argument(
            "--in_channels",
            type=int,
            required=False,
            default=1,
            help="number of input channels",
        )
        self._parser.add_argument(
            "--out_channels",
            type=int,
            required=False,
            default=1,
            help="number of output channels",
        )
        self._parser.add_argument(
            "--batch_size", type=int, required=False, default=32, help="batch size"
        )
        self._parser.add_argument(
            "--num_workers",
            type=int,
            required=False,
            default=4,
            help="number of workers",
        )
        self._parser.add_argument(
            "--train_dis_freq",
            type=int,
            required=False,
            default=1,
            help="number of discriminator iterations per generator iteration",
        )
        self._parser.add_argument(
            "--save_image_frequency",
            type=int,
            required=False,
            default=100,
            help="save image frequency",
        )
        self._parser.add_argument(
            "--print_freq", type=int, required=False, default=10, help="print frequency"
        )
        self._parser.add_argument(
            "--model_save_frequency",
            type=int,
            required=False,
            default=100,
            help="model save frequency",
        )
        self._parser.add_argument(
            "--n_vis_samples",
            type=int,
            required=False,
            default=32,
            help="number of samples to visualize",
        )
        self._parser.add_argument(
            "--seed", type=int, required=False, default=1221, help="random seed"
        )

        self._initialized = True
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
        self._opt.is_train = self._is_train

        args = vars(self._opt)
        # self._print(args)

        return self._opt

    def _print(self, args: Dict) -> None:
        """
        Prints the arguments passed to the script

        Parameters
        ----------
        args: dict
            The arguments to print

        Returns
        -------
        None
        """
        print("------------ Options -------------")
        for k, v in args.items():
            print(f"{str(k)}: {str(v)}")
        print("-------------- End ---------------")

    def float_or_none(self, value: str) -> Union[float, None]:
        """
        Converts a string to float or None

        Parameters
        ----------
        value: str
            The value to convert

        Returns
        -------
        float
            The converted value
        """
        if value.lower() == "none":
            return None
        try:
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid float or 'none' value: {}".format(value)
            )

    def list_or_none(self, value: str) -> Union[list, None]:
        """
        Converts a string to list or None

        Parameters
        ----------
        value: str
            The value to convert

        Returns
        -------
        list
            The converted value
        """
        if value.lower() == "none":
            return None
        try:
            return ast.literal_eval(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid list or 'none' value: {}".format(value)
            )
