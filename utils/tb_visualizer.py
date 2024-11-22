import argparse
import os
import time
from datetime import datetime

from tensorboardX import SummaryWriter
from torchvision.utils import make_grid, save_image


class Visualizer:
    """
    A class for visualizing the training process
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initialize the visualizer

        Parameters
        ----------
        opt: argparse.Namespace
            The training options

        Returns
        -------
        None
        """
        self.opt = opt
        self.log_dir = os.path.join(self.opt.log_dir, self.opt.experiment_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.opt_path = os.path.join(self.log_dir, "opt.json")
        self.log_path = os.path.join(self.log_dir, "log.txt")
        self.img_path = os.path.join(self.log_dir, "images")
        os.makedirs(self.img_path, exist_ok=True)

        with open(self.opt_path, "w") as f:
            for key, value in vars(opt).items():
                f.write(f"{key}: {value}\n")

        message = f'{"="*20} Experiment Log ({datetime.now()}) {"="*20}\n'
        print(f"{message}")
        with open(self.log_path, "w") as f:
            f.write(message + "\n")

    def log_performance(
        self,
        performance: dict,
        epoch: int,
        step: int,
        total_steps: int,
        is_train: bool = True,
        print_freq: int = 10,
    ) -> None:
        """
        Log the performance of the model

        Parameters
        ----------
        performance: dict
            The performance of the model
        epoch: int
            The current epoch
        step: int
            The current step
        total_steps: int
            The total number of steps
        is_train: bool
            Whether the model is in training mode
        print_freq: int
            The frequency of printing the performance

        Returns
        -------
        None
        """
        losses = performance["losses"]
        lr = performance["lr"]
        for label, scalars in losses.items():
            sum_name = "{}/{}".format("Train" if is_train else "Test", label)
            self.writer.add_scalar(sum_name, scalars, total_steps)
        if is_train:
            for label, scalars in lr.items():
                sum_name = "LR/{}".format(label)
                self.writer.add_scalar(sum_name, scalars, total_steps)
        self._print_performance(losses, lr, epoch, step, is_train, print_freq)

    def _print_performance(
        self,
        losses: dict,
        lr: dict,
        epoch: int,
        step: int,
        is_train: bool = True,
        print_freq: int = 10,
    ) -> None:
        """
        Print the performance of the model

        Parameters
        ----------
        losses: dict
            The losses of the model
        lr: dict
            The learning rates of the model
        epoch: int
            The current epoch
        step: int
            The current step
        is_train: bool
            Whether the model is in training mode
        print_freq: int
            The frequency of printing the performance

        Returns
        -------
        None
        """
        if step % print_freq == 0:
            if is_train:
                message = "Train "
            else:
                message = "Test "
            message += f"[Epoch {epoch}/{self.opt.n_epochs-1}][{step}] \n"
            for label, scalars in losses.items():
                message += f"{label}: {scalars:.4f} "
            for label, scalars in lr.items():
                message += f"{label}: {scalars} "
            print(message)
            with open(self.log_path, "a") as f:
                f.write(message + "\n")
        else:
            pass

    def log_image(
        self, vis_data: tuple, total_steps: int, is_train: bool = True
    ) -> None:
        """
        Log the images of the model

        Parameters
        ----------
        vis_data: tuple
            The images and labels to visualize
        total_steps: int
            The total number of steps
        is_train: bool
            Whether the model is in training mode

        Returns
        -------
        None
        """
        images = vis_data[0]
        labels = vis_data[1]
        image_grid = make_grid(
            images, nrow=int(images.shape[0] ** 0.5), normalize=False, scale_each=True
        )
        sum_name = "{}/{}".format("Train" if is_train else "Test", "images")
        self.writer.add_image(sum_name, image_grid, total_steps)

        self.image_folder = os.path.join(self.img_path, "train" if is_train else "test")
        os.makedirs(self.image_folder, exist_ok=True)
        for i in range(images.shape[0]):
            save_image(
                images[i],
                os.path.join(self.image_folder, f"{labels[i]}_{total_steps}_.png"),
            )

    def log_time(
        self,
        end: float,
        start: float,
        epoch: int,
        is_train: bool = True,
        training_end: bool = False,
    ) -> None:
        """
        Log the time of the model

        Parameters
        ----------
        end: float
            The end time
        start: float
            The start time
        epoch: int
            The current epoch
        is_train: bool
            Whether the model is in training mode
        training_end: bool
            Whether the training is finished

        Returns
        -------
        None
        """
        duration = time.strftime("%H:%M:%S", time.gmtime(end - start))
        if not training_end:
            message = f'{"="*50} Epoch {epoch} {"="*50}\n'
            message += f'{"Train" if is_train else "Test"} time: {duration}\n'
        else:
            message = f'{"="*45} {"Training Finished" if is_train else "Testing Finished"} {"="*45}\n'
            message += f"Total time: {duration}\n"
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def close(self) -> None:
        """
        Close the visualizer
        """
        self.writer.close()

    def log_progressive(
        self,
        current_depth: int,
        current_res: int,
    ) -> None:
        """
        Log the progressive training process

        Parameters
        ----------
        current_depth: int
            The current depth of the model
        current_res: int
            The current resolution of the model

        Returns
        -------
        None
        """
        message = f"Currently working on depth {current_depth} \n"
        message += f"Current resolution: {current_res} X {current_res} \n"
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def log_progressive_time(
        self,
        end: float,
        start: float,
        current_depth: int = 0,
        current_res: int = 4,
        training_end: bool = False,
    ) -> None:
        """
        Log the progressive training time

        Parameters
        ----------
        end: float
            The end time
        start: float
            The start time
        current_depth: int
            The current depth of the model
        current_res: int
            The current resolution of the model

        Returns
        -------
        None
        """
        duration = time.strftime("%H:%M:%S", time.gmtime(end - start))
        if not training_end:
            message = f'{"="*20} Depth {current_depth} Time {"="*20}\n'
            message += f"Time: {duration}\n"
            message += f"Current resolution: {current_res} X {current_res} \n"
        else:
            message = f'{"="*20} Training Finished {"="*20}\n'
            message += f"Total time: {duration}\n"
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")
