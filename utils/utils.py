import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the seed for the experiment

    Parameters
    ----------
    seed: int
        The seed to use for the experiment
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def update_average(
    target_model: torch.nn.Module, source_model: torch.nn.Module, beta: float
) -> None:
    """
    Update the exponential moving average of the target model using the source model

    Parameters
    ----------
    target_model: torch.nn.Module
        The model to update
    source_model: torch.nn.Module
        The model to use for the update
    beta: float
        The update coefficient
    """

    def toggle_grad(model: torch.nn.Module, requires_grad: bool) -> None:
        """
        Toggle gradient calculation

        Parameters
        ----------
        model: torch.nn.Module
            The model to toggle gradient calculation for
        requires_grad: bool
            Whether to calculate gradients
        """
        for p in model.parameters():
            p.requires_grad_(requires_grad)

    # turn off gradient calculation
    toggle_grad(target_model, False)
    toggle_grad(source_model, False)

    param_dict_src = dict(source_model.named_parameters())

    for p_name, target_p in target_model.named_parameters():
        source_p = param_dict_src[p_name]
        assert source_p is not target_p
        target_p.copy_(beta * target_p + (1.0 - beta) * source_p)

    # turn back on gradient calculation
    toggle_grad(target_model, True)
    toggle_grad(source_model, True)


import os


def delete_files_in_directory(directory: str) -> None:
    """
    Deletes all files in a directory

    Parameters
    ----------
    directory: str
        The directory to delete the files from
    """
    # Iterate over all the files and subdirectories in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # Check if it's a file
            if os.path.isfile(file_path):
                # Delete the file
                os.remove(file_path)
            # If it's a directory, recursively delete its content
            elif os.path.isdir(file_path):
                delete_files_in_directory(file_path)
                # After deleting all files in the subdirectory, remove the subdirectory itself
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {str(e)}")


def delete_directory(directory: str) -> None:
    """
    Deletes a directory

    Parameters
    ----------
    directory: str
        The directory to delete
    """
    try:
        # Remove the directory itself
        os.rmdir(directory)
        print(f"Directory '{directory}' and its contents deleted successfully.")
    except Exception as e:
        print(f"Failed to delete directory '{directory}'. Reason: {str(e)}")
