from torchvision import transforms
from typing import List, Tuple, Union


def get_transform(img_size: int, mean: float, std: float) -> transforms.Compose:
    """
    Gets the transforms for the dataset

    Parameters
    ----------
    img_size: int
        The size of the image
    mean: float
        The mean of the dataset
    std: float
        The standard deviation of the dataset

    Returns
    -------
    transforms.Compose
        The transforms for the dataset
    """
    transform = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )
    return transform


def get_fade_point(fade_in_perc: float, total_epochs: int, total_batch: int) -> int:
    """
    Gets the fade point

    Parameters
    ----------
    fade_in_perc: float
        The percentage of the fade in
    total_epochs: int
        The total number of epochs
    total_batch: int
        The total number of batches

    Returns
    -------
    int
        The fade point
    """
    fade_point = int(total_batch * total_epochs * fade_in_perc)
    return fade_point


def set_progressive_options(
    depth: int,
    batch_list: Union[List[int], int],
    epoch_list: Union[List[int], int],
    fade_in_perc: Union[List[float], float],
) -> Tuple[List[int], List[int], List[float]]:
    """
    Check and set the progressive options

    Parameters
    ----------
    depth: int
        The depth of the model
    batch_list: Union[List[int], int]
        The list of batch sizes
    epoch_list: Union[List[int], int]
        The list of epochs
    fade_in_perc: Union[List[float], float]
        The list of fade-in percentages

    Returns
    -------
    batch_list: List[int]
        The list of batch sizes
    epoch_list: List[int]
        The list of epochs
    fade_in_perc: List[float]
        The list of fade-in percentages
    """

    def pad_list(lst: List, target_length: int) -> List:
        return lst + [lst[-1]] * max(0, target_length - len(lst))

    batch_list = pad_list(
        [batch_list] * depth if isinstance(batch_list, int) else batch_list[:depth],
        depth,
    )
    epoch_list = pad_list(
        [epoch_list] * depth if isinstance(epoch_list, int) else epoch_list[:depth],
        depth,
    )
    fade_in_perc = pad_list([fade_in_perc] * depth if isinstance(fade_in_perc, float) else fade_in_perc[:depth], depth)  # type: ignore

    return batch_list, epoch_list, fade_in_perc
