import argparse
from typing import Union

import torch

from data.datasets import BaseDataset
from model.models import BaseModel


def make_model(model_name: str, *args, **kwargs) -> Union[BaseModel, BaseModel]:
    """
    Creates a model from the given model name

    Parameters
    ----------
    model_name: str
        The name of the model to create
    *args: list
        The arguments to pass to the model constructor
    **kwargs: dict
        The keyword arguments to pass to the model constructor

    Returns
    -------
    model: BaseModel
        The created model
    """
    model = None

    if model_name.lower() == "acgan":
        from model.acgan import ACGAN

        model = ACGAN(*args, **kwargs)

    elif model_name.lower() == "vanillagan":
        from model.vanillagan import VanillaGAN

        model = VanillaGAN(*args, **kwargs)

    elif model_name.lower() == "acvanillagan":
        from model.acvanilla import ACVanillaGAN

        model = ACVanillaGAN(*args, **kwargs)

    elif model_name.lower() == "wgangp":
        from model.wgan import WGAN_GP

        model = WGAN_GP(*args, **kwargs)

    elif model_name.lower() == "wganwc":
        from model.wgan import WGAN_WC

        model = WGAN_WC(*args, **kwargs)

    elif model_name.lower() == "acwgangp":
        from model.wgan import ACWGAN_GP

        model = ACWGAN_GP(*args, **kwargs)

    elif model_name.lower() == "stylegan":
        from model.stylegan import StyleGAN

        model = StyleGAN(*args, **kwargs)

    elif model_name.lower() == "morphgan":
        from model.morphgan import MorphGAN

        model = MorphGAN(*args, **kwargs)

    elif model_name.lower() == "wcgangp":
        from model.wgan import WCGAN_GP

        model = WCGAN_GP(*args, **kwargs)

    elif model_name.lower() == "blurgan":
        from model.blurgan import BlurGAN

        model = BlurGAN(*args, **kwargs)

    elif model_name.lower() == "acblurgan":
        from model.blurgan import ACBlurGAN

        model = ACBlurGAN(*args, **kwargs)

    elif model_name.lower() == "accblurgan":
        from model.blurgan import ACCBlurGAN

        model = ACCBlurGAN(*args, **kwargs)


    else:
        raise ValueError(f"Invalid model name: {model_name}")
    print(f"Model {model_name} was created")
    return model


def make_network(network_name: str, *args, **kwargs) -> torch.nn.Module:
    """
    Creates a network from the given network name

    Parameters
    ----------
    network_name: str
        The name of the network to create
    *args: list
        The arguments to pass to the network constructor
    **kwargs: dict
        The keyword arguments to pass to the network constructor

    Returns
    -------
    network: torch.nn.Module
        The created network
    """
    network = None
    if network_name.lower() == "acgangenerator":
        from model.generators import ACGANGenerator

        network = ACGANGenerator(*args, **kwargs)

    elif network_name.lower() == "acgandiscriminator":
        from model.discriminators import ACGANDiscriminator

        network = ACGANDiscriminator(*args, **kwargs)

    elif network_name.lower() == "vanillagenerator":
        from model.generators import VanillaGenerator

        network = VanillaGenerator(*args, **kwargs)

    elif network_name.lower() == "vanilladiscriminator":
        from model.discriminators import VanillaDiscriminator

        network = VanillaDiscriminator(*args, **kwargs)

    elif network_name.lower() == "acvanillagenerator":
        from model.generators import ACVanillaGenerator

        network = ACVanillaGenerator(*args, **kwargs)

    elif network_name.lower() == "acvanilladiscriminator":
        from model.discriminators import ACVanillaDiscriminator

        network = ACVanillaDiscriminator(*args, **kwargs)
    elif network_name.lower() == "wgancritic":
        from model.discriminators import WGANCritic

        network = WGANCritic(*args, **kwargs)
    elif network_name.lower() == "acwgancritic":
        from model.discriminators import ACWGANCritic

        network = ACWGANCritic(*args, **kwargs)
    elif network_name.lower() == "stylediscriminator":
        from model.discriminators import StyleDiscriminator

        network = StyleDiscriminator(*args, **kwargs)
    elif network_name.lower() == "stylegenerator":
        from model.generators import StyleGenerator

        network = StyleGenerator(*args, **kwargs)
    elif network_name.lower() == "convgangenerator":
        from model.generators import ConvGANGenerator

        network = ConvGANGenerator(*args, **kwargs)
    elif network_name.lower() == "convgancritic":
        from model.discriminators import ConvGANCritic

        network = ConvGANCritic(*args, **kwargs)
    elif network_name.lower() == "acconvgangenerator":
        from model.generators import ACConvGANGenerator

        network = ACConvGANGenerator(*args, **kwargs)
    elif network_name.lower() == "acwcgancritic":
        from model.discriminators import ACWCGANCritic

        network = ACWCGANCritic(*args, **kwargs)
    else:
        raise ValueError(f"Invalid network name: {network_name}")
    print(f"Network {network_name} was created")
    return network


def make_dataset(dataset_name: str, opt: argparse.Namespace, *args, **kwargs):
    """
    Creates a dataset from the given dataset name

    Parameters
    ----------
    dataset_name: str
        The name of the dataset to create
    opt: argparse.Namespace
        The training options
    *args: list
        The arguments to pass to the dataset constructor
    **kwargs: dict
        The keyword arguments to pass to the dataset constructor

    Returns
    -------
    dataset: BaseDataset
        The created dataset
    """
    dataset = None
    if dataset_name.lower() == "mnist":
        from data.mnist import MNISTDataset, MNISTTest

        train_dataset = MNISTDataset(opt, *args, **kwargs)
        test_dataset = MNISTTest(opt, *args, **kwargs)
        dataset = (train_dataset, test_dataset)

    elif dataset_name.lower() == "biological":
        from data.topographies import BiologicalObservation

        train_dataset = BiologicalObservation(opt, *args, **kwargs)
        dataset = (train_dataset,)

    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")

    for d in dataset:
        make_dataloader(
            d,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )
        d.print_dataloader_info()

    print(f"Dataset {dataset_name} was created")
    return dataset


def make_dataloader(dataset: BaseDataset, *args, **kwargs) -> None:
    """
    Creates a dataloader from the given dataset

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        The dataset to create the dataloader from
    *args: list
        The arguments to pass to the dataloader constructor
    **kwargs: dict
        The keyword arguments to pass to the dataloader constructor

    Returns
    -------
    None
    """
    dataset.dataloader = torch.utils.data.DataLoader(dataset, *args, **kwargs)
