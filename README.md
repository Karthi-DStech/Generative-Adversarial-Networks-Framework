# GAN Framework for Image Generation - (Biomaterial Discovery) 

**Directory Structure**
```
Directory structure:
└── karthi-dstech-generative-adversarial-networks-framework/
    ├── README.md
    ├── call_methods.py
    ├── evaluate.py
    ├── predict.py
    ├── requirements.txt
    ├── train.py
    ├── data/
    │   ├── datasets.py
    │   ├── mnist.py
    │   └── topographies.py
    ├── launch/
    │   ├── predict.sh
    │   └── train.sh
    ├── model/
    │   ├── acgan.py
    │   ├── acvanilla.py
    │   ├── blurgan.py
    │   ├── discriminators.py
    │   ├── generators.py
    │   ├── models.py
    │   ├── networks.py
    │   ├── stylegan.py
    │   ├── vanillagan.py
    │   └── wgan.py
    ├── options/
    │   ├── base_option.py
    │   ├── evaluate_option.py
    │   ├── predict_option.py
    │   └── train_option.py
    └── utils/
        ├── custom_layers.py
        ├── images_utils.py
        ├── losses.py
        ├── tb_visualizer.py
        ├── utils.py
        └── weight_init.py
```

## Overview

This repository contains Generative Adversarial Network (GAN) Framework specifically designed to address the challenges of scalability, maintainability, reproducibility, and effective bug tracking in AI development. This framework is built to handle complex and extensive datasets with performance optimization, making it suitable for diverse real-world applications. It features a modular architecture that ensures clean, well-organized code for easy updates and integration, adhering to industry best practices.

The implementation of GANs including **VanillaGAN**, **ACVanillaGAN**, **WGAN**, **ACWGAN**, **WGAN-GP**, **ACWGAN-GP**, **WGAN-WC**, **WCGAN-GP**, **BlurGAN**, **ACBlurGAN**, **STYLEGAN**, and **ACCBlurGAN**. 


Here are detailed descriptions of multiple GAN architectures contained in this project: 

1. **<span style="color:red;">ACGAN</span>**: Auxiliary Classifier GAN, which combines a standard GAN using a CNN backbone with an auxiliary classifier for conditional image generation.

2. **<span style="color:red;">VanillaGAN</span>**: A basic GAN implementation with an MLP as the backbone for both the generator and discriminator.

3. **<span style="color:red;">ACVanillaGAN</span>**: Combines VanillaGAN with an auxiliary classifier for conditional generation.

4. **<span style="color:red;">WGANGP</span>**: A Wasserstein GAN that uses an MLP backbone with Gradient Penalty, enhancing training stability by leveraging the Wasserstein distance and gradient penalty.

5. **<span style="color:red;">ACWGANGP</span>**: Auxiliary Classifier WGAN with Gradient Penalty, combines WGANGP with conditional generation, also using MLP as backbone.

6. **<span style="color:red;">WGANWC</span>**: Wasserstein GAN with Weight Clipping, an alternative to WGANGP using weight clipping for Lipschitz constraint, also using MLP as its backbone.


7. **<span style="color:red;">MorphGAN</span>**(new): a GAN variant based on WGANGP. It incorporates morphological operations into the GAN framework, specifically using closing or opening transformations on the generated images. The model includes an additional morphological loss term alongside the Wasserstein loss, aiming to enhance the structural properties of the generated images.

8. **<span style="color:red;">WCGANGP</span>**: Wasserstein GAN with Gradient Penalty, a variant of WGANGP that uses CNN as its backbone.


9. **<span style="color:red;">BlurGAN</span>**(new):  A GAN variant based on WGANGP that incorporates blurring and thresholding techniques. It applies Gaussian blur to generated images and then thresholds them, adding a blur loss term to the generator’s objective alongside the Wasserstein loss, aiming to enhance the blurring effects in the generated images.

10. **<span style="color:red;">ACBlurGAN</span>**(new): Extends BlurGAN by adding an auxiliary classifier, enabling conditional image generation. It combines the blurring and thresholding approach with class-conditional generation, using a cross-entropy loss for classification in addition to the Wasserstein and blur losses.

11. **<span style="color:red;">StyleGAN</span>**: StyleGAN is an advanced GAN architecture known for high-quality image generation. It features a mapping network that transforms the input latent code, a synthesis network that progressively generates the image, and a style-based generator that allows fine-grained control over image attributes. The model incorporates techniques like adaptive instance normalization, noise injection, and progressive growing, enabling the generation of highly detailed and diverse images.

12. **<span style="color:red;">ACCBlurGAN</span>**(new): Further enhances ACBlurGAN by using convolutional layers as the backbone for both the generator and discriminator.


## Enviroment Dependencies
To run this project, you'll need the following dependencies, the [requirements.txt](./requirements.txt) file is provided for your convenience:

    Python 3.7+
    kornia==0.7.1
    matplotlib==3.7.3
    pandas==2.0.3
    tensorboardX==2.6.2.2
    torch
    torchvision
    tqdm==4.66.2
    scipy==1.10.1
*Note*: For GPU acceleration, ensure that CUDA is installed and compatible with your version of PyTorch, and make sure to use the GPU version of PyTorch. Additionally, version requirements may vary, so it's recommended to check the import statements in the codebase for the most up-to-date dependency information.

## Quick Start
1. **Clone the repository**:
    ``` 
    git clone https://github.com/Biomaterials-for-Medical-Devices-AI/GANs.git
    ```

2. **Install the required dependencies**:
    ```
    pip install -r requirements.txt
    ```
3. **Prepare your dataset**:

   In this project, we offer a standard dataset (MNIST) and a custom dataset (Biological). You can specify the dataset type using the `--dataset_name` argument when running training or evaluation scripts, current available options are: `mnist`, `biological`.
   - For custom datasets:
     - Place your images in the appropriate folder (e.g., `../Datasets/Topographies/raw/FiguresStacked Same Size 4X4`).
     - If using labeled data, ensure your label file is correctly formatted (e.g., CSV for biological data).
   - For standard datasets:
     - The framework will automatically download them.
     - It creates both training (MNISTDataset) and test (MNISTTest) datasets.

    Dataset creation is handled by the `make_dataset` function, which creates appropriate dataset objects and dataloaders.

 4. **Train**:

    You can train a model using the following command:
    ```
     python train.py --model_name WGANGP --dataset_name biological --n_epochs 200 --img_size 224 --batch_size 32
    ```
    or change options in [train.sh](./launch/train.sh) and run it:
    ```
    bash launch/train.sh
    ```
    All the options can be used during training process can be found in the [base_option.py](./options/base_option.py) and [train_option.py](./options/train_option.py) folder.

5. **Generate Images**:
   When you have a trained model, you can use it to generate new images using the following command:
   ```
   python predict.py --model_path ./logs/your_experiment_name/net_Generator_latest.pth
   ```
   For other options, please check the [predict_option.py](./options/predict_option.py) and [base_option.py](./options/base_option.py) file.

