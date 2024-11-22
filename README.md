# GAN Framework for Image Generation - (Biomaterial Discovery) (Masters Dissertation)

## Overview

VersiGAN is a versatile GAN framework designed to facilitate the development and evaluation of GAN-based image generation models. This project integrates multiple state-of-the-art in addition to innovative GAN architectures, including **VanillaGAN**, **ACVanillaGAN**, **WGAN**, **ACWGAN**, **WGAN-GP**, **ACWGAN-GP**, **WGAN-WC**, **WCGAN-GP**, **BlurGAN**, **ACBlurGAN**, **STYLEGAN**, and **ACCBlurGAN**. It provides researchers and developers with a powerful and flexible toolkit, offering a modular architecture that allows for the easy creation, training, and evaluation of GAN models across various image generation tasks.


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

## Features
- **Modular Architecture**: The project is built with a modular architecture, enabling easy integration of new GAN architectures and the swapping of existing ones.
- **Versatility**: The framework supports a wide range of GAN architectures, making it suitable for various image generation tasks.
- **Customizable Training**: The framework offers extensive flexibility in configuring the training, evaluation, and prediction processes. Users can fine-tune a wide array of parameters, including, but not limited to, training strategies, model architecture, hyperparameters and dataset configurations. These adjustments can be made through the comprehensive options system, which includes base, train, predict, and evaluate options. The settings are easily accessible and modifiable via the options folder, shell scripts, or command-line arguments, allowing researchers and developers to precisely tailor the entire pipeline to their specific requirements and experimental needs.
- **Comprehensive Evaluation and Analysis**: The framework provides robust tools for model evaluation and result analysis. It includes an automated evaluation pipeline that calculates metrics such as FID scores and offers visualization capabilities for generated images. The Jupyter notebook allows for flexible, in-depth analysis of results, including the creation of video grids for visual comparison of different models or training stages.
 - **Advanced Performance Tracking and Experiment Management**: The project implements a robust logging system using TensorBoard. It tracks and records various performance metrics, losses, and learning rates throughout the training process. The system also saves generated images at regular intervals, allowing for visual inspection of the model's progress. Logs are organised by experiment name and include detailed configuration information, making it easy to compare different runs and analyse results over time.

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
