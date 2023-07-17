# FiberAug
## Intro
This project explored various techniques or repositories for data augmentation, including [albumentaions](https://github.com/albumentations-team/albumentations), GAN (Generative Adversarial Network), Diffusion models etc.

## Albumentations
Albumentations is a powerful library that provides a wide range of image augmentation functions for training and enhancing machine learning models in the field of computer vision.

The library offers a comprehensive set of image transformation methods such as geometric transformations (scaling, rotation, cropping), color manipulations (brightness, contrast, saturation), noise addition, and many more.

In this project, we have further encapsulated the functionalities provided by Albumentations, enabling it to process images in batches at a given directory while recording the operation information.

To use this project, you can refer to the `test_aug.py` file. Simply modify the input and output directories and set up the corresponding pipeline, then you can run the program.

If you want to extend the functionalities further, you can refer to the content of `image_processor.py` file. Define new types by inheriting from `ImageProcessor` and complete the corresponding constructor, `_process` function, and `_process_info` function.

## GANs
Generative Adversarial Networks (GANs) are a class of machine learning models that consist of two components: a generator and a discriminator. The primary goal of GANs is to generate new data samples that resemble a given training dataset.

The generator is responsible for creating synthetic data samples. Initially, it generates random noise as input and tries to generate samples that resemble the training data. The discriminator, on the other hand, acts as a critic and tries to distinguish between real and generated samples.

During training, the generator and discriminator engage in a competitive process. The generator aims to produce samples that the discriminator cannot differentiate from real data, while the discriminator learns to improve its ability to distinguish between real and generated samples. This adversarial training process helps both models to improve over time.

The `gan.py` file in this project builds a simple GAN model that allows adjusting the network structure through various parameters to adapt to different tasks or achieve better results by appropriately tuning the parameters for the same task. Additionally, the `gan_trainer.py` file creates a trainer specifically designed for GAN networks. By providing a properly configured GAN network to the trainer, it can automatically train the network, making it more user-friendly.

However, due to limitations in GPU memory on the server, this network was not able to undergo training. Therefore, we do not have a clear understanding of its performance and have been unable to adjust its parameters based on the results.

## Diffusion model
### Why diffusion

## Hugging Face
### Why Hugging Face

## Problems

## Conclusion