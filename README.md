# FiberAug
## Intro
This project explored various techniques or repositories for data augmentation, including [albumentaions](https://github.com/albumentations-team/albumentations), GAN (Generative Adversarial Network), Diffusion models etc.

## Albumentations
Albumentations is a powerful library that provides a wide range of image augmentation functions for training and enhancing machine learning models in the field of computer vision.

The library offers a comprehensive set of image transformation methods such as geometric transformations (scaling, rotation, cropping), color manipulations (brightness, contrast, saturation), noise addition, and many more.

In this project, we have further encapsulated the functionalities provided by Albumentations, enabling it to process images in batches at a given directory while recording the operation information.

To use this project, you can refer to the `test_aug.py` file. Simply modify the input and output directories and set up the corresponding pipeline, then you can run the program.

If you want to extend the functionalities further, you can refer to the content of `image_processor.py` file. Define new types by inheriting from `ImageProcessor` and complete the corresponding constructor, `_process` function, and `_process_info` function.

## GAN
### GAN & VAE

## Diffusion model
### Why diffusion

## Hugging Face
### Why Hugging Face