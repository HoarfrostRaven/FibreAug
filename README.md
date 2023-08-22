# FiberAug
## Intro
This project explored various techniques or repositories for data augmentation, including [albumentaions](https://github.com/albumentations-team/albumentations), GAN (Generative Adversarial Network), Diffusion models etc.

## Albumentations
![Albumentations](https://camo.githubusercontent.com/3bb6e4bb500d96ad7bb4e4047af22a63ddf3242a894adf55ebffd3e184e4d113/68747470733a2f2f686162726173746f726167652e6f72672f776562742f62642f6e652f72762f62646e6572763563746b75646d73617a6e687734637273646669772e6a706567)

Albumentations is a powerful library that provides a wide range of image augmentation functions for training and enhancing machine learning models in the field of computer vision.

The library offers a comprehensive set of image transformation methods such as geometric transformations (scaling, rotation, cropping), color manipulations (brightness, contrast, saturation), noise addition, and many more.

In this project, we have further encapsulated the functionalities provided by Albumentations, enabling it to process images in batches at a given directory while recording the operation information.

To use this project, you can refer to the `test_aug.py` file. Simply modify the input and output directories and set up the corresponding pipeline, then you can run the program.

If you want to extend the functionalities further, you can refer to the content of `image_processor.py` file. Define new types by inheriting from `ImageProcessor` and complete the corresponding constructor, `_process` function, and `_process_info` function.

## GANs
Generative Adversarial Networks (GANs) are a class of machine learning models that consist of two components: a generator and a discriminator. The primary goal of GANs is to generate new data samples that resemble a given training dataset.

![GANs](https://github.com/HoarfrostRaven/FibreAug/blob/main/readme_images/basic_gan.PNG?raw=true)

The generator is responsible for creating synthetic data samples. Initially, it generates random noise as input and tries to generate samples that resemble the training data. The discriminator, on the other hand, acts as a critic and tries to distinguish between real and generated samples.

During training, the generator and discriminator engage in a competitive process. The generator aims to produce samples that the discriminator cannot differentiate from real data, while the discriminator learns to improve its ability to distinguish between real and generated samples. This adversarial training process helps both models to improve over time.

The `gan.py` file in this project builds a simple GAN model that allows adjusting the network structure through various parameters to adapt to different tasks or achieve better results by appropriately tuning the parameters for the same task. Additionally, the `gan_trainer.py` file creates a trainer specifically designed for GAN networks. By providing a properly configured GAN network to the trainer, it can automatically train the network, making it more user-friendly.

En raison de contraintes de mémoire GPU, on n'a pas pu effectuer les tests sur l'ensemble de données cible. On a utilisé un ensemble de données d'images de pixels de taille 16*16 pour les tests. On a testé les fonctions de perte BCELoss et Wasserstein, mais même après avoir entraîné plus de 100 epochs, les résultats n'étaient toujours pas satisfaisants.

## Diffusion model
The diffusion model is a probabilistic generative model that aims to model complex data distributions by iteratively applying a diffusion process. It is particularly effective in modeling high-dimensional data such as images and audio.

The underlying principle of the diffusion model is to transform an initial sample (e.g., noise) into a target sample by gradually adding noise at each iteration. The diffusion process starts with a random noise vector and applies a series of transformations, referred to as diffusion steps. In each step, the noise is gradually added to the original sample, creating a sequence of intermediate samples. This process is designed to smoothen and disentangle the high-dimensional data space, allowing the model to capture complex patterns and generate realistic samples.

During training, the diffusion model learns the parameters that govern the diffusion process by maximizing the likelihood of generating the target sample from the initial noise vector. This is typically achieved using maximum likelihood estimation or variational inference techniques. Once trained, the model can generate new samples by running the diffusion process in reverse, starting from a fixed point and iteratively removing noise until the initial noise vector is reached.

The neural network architecture we use for diffusion models is a UNet.

![UNet](https://hoarfrostraven.github.io/2023/06/12/Deep-Learning-Diffusion-Models-Part-2/Model%20Structure.jpg)

The most important thing about UNet is that it takes an image as input and output an image in the same size of the input. What it does is first embeds information about the input into an embedding that compresses all the information in smaller space, so it downsamples with a lot of convolutional layers. And then it upsamples with the same number of upsampling blocks until get the output.

This project integrates multiple running modes, allowing users to choose whether to import pre-trained models, control the output images using prompts, and accelerate the sampling process using DDIM (Denoising Diffusion Implicit Models). These choices can be easily modified in the "test_diffusion_model.ipynb" file to suit the user's preferences.

The structural details of the network can be found in the file "context_unet.py," and any modifications to the network's structure should also be made here.

The model has been proven to function properly on smaller-sized image datasets and delivers satisfactory results. However, due to memory limitations, the performance of the model has not been validated yet for large-scale images as required by the project.

## Diffuser
![Diffuser](https://raw.githubusercontent.com/huggingface/diffusers/77aadfee6a891ab9fcfb780f87c693f7a5beeb8e/docs/source/imgs/diffusers_library.jpg)
[Diffuser](https://huggingface.co/docs/diffusers/index) is a tool developed by [Hugging Face](https://huggingface.co/) that assists us in building a diffusion model. Due to its excellent internal implementation, it enables us to handle larger-sized images while maintaining a structure similar to the diffusion model we construct ourselves. The trade-off is that the training speed is slower, and it's difficult to modify the code to meet our needs except for the network structure.

All code is available in the file "test_diffusers.ipynb". Before running the code, it's necessary to create the corresponding database and model folder in Hugging Face by following the instructions provided in the comments. According to the [Diffuser documentation](https://huggingface.co/docs/diffusers/index), this will enable us to save the trained model to the cloud. However, this cloud-saving functionality hasn't been achieved in the actual experimental process. In order to save the trained model, I attempted to use the conventional method of saving checkpoints. However, for reasons unknown, this only resulted in generating a folder that cannot be opened.

After completing the preparatory steps, you can modify the network structure according to the comments and proceed with training the constructed network. By applying PEFT, the number of parameters that need to be trained can be reduced. However, this requires a network that has already undergone pretraining. If applied directly to a new network, it would significantly diminish the training effectiveness.

## Problems
The entire project was carried out with insufficient GPU memory, which resulted in the model's inability to handle large-sized images. Improvement can be attempted through the following methods:

1. Utilizing cloud computing.
2. Applying PEFT to the model for reducing the number of parameters that need training.
3. Purchasing new GPU.