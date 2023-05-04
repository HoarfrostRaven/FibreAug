import os
import torch
from torchvision.utils import save_image


class GANGenerator:
    def __init__(self, generator, save_dir, n_generated=0, device=None):
        # Init function, receiving generator
        self.generator = generator
        self.save_dir = save_dir
        self.n_generated = n_generated
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Move the generator to the device
        self.generator.to(self.device)

        # Set the generator to evaluation mode
        self.generator.eval()

    def generate(self, n_images=1, latent_dim=100):
        # Generate function which takes input and returns the generated images

        # Create random noise for input to generator
        z = torch.randn(n_images, latent_dim, 1, 1, device=self.device)

        # Generate images from noise
        with torch.no_grad():
            generated_images = self.generator(z)

        # Save generated images
        for i in range(n_images):
            save_image(generated_images[i],
                       os.path.join(self.save_dir,
                                    f"{self.n_generated+i}.png"))

        # Update the number of generated images
        self.n_generated += n_images

        return generated_images


# # Usage example
# # Create model and load pre-trained parameters
# generator = Generator(nz=100, generator_feature_size=64, num_channels=3)
# generator.load_state_dict(torch.load('generator.pth'))

# # Create Generator
# generator_wrapper = GANGenerator(generator, "generated_images")

# # Generate images
# generated_images = generator_wrapper.generate(n_images=10, latent_dim=100)
