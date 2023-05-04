import os
import cv2
from models import gan
from utils import gan_trainer
from utils import gan_generator
from utils import image_processor
from utils import text_processor
from utils.config import Config
from matplotlib import pyplot as plt
from tqdm import tqdm


def visualize(image):
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    plt.waitforbuttonpress(0)


def process(input_dir, output_dir, pipeline):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each input file using the pipeline and processor function
    print("Processing:")
    for filename in tqdm(os.listdir(input_dir)):
        for processor in pipeline:
            processor.process(filename)
    print("Finished.")


def text_augment():
    # Load configuration file
    config = Config("config.json")

    # Define input and output directories
    text_input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  config.raw_data_dir,
                                  "annotations")
    text_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   config.generated_data_dir,
                                   "test2")

    # Define text processing pipelines
    text_pipeline = [
        text_processor.ChangeCase(text_input_dir, text_output_dir)
    ]

    # Process text using the shared 'process' function
    process(text_input_dir, text_output_dir, text_pipeline)


def image_augment():
    # Load configuration file
    config = Config("config.json")

    # Define input and output directories
    image_input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   config.raw_data_dir,
                                   "images")
    image_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    config.generated_data_dir,
                                    "test1")

    # # test the code
    # print(os.path.join(image_input_dir, "000.jpg"))
    # image = cv2.imread(os.path.join(image_input_dir, "000.jpg"))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # visualize(image)

    # Define image and text processing pipelines
    image_pipeline = [
        # image_processor.Rotate(image_input_dir, image_output_dir)
        # image_processor.ShiftScaleRotate(image_input_dir, image_output_dir)
        image_processor.Flip(image_input_dir, image_output_dir)
        # image_processor.StyleTransfer(image_input_dir, image_output_dir, model_path=config.model_path)
    ]

    # Process images using the shared 'process' function
    process(image_input_dir, image_output_dir, image_pipeline)


def train_model():
    # Load configuration file
    config = Config("config.json")

    # Paths and directories
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            config.all_data_dir)
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            config.tests_dir,
                            "test0")
    model_info_dir = os.path.join(test_dir, "model_info")
    generated_images_dir = os.path.join(test_dir, "generated_images")

    # Create directories if they don't exist
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(model_info_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)

    # Initialize Generator and Discriminator
    net_generator = gan.Generator()
    net_discriminator = gan.Discriminator()

    # Create GAN trainer
    trainer = gan_trainer.GANTrainer(gen=net_generator,
                                     dis=net_discriminator,
                                     data_path=data_dir,
                                     save_path=model_info_dir)
    # Train GAN
    trainer.train(2)
    # Save train data
    trainer.save_train_data()

    # Generate images
    gan_generator.GANGenerator(generator=net_generator,
                               save_dir=generated_images_dir,
                               n_generated=10)


if __name__ == '__main__':
    train_model()
