import os
import cv2
from utils import image_processor
from utils import text_processor
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


# Get project file address
project_dir = os.path.abspath(os.path.join(os.getcwd()))

# Define input and output directories
image_data_dir = os.path.join(project_dir, "dataset", "raw_data", "images")
image_save_dir = os.path.join(project_dir, "dataset", "generated_data", "test1")
text_data_dir = os.path.join(project_dir, "dataset", "raw_data", "annotations")
text_save_dir = os.path.join(project_dir, "dataset", "generated_data", "test2")


# test the code
print(os.path.join(image_data_dir, "000.jpg"))
image = cv2.imread(os.path.join(image_data_dir, "000.jpg"))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
visualize(image)


# Define image and text processing pipelines
image_pipeline = [
    # image_processor.Rotate(image_input_dir, image_output_dir)
    # image_processor.ShiftScaleRotate(image_input_dir, image_output_dir)
    image_processor.Flip(image_data_dir, image_save_dir)
]

text_pipeline = [
    text_processor.ChangeCase(text_data_dir, text_save_dir)
]

process(image_data_dir, image_save_dir, image_pipeline)
process(text_data_dir, text_save_dir, text_pipeline)