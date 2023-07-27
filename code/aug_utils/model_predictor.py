import torch


class ModelPredictor:
    def __init__(self, model, device=None):
        # Init function, receiving model
        self.model = model
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the device
        self.model.to(self.device)

        # Set the model to evaluation mode
        self.model.eval()

    def predict(self, image_data, text_data):
        # Predict function which takes input and returns the predicted result

        # Move the inputs to the device
        image_data, text_data = image_data.to(
            self.device), text_data.to(self.device)

        with torch.no_grad():
            # Forward pass
            output = self.model(image_data, text_data)

            # Get the predicted class
            _, predicted = output.max(1)

        return predicted.item()


# # Usage example
# # Create model and load pre-trained parameters
# model = MyModel()
# model.load_state_dict(torch.load('my_model.pth'))

# # Create Predictor
# predictor = ModelPredictor(model)

# # Prediction for a single input
# image_data = torch.rand(1, 3, 224, 224)
# text_data = torch.rand(1, 100)
# predicted = predictor.predict(image_data, text_data)
# print(predicted)

# # Prediction for batch input
# image_data = torch.rand(4, 3, 224, 224)
# text_data = torch.rand(4, 100)
# predicted = predictor.predict(image_data, text_data)
# print(predicted)
