import os
import cv2
import json
import albumentations as A


class ImageProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Create txt file to store processing information
        with open(os.path.join(output_path, 'processing_info.json'), 'w') as f:
            f.write(self._process_info())

    def process(self, image_path):
        # Read an image with OpenCV and convert it to the RGB colorspace
        image = cv2.imread(os.path.join(self.input_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processing of image
        processed_image = self._process(image)

        # Convert the color back to BGR colorspace for OpenCV
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

        # Save image
        output_image_path = os.path.join(self.output_path, image_path)
        cv2.imwrite(output_image_path, processed_image)

    def _process(self, image):
        # Add image processing methods here
        # Return the processed image
        return image

    def _process_info(self):
        # Return processing information
        return 'Default processing information'


class Rotate(ImageProcessor):
    def __init__(self,
                 input_path,
                 output_path,
                 limit=15,
                 interpolation=1,
                 border_mode=4,
                 value=None,
                 mask_value=None,
                 rotate_method='largest_box',
                 crop_border=False,
                 p=1.0
                 ):
        """Rotate the input by an angle selected randomly from the uniform distribution.

        Args:
            input_path (str): Path of input file.
            output_path (str): Path of output file.
            limit ([int, int] or int, optional): 
                    Range from which a random angle is picked. 
                    If limit is a single int an angle is picked from (-limit, limit). 
                    Defaults to (-30, 30).
            interpolation (int, optional): 
                    Flag that is used to specify the interpolation algorithm. 
                    Should be one of: 
                    0: cv2.INTER_NEAREST, 
                    1: cv2.INTER_LINEAR, 
                    2: cv2.INTER_CUBIC, 
                    3: cv2.INTER_AREA, 
                    4: cv2.INTER_LANCZOS4.
                    Defaults to 1.
            border_mode (int, optional): 
                    Flag that is used to specify the pixel extrapolation method. 
                    Should be one of: 
                    0: cv2.BORDER_CONSTANT, 
                    1: cv2.BORDER_REPLICATE, 
                    2: cv2.BORDER_REFLECT, 
                    3: cv2.BORDER_WRAP, 
                    4: cv2.BORDER_REFLECT_101. 
                    Defaults to 4.
            value (int, float, list of ints, list of float, optional): 
                    Padding value if border_mode is cv2.BORDER_CONSTANT. 
                    Defaults to None.
            mask_value (int, float, list of ints, list of float, optional): 
                    Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks. 
                    Defaults to None.
            rotate_method (str, optional): 
                    Rotation method used for the bounding boxes. 
                    Should be one of "largest_box" or "ellipse". 
                    Defaults to 'largest_box'.
            crop_border (bool, optional): 
                    If True would make a largest possible crop within rotated image. 
                    Defaults to False.
            p (float, optional): 
                    Probability of applying the transform. 
                    Defaults to 1.0.
        """
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.rotate_method = rotate_method
        self.crop_border = crop_border
        self.p = p
        super().__init__(input_path, os.path.join(output_path, "Rotate"))

        self.transform = A.Rotate(limit=self.limit,
                                  interpolation=self.interpolation,
                                  border_mode=self.border_mode,
                                  value=self.value,
                                  mask_value=self.mask_value,
                                  rotate_method=self.rotate_method,
                                  crop_border=self.crop_border,
                                  p=self.p
                                  )

    def _process(self, image):
        return self.transform(image=image)['image']

    def _process_info(self):
        info_dict = {"Rotate": []}

        # Add information for each parameter
        info = [{"key": "input_path", "value": self.input_path},
                {"key": "output_path", "value": self.output_path},
                {"key": "limit", "value": str(self.limit)},
                {"key": "interpolation", "value": str(self.interpolation)},
                {"key": "border_mode", "value": str(self.border_mode)},
                {"key": "value", "value": str(self.value)},
                {"key": "mask_value", "value": str(self.mask_value)},
                {"key": "rotate_method", "value": self.rotate_method},
                {"key": "crop_border", "value": self.crop_border},
                {"key": "p", "value": str(self.p)}]
        info_dict["Rotate"].extend(info)

        # Convert dictionary to JSON string and return
        return json.dumps(info_dict, indent=4)


class ShiftScaleRotate(ImageProcessor):
    def __init__(self,
                 input_path,
                 output_path,
                 shift_limit=0.0625,
                 scale_limit=0.1,
                 rotate_limit=30,
                 interpolation=1,
                 border_mode=4,
                 value=None,
                 mask_value=None,
                 shift_limit_x=None,
                 shift_limit_y=None,
                 rotate_method='largest_box',
                 p=1.0
                 ):
        """Randomly apply affine transforms: translate, scale and rotate the input.

        Args:
            input_path (str): Path of input file.
            output_path (str): Path of output file.
            shift_limit ([float, float] or float, optional): 
                    Shift factor range for both height and width.
                    Absolute values for lower and upper bounds should lie in range [0, 1].
                    Defaults to (-0.0625, 0.0625).
            scale_limit ([float, float] or float, optional): 
                    Note that the scale_limit will be biased by 1. 
                    If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high). 
                    Defaults to (-0.1, 0.1).
            rotate_limit ([int, int] or int, optional): 
                    Range from which a random angle is picked. 
                    If limit is a single int an angle is picked from (-limit, limit). 
                    Defaults to (-30, 30).
            interpolation (int, optional): 
                    Flag that is used to specify the interpolation algorithm. 
                    Should be one of: 
                    0: cv2.INTER_NEAREST, 
                    1: cv2.INTER_LINEAR, 
                    2: cv2.INTER_CUBIC, 
                    3: cv2.INTER_AREA, 
                    4: cv2.INTER_LANCZOS4.
                    Defaults to 1.
            border_mode (int, optional): 
                    Flag that is used to specify the pixel extrapolation method. 
                    Should be one of: 
                    0: cv2.BORDER_CONSTANT, 
                    1: cv2.BORDER_REPLICATE, 
                    2: cv2.BORDER_REFLECT, 
                    3: cv2.BORDER_WRAP, 
                    4: cv2.BORDER_REFLECT_101. 
                    Defaults to 4.
            value (int, float, list of ints, list of float, optional): 
                    Padding value if border_mode is cv2.BORDER_CONSTANT. 
                    Defaults to None.
            mask_value (int, float, list of ints, list of float, optional): 
                    Padding value if border_mode is cv2.BORDER_CONSTANT applied for masks. 
                    Defaults to None.
            shift_limit_x ([float, float] or float, optional):
                    Shift factor range for width. 
                    If it is set then this value instead of shift_limit will be used for shifting width.
                    Absolute values for lower and upper bounds should lie in the range [0, 1].
                    Defaults to None.
            shift_limit_y ([float, float] or float, optional):
                    Shift factor range for height.
                    If it is set then this value instead of shift_limit will be used for shifting height.
                    Absolute values for lower and upper bounds should lie in the range [0, 1].
                    Defaults to None.
            rotate_method (str, optional): 
                    Rotation method used for the bounding boxes. 
                    Should be one of "largest_box" or "ellipse". 
                    Defaults to 'largest_box'.
            p (float, optional): 
                    Probability of applying the transform. 
                    Defaults to 1.0.
        """
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value
        self.shift_limit_x = shift_limit_x
        self.shift_limit_y = shift_limit_y
        self.rotate_method = rotate_method
        self.p = p
        super().__init__(input_path, os.path.join(output_path, "ShiftScaleRotate"))

        self.transform = A.ShiftScaleRotate(shift_limit=self.shift_limit,
                                            scale_limit=self.scale_limit,
                                            rotate_limit=self.rotate_limit,
                                            interpolation=self.interpolation,
                                            border_mode=self.border_mode,
                                            value=self.value,
                                            mask_value=self.mask_value,
                                            shift_limit_x=self.shift_limit_x,
                                            shift_limit_y=self.shift_limit_y,
                                            rotate_method=self.rotate_method,
                                            p=self.p
                                            )

    def _process(self, image):
        return self.transform(image=image)['image']

    def _process_info(self):
        info_dict = {"ShiftScaleRotate": []}

        # Add information for each parameter
        info = [{"key": "input_path", "value": self.input_path},
                {"key": "output_path", "value": self.output_path},
                {"key": "shift_limit", "value": str(self.shift_limit)},
                {"key": "scale_limit", "value": str(self.scale_limit)},
                {"key": "rotate_limit", "value": str(self.rotate_limit)},
                {"key": "interpolation", "value": str(self.interpolation)},
                {"key": "border_mode", "value": str(self.border_mode)},
                {"key": "value", "value": str(self.value)},
                {"key": "mask_value", "value": str(self.mask_value)},
                {"key": "shift_limit_x", "value": str(self.shift_limit_x)},
                {"key": "shift_limit_y", "value": str(self.shift_limit_y)},
                {"key": "rotate_method", "value": self.rotate_method},
                {"key": "p", "value": str(self.p)}]
        info_dict["ShiftScaleRotate"].extend(info)

        # Convert dictionary to JSON string and return
        return json.dumps(info_dict, indent=4)


class Flip(ImageProcessor):
    def __init__(self,
                 input_path,
                 output_path,
                 flip_mode='horizontal',
                 p=1.0
                 ):
        """Flip the input.

        Args:
            input_path (str): Path of input file.
            output_path (str): Path of output file.
            flip_mode (str, optional): 
                    Type of flip to apply, can be: 
                    1. 'horizontal': Flip the input either horizontally.
                    2. 'vertical': Flip the input either vertically.
                    3. None: Flip the input either horizontally, vertically or both horizontally and vertically. 
                    Defaults to 'horizontal'.
            p (float, optional): 
                    Probability of applying the transform. 
                    Defaults to 1.0.

        Raises:
            ValueError: If flip_mode is not one of 'horizontal', 'vertical' or None, a ValueError will be raised
        """
        self.p = p
        self.flip_mode = flip_mode

        if self.flip_mode == 'horizontal':
            self.transform = A.HorizontalFlip(p=self.p)
        elif self.flip_mode == 'vertical':
            self.transform = A.VerticalFlip(p=self.p)
        elif self.flip_mode is None:
            self.transform = A.Flip(p=self.p)
        else:
            raise ValueError(
                "Invalid flip mode. Allowed values are 'horizontal', 'vertical' or None.")

        super().__init__(input_path, os.path.join(output_path, "Flip"))

    def _process(self, image):
        return self.transform(image=image)['image']

    def _process_info(self):
        info_dict = {"Flip": []}

        # Add information for each parameter
        info = [{"key": "input_path", "value": self.input_path},
                {"key": "output_path", "value": self.output_path},
                {"key": "flip_mode", "value": self.flip_mode},
                {"key": "p", "value": str(self.p)}]
        info_dict["Flip"].extend(info)

        # Convert dictionary to JSON string and return
        return json.dumps(info_dict, indent=4)


# class StyleTransfer(ImageProcessor):
#     def __init__(self, input_path, output_path, model_path):
#         super().__init__(input_path, os.path.join(output_path, "StyleTransfer"))
#         self.model_path = model_path

#         # Load model
#         self.model = load_model(model_path)

#     def _process(self, image):
#         # Style transfer
#         stylized_image = self.model(image)
#         return stylized_image.numpy()

#     def _process_info(self):
#         # Return processing information for this class
#         return f"Model path: {self.model_path}"
