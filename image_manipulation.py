import cv2
import numpy as np

from constant_variable import IMAGE_PATH
from read_image import ImageReader


class ImageManipulation():
    def __init__(self, image_path: str):
        self.window_title, self.image = ImageReader.read_image_color(image_path)

    def get_size(self):
        height, width = self.image.shape[:2]
        return height, width

    def translation(self, vector_x: int, vector_y: int):
        height, width = self.get_size()
        translation_matrix = np.float32([[1, 0, vector_x], [0, 1, vector_y]])
        translated_image = cv2.warpAffine(self.image, translation_matrix, (width, height))
        return translated_image, self.window_title

    def rotation(self, angle: int):
        height, width = self.get_size()
        rotation_matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle, 1)
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (width, height))
        return rotated_image, window_title

    # Linear interpolation
    def scale_image_by_coefficient(self, x_coefficient: int, y_coefficient: int):
        scaled_image = cv2.resize(self.image, None, fx=x_coefficient, fy=y_coefficient, interpolation=cv2.INTER_CUBIC)
        return scaled_image, window_title

    def scale_image_by_size(self, x_size: int, y_size: int):
        height, width = self.get_size()
        scaled_image = cv2.resize(self.image, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
        return scaled_image, window_title


if __name__ == '__main__':
    manipulation_test = ImageManipulation(image_path=IMAGE_PATH)
    translated_image, window_title = manipulation_test.translation(vector_x=50, vector_y=25)
    ImageReader.plot_image(translated_image, window_title)
    rotated_image, window_title = manipulation_test.rotation(angle=45)
    ImageReader.plot_image(rotated_image, window_title)
    scaled_image, window_title = manipulation_test.scale_image_by_coefficient(x_coefficient=2, y_coefficient=2)
    ImageReader.plot_image(scaled_image, window_title)
    scaled_image, window_title = manipulation_test.scale_image_by_coefficient(x_coefficient=2, y_coefficient=2)
    ImageReader.plot_image(scaled_image, window_title)
