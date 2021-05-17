import cv2
import numpy as np

from constant_variable import IMAGE_PATH
from read_image import ImageReader


class ImageManipulation():
    def __init__(self, vector_x: int, vector_y:int):
        self.vector_x = vector_x
        self.vector_y = vector_y
        self.translation_matrix : list = np.float32([[1,0,vector_x], [0,1,vector_y]])


    @staticmethod
    def translation(image, translation_matrix, window_title):
        height, width = image.shape[:2]
        translated_image = cv2.warpAffine(image,translation_matrix,(width, height))
        ImageReader.plot_image(translated_image, window_title)

if __name__ == '__main__':
    translation_test = ImageManipulation(50, 25)
    window_title, input_image = ImageReader.read_image_color(IMAGE_PATH)
    ImageManipulation.translation(input_image, translation_test.translation_matrix, window_title)