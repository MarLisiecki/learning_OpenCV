import cv2
import matplotlib.pyplot as plt

from read_image import ImageReader
from constant_variable import IMAGE_PATH

class HistogramEqualization():
    @staticmethod
    def apply_histogram_equalization(image_path):
        title, image_greyscale = ImageReader.read_image_greyscale(image_path)
        equalized_image = cv2.equalizeHist(image_greyscale)
        return equalized_image

    @staticmethod
    def plot_histogram(image):
        plt.hist(image.ravel(), 256, [0,256])
        plt.show()

if __name__ == '__main__':
    title, input_image = ImageReader.read_image_greyscale(IMAGE_PATH)
    output_image = HistogramEqualization.apply_histogram_equalization(IMAGE_PATH)
    ImageReader.plot_image(input_image, 'Input photo')
    ImageReader.plot_image(output_image, 'Photo for histogram')
    HistogramEqualization.plot_histogram(output_image)


