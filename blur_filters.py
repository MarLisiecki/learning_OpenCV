import cv2

from constant_variable import IMAGE_PATH
from read_image import ImageReader


class BlurFilters():
    @staticmethod
    def apply_gaussian_filter(image_path):
        title, image_with_color = ImageReader.read_image_color(image_path)
        blured_image = cv2.GaussianBlur(image_with_color, (25, 25), 0)
        return blured_image

    @staticmethod
    def apply_median_filter(image_path):
        title, image_with_color = ImageReader.read_image_color(image_path)
        blured_image = cv2.medianBlur(image_with_color, 25)  # size of window = 25
        return blured_image

    @staticmethod
    def apply_bilateral_filter(image_path):
        title, image_with_color = ImageReader.read_image_color(image_path)
        blured_image = cv2.bilateralFilter(image_with_color, 12, 100, 100)
        return blured_image


if __name__ == "__main__":
    ImageReader.plot_image(BlurFilters.apply_gaussian_filter(IMAGE_PATH), "Gausian blur")
    ImageReader.plot_image(BlurFilters.apply_median_filter(IMAGE_PATH), "Median blur")
    ImageReader.plot_image(BlurFilters.apply_bilateral_filter(IMAGE_PATH), "Bilateral blur")
