import cv2

from constant_variable import IMAGE_PATH


class ImageReader():
    @staticmethod
    def read_image_color(image_path):
        image_with_color = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        title = "Image in color"
        return title, image_with_color

    @staticmethod
    def read_image_greyscale(image_path):
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        title = "Image in greyscale"
        return title, image_greyscale

    @staticmethod
    def show_image(title, image):
        cv2.imshow(title, image)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    window_title, input_image = ImageReader.read_image_greyscale(IMAGE_PATH)
    ImageReader.show_image(window_title, input_image)

