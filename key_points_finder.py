import cv2
import numpy as np

from constant_variable import IMAGE_PATH
from read_image import ImageReader


class KeyPointsFinder():
    @staticmethod
    def find_keypoints_harris(image_path):
        title, image_greyscale = ImageReader.read_image_greyscale(image_path)
        title_for_color, image_color = ImageReader.read_image_color(image_path)
        image_greyscale = np.float32(image_greyscale)
        founded_key_points = cv2.cornerHarris(image_greyscale, 2, 3, 0.04)
        founded_key_points = cv2.dilate(founded_key_points, None)
        image_color[founded_key_points > 0.01 * founded_key_points.max()] = [0, 255, 0]
        return image_color

    @staticmethod
    def find_keypoints_shitomasi(image_path):
        title, image_greyscale = ImageReader.read_image_greyscale(image_path)
        title_for_color, image_color = ImageReader.read_image_color(image_path)
        founded_key_points = cv2.goodFeaturesToTrack(image_greyscale, 500, 0.01, 10)
        founded_key_points = np.int0(founded_key_points)
        for i in founded_key_points:
            x, y = i.ravel()
            cv2.circle(image_color, (x, y), 5, [0, 255, 0], -1)
        return image_color

    @staticmethod
    def find_keypoints_FAST(image_path):
        title_for_color, image_color = ImageReader.read_image_color(image_path)
        fast = cv2.FastFeatureDetector_create()
        founded_key_points = fast.detect(image_color, None)
        output_image = cv2.drawKeypoints(image_color, founded_key_points, None, color=(0, 255, 0))
        return output_image

    @staticmethod
    def find_keypoints_ORB(image_path):
        title_for_color, image_color = ImageReader.read_image_color(image_path)
        orb = cv2.cv2.ORB_create()
        founded_key_points = orb.detect(image_color, None)
        output_image = cv2.drawKeypoints(image_color, founded_key_points, None, color=(0, 255, 0), flags=0)
        return output_image


if __name__ == '__main__':
    ImageReader.plot_image(
        KeyPointsFinder.find_keypoints_harris(IMAGE_PATH),
        "Key points Harris's method")
    ImageReader.plot_image(
        KeyPointsFinder.find_keypoints_shitomasi(IMAGE_PATH),
        "Key points Shi-Tomasi's method")
    ImageReader.plot_image(
        KeyPointsFinder.find_keypoints_FAST(IMAGE_PATH),
        "Key points FAST method")
    ImageReader.plot_image(
        KeyPointsFinder.find_keypoints_ORB(IMAGE_PATH),
        "Key points ORB method")
