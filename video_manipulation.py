import cv2
import numpy as np

from constant_variable import VIDEO_SOURCE
from read_video import VideoReader


class VideoManipulation():
    def __init__(self, video_path):
        self.capture = VideoReader.create_capture(video_path)

    def show_dynamic_scaling_video(self, scale, step, max_scale):
        while self.capture.isOpened():
            correct_read, video_frame = self.capture.read()
            if correct_read:
                if scale < max_scale:
                    scale += step
                scaled_frame = cv2.resize(video_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                cv2.imshow('Video dynamic scaling', scaled_frame)
                break_key = cv2.waitKey(40)
                if break_key == ord('q'):  # q like quit
                    break
            else:
                break
        self.capture.release()
        cv2.destroyAllWindows()

    def show_translated_video(self, start_vector_x: int, start_vector_y: int):
        while self.capture.isOpened():
            correct_read, video_frame = self.capture.read()
            if correct_read:
                height, width = video_frame.shape[:2]
                start_vector_x += 1
                start_vector_y += 1
                translation_matrix = np.float32([[1, 0, start_vector_x], [0, 1, start_vector_y]])
                translated_frame = cv2.warpAffine(video_frame, translation_matrix, (width, height))
                cv2.imshow('Video dynamic translation', translated_frame)
                break_key = cv2.waitKey(40)
                if break_key == ord('q'):  # q like quit
                    break
            else:
                break
        self.capture.release()
        cv2.destroyAllWindows()

    def show_rotated_video(self, start_angle):
        while self.capture.isOpened():
            correct_read, video_frame = self.capture.read()
            if correct_read:
                height, width = video_frame.shape[:2]
                start_angle += 1
                rotation_matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), start_angle, 1)
                rotated_frame = cv2.warpAffine(video_frame, rotation_matrix, (width, height))
                cv2.imshow('Video dynamic translation', rotated_frame)
                break_key = cv2.waitKey(40)
                if break_key == ord('q'):  # q like quit
                    break
            else:
                break
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_video_manipulation = VideoManipulation(VIDEO_SOURCE[0])
    # test_video_manipulation.show_dynamic_scaling_video(0.1, 0.1, 2)
    # test_video_manipulation.show_translated_video(0, 0)
    # test_video_manipulation.show_rotated_video(0)
