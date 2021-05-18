import cv2

from constant_variable import VIDEO_SOURCE


class VideoReader():
    @staticmethod
    def create_capture(video_source):
        capture = cv2.VideoCapture(video_source)
        return capture

    @staticmethod
    def show_video(capture):
        while capture.isOpened():
            correct_read, video_frame = capture.read()
            if correct_read:
                cv2.imshow('Video', video_frame)
                break_key = cv2.waitKey(40)
                if break_key == ord('q'):  # q like quit
                    break
            else:
                break
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_capture = VideoReader.create_capture(VIDEO_SOURCE[0])
    VideoReader.show_video(video_capture)
