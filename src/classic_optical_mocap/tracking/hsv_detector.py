import cv2
import numpy as np

class HSVDetector:

    def __init__(
        self,
        lower_hsv,
        upper_hsv
    ):
        self.lower_hsv = np.array(lower_hsv)
        self.upper_hsv = np.array(upper_hsv)

    def detect(self, frame):

        hsv = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2HSV
        )

        mask = cv2.inRange(
            hsv,
            self.lower_hsv,
            self.upper_hsv
        )

        return mask