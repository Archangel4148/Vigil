import cv2
import numpy as np


class BaseCaptureProcessor:
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a captured frame (OpenCV image), returning the processed image."""
        raise NotImplementedError("You're processing using a BaseCaptureProcessor. Instead use a subclass!")


class ColorFinder(BaseCaptureProcessor):
    def __init__(self, color_rgb: tuple[int, int, int], tolerance: int = 10):
        super().__init__()
        self.bgr = np.array(color_rgb[::-1], dtype=np.uint8)
        self.tolerance = tolerance

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]  # Drop alpha channel if present

        # Use numpy arrays instead of tuples for inRange
        bgr_int = self.bgr.astype(np.int16)
        lower_bound = np.clip(bgr_int - self.tolerance, 0, 255).astype(np.uint8)
        upper_bound = np.clip(bgr_int + self.tolerance, 0, 255).astype(np.uint8)

        mask = cv2.inRange(frame, lower_bound, upper_bound)
        match_count = cv2.countNonZero(mask)

        # Return the original frame if no matches
        if match_count == 0:
            return None

        # Convert frame to grayscale and back to BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Replace grayscale pixels with highlight color where mask is True
        highlight_frame = gray_bgr.copy()
        highlight_frame[mask > 0] = self.bgr  # Or another highlight color if preferred

        # Draw highlights around detected areas (high performance cost)
        # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(highlight_frame, contours, -1, (0, 255, 255), thickness=2)  # Yellow contour

        # Fall back on the original frame if highlight_frame breaks
        if not isinstance(highlight_frame, np.ndarray) or highlight_frame.size == 0:
            return frame

        return highlight_frame


if __name__ == '__main__':
    # Test script to ensure logic is working
    frame = np.full((200, 200, 3), (0, 255, 0), dtype=np.uint8)  # Solid green image
    color_finder = ColorFinder((0, 255, 0), tolerance=10)
    color_finder.process(frame)
