import cv2
import numpy as np

class BaseCaptureProcessor:
    def process(self, frame: np.ndarray) -> None:
        """Process a captured frame (OpenCV image)."""
        raise NotImplementedError("You're processing using a BaseCaptureProcessor. Instead use a subclass!")


class ColorFinder(BaseCaptureProcessor):
    def __init__(self, color_rgb: tuple[int, int, int], tolerance: int = 10):
        super().__init__()
        self.bgr = np.array(color_rgb[::-1], dtype=np.uint8)
        self.tolerance = tolerance

    def process(self, frame: np.ndarray) -> None:
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]  # Drop alpha channel if present

        # Use numpy arrays instead of tuples for inRange
        bgr_int = self.bgr.astype(np.int16)
        lower_bound = np.clip(bgr_int - self.tolerance, 0, 255).astype(np.uint8)
        upper_bound = np.clip(bgr_int + self.tolerance, 0, 255).astype(np.uint8)

        mask = cv2.inRange(frame, lower_bound, upper_bound)
        match_count = cv2.countNonZero(mask)

        if match_count > 0:
            print(f"Color {tuple(self.bgr[::-1])} FOUND in image ({match_count} matches).")
        else:
            print(f"Color {tuple(self.bgr[::-1])} NOT found.")

if __name__ == '__main__':
    # Test script to ensure logic is working
    frame = np.full((200, 200, 3), (0, 255, 0), dtype=np.uint8)  # Solid green image
    color_finder = ColorFinder((0, 255, 0), tolerance=10)
    color_finder.process(frame)