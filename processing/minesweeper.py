import os
import time

import cv2
import numpy as np

from processing.processing_modules import BaseCaptureProcessor


class MinesweeperProcessor(BaseCaptureProcessor):
    PER_CELL_PROCESS_TIME = 0.000683  # s
    OVERRIDE_PROCESS_INTERVAL = 0.25

    def __init__(self):
        self.cell_templates = self.load_templates()
        self.last_process_time = 0
        self.last_board_search = 0
        self.process_interval = None
        self.last_result = None

        # Game position/dimensions:
        self.board_region = None

        print("Loaded templates! While focused on the screen capture window, press 'r' to re-detect the board region.")

    @staticmethod
    def load_templates() -> dict[str, tuple[np.ndarray, tuple[int, int, int]]]:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cells_dir = os.path.join(base_dir, "..", "images", "minesweeper", "cells")

        def load_img(name):
            path = os.path.join(cells_dir, name)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Template image not found: {path}")
            return img

        return {
            "empty": (load_img("empty_cell.png"), (80, 80, 80)),
            "empty_bottom": (load_img("empty_cell_bottom.png"), (80, 80, 120)),
            "empty_right": (load_img("empty_cell_right.png"), (80, 120, 80)),
            "empty_bottom_right": (load_img("empty_cell_bottom_right.png"), (80, 120, 120)),
            "closed": (load_img("closed_cell.png"), (255, 0, 0)),
            "flag": (load_img("flag.png"), (0, 0, 255)),
            "mine": (load_img("mine.png"), (0, 0, 0)),
            "mine_clicked": (load_img("mine_clicked.png"), (0, 0, 0)),
            "1": (load_img("cell_1.png"), (0, 255, 0)),
            "2": (load_img("cell_2.png"), (255, 255, 0)),
            "3": (load_img("cell_3.png"), (0, 165, 255)),
            "4": (load_img("cell_4.png"), (255, 0, 255)),
            "5": (load_img("cell_5.png"), (0, 255, 255)),
            "6": (load_img("cell_6.png"), (255, 255, 255)),
        }

    def find_board(self, gray_img):
        all_matches = []

        # Collect matches from all templates
        for name, (template, _) in self.cell_templates.items():
            matches = get_matches(gray_img, template, threshold=0.99)
            all_matches.extend(matches)

        if not all_matches:
            raise RuntimeError("No matches found for board detection.")

        xs = [x for x, y, w, h in all_matches]
        ys = [y for x, y, w, h in all_matches]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(x + w for x, y, w, h in all_matches)
        y_max = max(y + h for x, y, w, h in all_matches)

        padding = 10  # pixels of padding to add on each side
        x0 = max(0, x_min - padding)
        y0 = max(0, y_min - padding)
        width = x_max - x_min + 2 * padding
        height = y_max - y_min + 2 * padding

        self.board_region = (x0, y0, width, height)

        # Set process time based on the number of detected cells
        self.process_interval = (
            self.PER_CELL_PROCESS_TIME * len(all_matches)
            if self.OVERRIDE_PROCESS_INTERVAL is None
            else self.OVERRIDE_PROCESS_INTERVAL
        )
        print(f"Processing tuned to {1 / self.process_interval:.2f} FPS")

    def process(self, frame: np.ndarray) -> np.ndarray:
        now = time.time()

        if self.board_region is None:
            # Only search for the board once per second to prevent lag
            if now - self.last_board_search >= 1:
                self.last_board_search = now
                try:
                    self.find_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    print("Board found!")
                except RuntimeError:
                    # Board not found, stay None
                    print("Board not found, will retry in 1 second.")
                    self.board_region = None

            return frame

        # Throttle processing to process interval
        if now - self.last_process_time >= self.process_interval:
            # Process the frame
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Crop the capture to the board
            x0, y0, width, height = self.board_region
            gray_frame = gray_frame[y0:y0 + height, x0:x0 + width]
            result_frame = frame.copy()

            # Highlight all cell matches
            for template, color in self.cell_templates.values():
                highlight_matches(gray_frame, template, result_frame, color, offset=(x0, y0))
            self.last_process_time = now
            self.last_result = result_frame
            return result_frame
        else:
            # Return the last result
            return self.last_result if self.last_result is not None else frame

    def handle_keypress(self, key: int) -> None:
        if key == ord('r'):
            # Clear the board region for re-detection
            self.board_region = None


def get_matches(gray_img: np.ndarray, gray_template: np.ndarray, threshold=0.99):
    w, h = gray_template.shape[::-1]

    res = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return [(pt[0], pt[1], w, h) for pt in zip(*loc[::-1])]  # list of (x, y, w, h) for each match


def highlight_matches(
        gray_img: np.ndarray,
        template: np.ndarray,
        output_image: np.ndarray,
        rect_color: tuple = (0, 0, 255),
        offset: tuple[int, int] = (0, 0)
) -> tuple[np.ndarray, int]:
    # Get the matches
    matches = get_matches(gray_img, template)

    # Outline the matches
    for x, y, w, h in matches:
        x += offset[0]
        y += offset[1]
        cv2.rectangle(output_image, (x, y), (x + w, y + h), rect_color, 2)

    return output_image, len(matches)


def benchmark_per_cell_time(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Failed to load image from {image_path}")

    processor = MinesweeperProcessor()
    processor.OVERRIDE_PROCESS_INTERVAL = 0  # Force processing

    start = time.time()
    processor.process(frame)
    end = time.time()

    elapsed = end - start
    board_region = processor.board_region
    if board_region is None:
        raise RuntimeError("Board region was not detected.")

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x0, y0, w, h = board_region
    cropped = gray_frame[y0:y0 + h, x0:x0 + w]

    # Use just one reliable template to count the number of cells
    closed_template = processor.cell_templates["closed"][0]
    matches = get_matches(cropped, closed_template)
    cell_count = len(matches)

    if cell_count == 0:
        raise RuntimeError("No closed cells found for benchmarking.")

    per_cell_time = elapsed / cell_count
    print(f"Total processing time: {elapsed:.4f} seconds")
    print(f"Cells matched: {cell_count}")
    print(f"Estimated PER_CELL_PROCESS_TIME: {per_cell_time:.6f} seconds")

    return per_cell_time


if __name__ == "__main__":
    # Profile the per-cell processing time for this computer
    image_path = "../images/minesweeper/start_game.png"
    per_cell_time = benchmark_per_cell_time(image_path)
    print(f"\nYou can set `PER_CELL_PROCESS_TIME = {per_cell_time:.6f}` in your MinesweeperProcessor.")
