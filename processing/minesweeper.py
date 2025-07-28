import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from processing.processing_modules import BaseCaptureProcessor


@dataclass
class Cell:
    position: tuple[int, int] | None = None
    state: str = "unknown"
    bomb_probability: float = float("inf")

    def __repr__(self):
        return f"Cell(state={self.state}, position={self.position})"

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return isinstance(other, Cell) and self.position == other.position

    @property
    def mine_number(self):
        if self.state.isdigit():
            return int(self.state)
        return None


class MinesweeperProcessor(BaseCaptureProcessor):
    PER_CELL_PROCESS_TIME = 0.000683  # s
    OVERRIDE_PROCESS_INTERVAL = 0.25

    def __init__(self):
        self.cell_templates = self.load_templates()
        self.last_process_time = 0
        self.last_board_search = 0
        self.process_interval = None
        self.last_result = None

        # Game position/dimensions
        self.board_region = None
        self.num_rows = None
        self.num_cols = None
        self.position_to_index = None

        # Game cell grids
        self.grid: list[list[Cell | None]] | None = None

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
            named_matches = [(name, x, y, w, h) for x, y, w, h in matches]
            all_matches.extend(named_matches)

        if not all_matches:
            raise RuntimeError("No matches found for board detection.")

        xs = [x for name, x, y, w, h in all_matches]
        ys = [y for name, x, y, w, h in all_matches]

        x_min = min(xs)
        y_min = min(ys)
        x_max = max(x + w for name, x, y, w, h in all_matches)
        y_max = max(y + h for name, x, y, w, h in all_matches)

        # Create the grid from the matches, and build the position lookup
        self.create_grid(all_matches)
        self.build_position_lookup()

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
        print(f"\nBoard found! ({self.num_rows}x{self.num_cols}, {len(all_matches)} cells)")
        print(f"Processing tuned to {1 / self.process_interval:.2f} FPS")

    def create_grid(self, all_matches):
        # Sort matches by y (row), then x (col)
        all_matches = sorted(all_matches, key=lambda m: (m[2], m[1]))
        self.grid = []
        current_row = []
        last_y = None
        row_tol = all_matches[0][4] // 2  # half cell height tolerance

        for match in all_matches:
            name, x, y, w, h = match
            if last_y is None:
                current_row.append(Cell(position=(x, y), state=name))
                last_y = y
                continue

            if abs(y - last_y) > row_tol:
                # New row detected, sort previous row by x before appending
                current_row.sort(key=lambda c: c.position[0])
                self.grid.append(current_row)

                current_row = [Cell(position=(x, y), state=name)]
                last_y = y
            else:
                # Same row, append match
                current_row.append(Cell(position=(x, y), state=name))

        # Sort and append the last row
        if current_row:
            current_row.sort(key=lambda c: c.position[0])
            self.grid.append(current_row)

        self.num_rows = len(self.grid)
        self.num_cols = max(len(row) for row in self.grid) if self.grid else 0

        # Pad rows to rectangular shape with None
        for row in self.grid:
            while len(row) < self.num_cols:
                row.append(None)

    def build_position_lookup(self):
        self.position_to_index = {}
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell is not None and cell.position is not None:
                    x, y = cell.position
                    self.position_to_index[(x, y)] = (r, c)

    def process(self, frame: np.ndarray) -> np.ndarray:
        now = time.time()

        if self.board_region is None:
            # Only search for the board once per second to prevent lag
            if now - self.last_board_search >= 1:
                self.last_board_search = now
                try:
                    self.find_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                except RuntimeError:
                    # Board not found, stay None
                    print("Board not found, will retry in 1 second.")
                    self.board_region = None

            return frame

        # Throttle processing speed
        if now - self.last_process_time >= self.process_interval:
            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Crop the capture to the board
            x0, y0, width, height = self.board_region
            gray_frame = gray_frame[y0:y0 + height, x0:x0 + width]
            result_frame = frame.copy()

            for name, (template, color) in self.cell_templates.items():
                self.update_cell_states(gray_frame, template, result_frame, color, offset=(x0, y0), state_name=name)
            self.last_process_time = now
            self.last_result = result_frame

            result_frame = self.mark_cells(result_frame)

            return result_frame
        else:
            # Return the last result
            return self.last_result if self.last_result is not None else frame

    def handle_keypress(self, key: int) -> None:
        if key == ord('r'):
            # Clear the board region for re-detection
            self.board_region = None

    def update_cell_states(
            self,
            gray_img: np.ndarray,
            template: np.ndarray,
            output_image: np.ndarray,
            rect_color: tuple = (0, 0, 255),
            offset: tuple[int, int] = (0, 0),
            state_name: str = None
    ) -> tuple[np.ndarray, int]:
        # Get the matches
        matches = get_matches(gray_img, template)

        # Outline the matches
        for x, y, w, h in matches:
            x += offset[0]
            y += offset[1]
            # cv2.rectangle(output_image, (x, y), (x + w, y + h), rect_color, 2)

            # Update the state grid
            if state_name is not None:
                snapped_x, snapped_y = snap_position(x, y, [cell.position[:2] for row in self.grid for cell in row if
                                                            cell is not None])
                idx = self.position_to_index.get((snapped_x, snapped_y))
                if idx is not None:
                    r, c = idx
                    cell = self.grid[r][c]
                    if cell is not None:
                        cell.state = state_name

        # Update mine probabilities (first with basic sweep, then with subset inference)
        self.update_mine_probabilities_basic()
        apply_subset_inference(self.grid)

        return output_image, len(matches)

    def mark_cells(self, frame) -> np.ndarray:
        # Draw green circles on safe cells
        cell_w, cell_h = self.cell_templates["closed"][0].shape[::-1]  # (width, height)
        marked_cell = False
        game_over = True
        for row in self.grid:
            for cell in row:
                if cell is not None and cell.position is not None and cell.state == "closed":
                    game_over = False
                    x, y = cell.position
                    center = (x + cell_w // 2, y + cell_h // 2)
                    if cell.bomb_probability == 0.0:
                        marked_cell = True
                        cv2.circle(frame, center, radius=cell_w // 4, color=(0, 255, 0), thickness=2)
                    elif cell.bomb_probability == 1.0:
                        marked_cell = True
                        cv2.circle(frame, center, radius=cell_w // 4, color=(0, 0, 255), thickness=2)
        if not marked_cell and not game_over:
            # No cells were marked, mark the best guess with a yellow circle
            best_guess = min([cell for row in self.grid for cell in row if cell is not None and cell.state == "closed"],
                             key=lambda cell: cell.bomb_probability)
            x, y = best_guess.position
            center = (x + cell_w // 2, y + cell_h // 2)
            cv2.circle(frame, center, radius=cell_w // 4, color=(0, 255, 255), thickness=2)

        return frame

    def update_mine_probabilities_basic(self):
        if self.grid is None:
            return

        # Reset all bomb probabilities to infinity before recalculation
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell is None:
                    continue
                cell.bomb_probability = float("inf")
                # Don't reset cells already confirmed safe (bomb_probability == 0)
                if cell.state != "closed":
                    # Only reset if they are still unknown
                    cell.bomb_probability = 0.0
                if cell.bomb_probability != 0.0:
                    cell.bomb_probability = float("inf")

        # Process numbered cells to update neighbors probabilities
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell is None:
                    continue
                if cell.mine_number is not None:
                    adjacent = get_adjacent_cells(r, c, self.grid)
                    flagged_count = sum(1 for adj_cell in adjacent if adj_cell.state == "flag")
                    unknown_cells = [adj_cell for adj_cell in adjacent if
                                     adj_cell.state == "closed" and adj_cell.bomb_probability != 1.0]
                    remaining_mines = cell.mine_number - flagged_count

                    if remaining_mines == 0:
                        # All mines accounted for, unknown neighbors are safe
                        for adj_cell in unknown_cells:
                            # Mark as safe only if not already marked as mine or flagged
                            adj_cell.bomb_probability = 0.0
                    elif unknown_cells:
                        prob = remaining_mines / len(unknown_cells)
                        for adj_cell in unknown_cells:
                            if prob == 1.0:
                                # If a cell has a probability of 1.0, mark it as a mine
                                adj_cell.bomb_probability = 1.0
                            elif adj_cell.bomb_probability not in (0.0, 1.0) and adj_cell.bomb_probability > prob:
                                # If the cell is unknown, and the new probability is lower, update it
                                adj_cell.bomb_probability = prob


def get_matches(gray_img: np.ndarray, gray_template: np.ndarray, threshold=0.99):
    w, h = gray_template.shape[::-1]

    res = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return [(int(pt[0]), int(pt[1]), w, h) for pt in zip(*loc[::-1])]  # list of (x, y, w, h) for each match


def snap_position(x: int, y: int, known_positions: list[tuple[int, int]]) -> tuple[int, int]:
    # Returns the known (x, y) that's closest to the given (x, y)
    return min(known_positions, key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y))


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

    print(f"Total processing time: {elapsed:.4f} seconds")
    print(f"Cells matched: {cell_count}")
    print(f"Estimated PER_CELL_PROCESS_TIME: {elapsed / cell_count:.6f} seconds")

    return elapsed / cell_count


def get_adjacent_cells(row: int, col: int, grid: list[list[Cell | None]], print_debug=False) -> list[Cell]:
    num_rows = len(grid)
    num_cols = len(grid[0]) if num_rows > 0 else 0
    adjacent_cells = []
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in offsets:
        if 0 <= row + dx < num_rows and 0 <= col + dy < num_cols:
            adjacent_cells.append(grid[row + dx][col + dy])
            if print_debug:
                print(f"({row + dy}, {col + dx}) - {grid[row + dy][col + dx].state}  FROM ({row}, {col})")
    return adjacent_cells


def apply_subset_inference(grid: list[list[Cell | None]]):
    constraints = []

    # Step 1: Build constraints from numbered cells
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell is None or cell.mine_number is None:
                continue

            adjacent = get_adjacent_cells(r, c, grid)
            unknowns = [adj for adj in adjacent if adj is not None and adj.state == "closed"]
            flagged = sum(1 for adj in adjacent if adj is not None and adj.state == "flag")
            remaining = cell.mine_number - flagged

            if remaining > 0 and unknowns:
                constraints.append((set(unknowns), remaining))

    # Step 2: Compare constraints for subset relationships
    for i in range(len(constraints)):
        for j in range(len(constraints)):
            if i == j:
                continue
            cells_i, mines_i = constraints[i]
            cells_j, mines_j = constraints[j]

            if cells_i.issubset(cells_j):
                diff_cells = cells_j - cells_i
                diff_mines = mines_j - mines_i

                # If the difference is zero mines, then all diff_cells are safe
                if diff_mines == 0:
                    for cell in diff_cells:
                        if cell.bomb_probability != 0.0:
                            cell.bomb_probability = 0.0

                # If the number of diff_cells == diff_mines, all are mines
                elif diff_mines == len(diff_cells):
                    for cell in diff_cells:
                        if cell.bomb_probability != 1.0:
                            cell.bomb_probability = 1.0


if __name__ == "__main__":
    # Load an image
    path = "../images/minesweeper/difficult_game.png"
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Failed to load image from {path}")

    processor = MinesweeperProcessor()
    processor.OVERRIDE_PROCESS_INTERVAL = None  # ignore intervals

    # Force find the board once
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processor.find_board(gray)

    # Process the frame once (this updates cell states and probabilities)
    output = processor.process(frame)

    # Get cell size (assumes all cells are the same size)
    w, h = processor.cell_templates["closed"][0].shape[::-1]

    # Set font and style for probability text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    prob_color = (0, 0, 255)  # Red text

    # Draw probability text on each cell
    for row in processor.grid:
        for cell in row:
            if cell is not None and cell.position is not None and cell.state == "closed":
                x, y = cell.position
                if cell.bomb_probability is not None:
                    prob_text = f"{cell.bomb_probability * 100:.1f}%"
                    text_pos = (x + 2, y + 12)
                    cv2.putText(output, prob_text, text_pos, font, font_scale, prob_color, font_thickness, cv2.LINE_AA)

    # Save the resulting image
    cv2.imwrite("result.png", output)
    print("Result saved as result.png")
