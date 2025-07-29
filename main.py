# Select a window and begin capturing it

from processing.minesweeper import MinesweeperProcessor
from screen_capture import wait_for_click_and_get_window, list_and_select_window, WindowCapture

CLICK_SELECT_WINDOW = True
if CLICK_SELECT_WINDOW:
    window = wait_for_click_and_get_window()
else:
    window = list_and_select_window()

if window is not None:
    # Find target RGB color:
    # processor = ColorFinder(color_rgb=(73, 79, 137), tolerance=25)

    # Find target template image:
    # template_image = cv2.imread("images/pineapple.jpg")
    # processor = ImageFinder(template_image)

    # Read number(s) from region:
    # ocr_region = (100, 110, 200, 140)
    # processor = DigitRegionReader(ocr_region)

    # Analyze a minesweeper board:
    processor = MinesweeperProcessor(autoplay_delay=0.05, stop_on_guess=False, auto_revive=True)

    wc = WindowCapture(window_id=window, processor=processor)
    processor.set_capture_interface(wc)
    wc.run()
