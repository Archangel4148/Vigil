# Select a window and begin capturing it
import cv2

from processing.processing_modules import ColorFinder, ImageFinder, DigitRegionReader
from screen_capture import wait_for_click_and_get_window, list_and_select_window, WindowCapture

CLICK_SELECT_WINDOW = True
if CLICK_SELECT_WINDOW:
    window = wait_for_click_and_get_window()
else:
    window = list_and_select_window()

if window is not None:
    # Find target RGB color:
    processor = ColorFinder(color_rgb=(73, 79, 137), tolerance=25)

    # Find target template image:
    # template_image = cv2.imread("images/pineapple.jpg")
    # processor = ImageFinder(template_image)

    # Read number(s) from region:
    # ocr_region = (100, 110, 200, 140)
    # processor = DigitRegionReader(ocr_region)

    wc = WindowCapture(window_id=window, processor=processor)
    wc.run()
