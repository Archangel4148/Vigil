# Select a window and begin capturing it
from processing.processing_modules import ColorFinder
from screen_capture import wait_for_click_and_get_window, list_and_select_window, WindowCapture

CLICK_SELECT_WINDOW = True
if CLICK_SELECT_WINDOW:
    window = wait_for_click_and_get_window()
else:
    window = list_and_select_window()

if window is not None:
    processor = ColorFinder(color_rgb=(21, 255, 0))
    wc = WindowCapture(window_id=window, processor=processor)
    wc.run()
