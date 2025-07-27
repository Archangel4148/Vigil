import time

import cv2
import numpy as np
import win32api
import win32con
import win32gui
from mss import mss

from processing.processing_modules import BaseCaptureProcessor


class MonitorCapture:
    def __init__(self, monitor_number=1, region=None, processor: BaseCaptureProcessor = None,
                 resize_factor=0.5, show_fps=True, print_fps=False, visible=True):
        self.monitor_number = monitor_number
        self.processor = processor
        self.resize_factor = resize_factor
        self.show_fps = show_fps
        self.print_fps = print_fps
        self.visible = visible
        self.sct = mss()
        self.monitor = region if region else self._get_monitor_region()
        self.prev_time = time.perf_counter()
        self.fps_list = [0.0] * 10
        self.fps_display_delay = 0
        self.fps_delayed = 0

    def _get_monitor_region(self):
        mon = self.sct.monitors[self.monitor_number]
        return {
            "top": mon["top"],
            "left": mon["left"],
            "width": mon["width"],
            "height": mon["height"],
            "mon": self.monitor_number
        }

    def _calculate_fps(self):
        curr_time = time.perf_counter()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        self.fps_list.append(fps)
        self.fps_list.pop(0)
        avg_fps = sum(self.fps_list) / len(self.fps_list)

        self.fps_display_delay += 1
        if self.fps_display_delay >= 3:
            self.fps_delayed = avg_fps
            self.fps_display_delay = 0

        return avg_fps

    def run(self):
        while True:
            img = np.asarray(self.sct.grab(self.monitor))
            avg_fps = self._calculate_fps()

            # Process the image using the current module
            if (processed_frame := self.processor.process(img)) is not None:
                img = processed_frame

            if self.visible:
                if self.show_fps:
                    # Draw FPS on screen (costs performance)
                    cv2.putText(img, f"FPS: {self.fps_delayed:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Resize and display (costs performance)
                resized_img = cv2.resize(img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                cv2.imshow("Monitor Capture", resized_img)

            if self.print_fps:
                # Print to terminal (costs performance)
                print(f"Average FPS: {avg_fps:.2f}")

            # Handle window closing
            cv2.waitKey(1)
            if cv2.getWindowProperty("Monitor Capture", cv2.WND_PROP_VISIBLE) < 1:
                break

        # Close everything
        cv2.destroyAllWindows()
        self.sct.close()


class WindowCapture(MonitorCapture):
    def __init__(self, window_id: int = None, window_title: str = "", processor: BaseCaptureProcessor = None,
                 resize_factor=0.5, show_fps=True, print_fps=False,
                 visible=True):
        if window_id:
            self.target_window_id = window_id
            self.window_title = win32gui.GetWindowText(self.target_window_id)
        elif window_title:
            self.window_title = window_title
            self.target_window_id = win32gui.FindWindow(None, self.window_title)
        else:
            self.target_window_id = win32gui.GetForegroundWindow()
            self.window_title = win32gui.GetWindowText(self.target_window_id)

        # Validate window capture creation
        if not self.target_window_id or not win32gui.IsWindowVisible(self.target_window_id):
            raise Exception(f"Window not found or is not visible: \"{self.window_title}\"")

        print(
            f"Created capture of window: ({self.target_window_id}) \"{win32gui.GetWindowText(self.target_window_id)}\""
        )

        # Get window region and initialize base class
        region = self.get_window_geometry()
        super().__init__(
            region=region,
            processor=processor,
            resize_factor=resize_factor,
            show_fps=show_fps,
            print_fps=print_fps,
            visible=visible
        )

    def get_window_geometry(self):
        # Get client coords
        cl, ct, cr, cb = win32gui.GetClientRect(self.target_window_id)

        # Convert client coords to screen coords
        left, top = win32gui.ClientToScreen(self.target_window_id, (cl, ct))
        right, bottom = win32gui.ClientToScreen(self.target_window_id, (cr, cb))

        return {
            "top": top,
            "left": left,
            "width": right - left,
            "height": bottom - top
        }

    def run(self):
        while True:
            visible = (
                    win32gui.IsWindow(self.target_window_id)
                    and win32gui.IsWindowVisible(self.target_window_id)
                    and not win32gui.IsIconic(self.target_window_id)
            )
            if visible:
                self.monitor = self.get_window_geometry()  # refresh region every frame
                img = np.asarray(self.sct.grab(self.monitor))
                avg_fps = self._calculate_fps()

                # Process the image using the current module
                if (processed_frame := self.processor.process(img)) is not None:
                    # Make a fresh copy before drawing text to avoid stacking
                    img_to_show = processed_frame.copy()
                else:
                    img_to_show = img.copy()

                if self.visible:
                    if self.show_fps:
                        cv2.putText(img_to_show, f"FPS: {self.fps_delayed:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    processed_img = cv2.resize(img_to_show, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                    cv2.imshow("Window Capture", processed_img)

                if self.print_fps:
                    print(f"Average FPS: {avg_fps:.2f}")

            else:
                # Window is minimized/invisible – display black screen
                black_frame = np.zeros((240, 320, 3), dtype=np.uint8)  # Small placeholder
                cv2.putText(black_frame, "Paused...", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                cv2.imshow("Window Capture", black_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF  # Get the key
            self.processor.handle_keypress(key)

            # The window is closed
            if cv2.getWindowProperty("Window Capture", cv2.WND_PROP_VISIBLE) < 1:
                break

        # Close everything
        cv2.destroyAllWindows()
        self.sct.close()


def get_top_level_window(hwnd):
    # Walk up the window hierarchy until you find a top-level window
    # (A top-level window usually has no parent or its parent is the desktop)
    parent = win32gui.GetParent(hwnd)
    while parent:
        hwnd = parent
        parent = win32gui.GetParent(hwnd)
    return hwnd


def list_and_select_window():
    windows = []

    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))

    win32gui.EnumWindows(enum_handler, None)

    print("Select a window:")
    for i, (_, title) in enumerate(windows):
        print(f"{i}: {title}")

    while True:
        try:
            index = int(input("Enter the number of the window to select: "))
            if 0 <= index < len(windows):
                return windows[index][0]  # Return HWND
        except ValueError:
            print("Invalid input, try again.")


def wait_for_click_and_get_window():
    print("Hover over the window you want to capture and left-click...")

    # Wait for left mouse click
    while True:
        state_left = win32api.GetKeyState(win32con.VK_LBUTTON)
        if state_left < 0:
            break
        time.sleep(0.01)

    # Get cursor position at click time
    x, y = win32api.GetCursorPos()

    # Get window under cursor
    hwnd = win32gui.WindowFromPoint((x, y))

    while hwnd:
        class_name = win32gui.GetClassName(hwnd)
        window_title = win32gui.GetWindowText(hwnd)

        if class_name in ("Progman", "WorkerW") or window_title == "FolderView":
            print(f"Ignored system window: {class_name}")
            hwnd = None
        else:
            break

    if hwnd:
        hwnd = get_top_level_window(hwnd)
        window_title = win32gui.GetWindowText(hwnd)
        print(f"Selected window: ({hwnd}) \"{window_title}\"")
        return hwnd
    else:
        print("No window found at click position.")
        return None
