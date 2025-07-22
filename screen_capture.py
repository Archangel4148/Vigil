import time

import cv2
import numpy as np
from mss import mss


class MonitorCapture:
    def __init__(self, monitor_number=1, resize_factor=0.5, show_fps=True, print_fps=False, visible=True):
        self.monitor_number = monitor_number
        self.resize_factor = resize_factor
        self.show_fps = show_fps
        self.print_fps = print_fps
        self.visible = visible
        self.sct = mss()
        self.monitor = self._get_monitor_region()
        self.prev_time = time.perf_counter()
        self.fps_list = [0.0] * 10
        self.fps_display_delay = 0
        self.fps_delayed = 0

    def _get_monitor_region(self):
        mon = self.sct.monitors[self.monitor_number]
        return {
            'top': mon['top'],
            'left': mon['left'],
            'width': mon['width'],
            'height': mon['height'],
            'mon': self.monitor_number
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

            if self.visible:
                if self.show_fps:
                    # Draw FPS on screen (costs performance)
                    cv2.putText(img, f"FPS: {self.fps_delayed:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Resize and display (costs performance)
                processed_img = cv2.resize(img, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
                cv2.imshow('Monitor Capture', processed_img)

            if self.print_fps:
                # Print to terminal (costs performance)
                print(f"Average FPS: {avg_fps:.2f}")

            # Quit on 'q'
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    mc = MonitorCapture(
        monitor_number=2,
        resize_factor=0.5,
        show_fps=True,
        print_fps=False,
        visible=True
    )
    mc.run()
