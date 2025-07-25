import cv2
import numpy as np
from pytesseract import pytesseract

pytesseract.tesseract_cmd = "E:\\Tesseract\\tesseract.exe"


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


class ImageFinder(BaseCaptureProcessor):
    def __init__(self, template_img, min_matches=15):
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_matches = min_matches

        self.template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
        self.kp_template, self.des_template = self.orb.detectAndCompute(self.template_gray, None)

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.template_gray.shape
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)

        if des_frame is None or len(kp_frame) < self.min_matches:
            return frame

        matches = self.matcher.knnMatch(self.des_template, des_frame, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        # matches = sorted(matches, key=lambda x: x.distance)

        if len(good_matches) >= self.min_matches:
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if matrix is not None and mask is not None:
                mask = mask.ravel().tolist()
                inliers = [i for i, v in enumerate(mask) if v]

                if len(inliers) > 3:
                    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, matrix)
                    cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3)

        return frame


class DigitRegionReader(BaseCaptureProcessor):
    def __init__(self, region: tuple, ocr_threshold=150, show_text=True):
        """
        Args:
            region: (x1, y1, x2, y2) tuple defining the region to crop from each frame.
            ocr_threshold: threshold value for binarization (default 150).
            show_text: whether to overlay detected Y value on the frame.
        """
        self.region = region
        self.ocr_threshold = ocr_threshold
        self.show_text = show_text

    def process(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self.region
        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        config = (
            "--oem 1 "
            "--psm 7 "
            "-c tessedit_char_whitelist=0123456789.- "
            "-c load_system_dawg=0 "
            "-c load_freq_dawg=0"
        )
        text = pytesseract.image_to_string(binary, config=config).strip()

        try:
            value = float(text)
            rect_color = (0, 255, 0)  # Green for valid parse
            display_text = f"Value: {value}"
        except ValueError:
            rect_color = (0, 0, 255)  # Red for invalid/no parse
            display_text = "Value: --"

        # Draw rectangle around OCR region
        cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)

        # Draw parsed value below the rectangle
        if self.show_text:
            text_pos = (x1, y2 + 25)  # 25 pixels below bottom-left corner of rectangle
            cv2.putText(frame, display_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, rect_color, 2)

        return frame
