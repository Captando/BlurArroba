import cv2
import easyocr


class AtDetector:
    def __init__(self, langs=("en", "pt"), gpu=False):
        self.reader = easyocr.Reader(list(langs), gpu=gpu)

    def detect(self, frame, scale=1.0, min_conf=0.30, marker="@"):
        img = frame
        if scale != 1.0:
            img = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        boxes = []
        for poly, text, conf in self.reader.readtext(img):
            if conf < min_conf or marker not in text:
                continue
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            if scale != 1.0:
                inv = 1.0 / scale
                x1, y1, x2, y2 = x1 * inv, y1 * inv, x2 * inv, y2 * inv
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes
