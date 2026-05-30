import os
import subprocess
import tempfile

import cv2


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def _merge(active, new_boxes, hold, iou_thr=0.3):
    for nb in new_boxes:
        matched = False
        for e in active:
            if _iou(e["box"], nb) > iou_thr:
                e["box"] = nb
                e["ttl"] = hold
                matched = True
                break
        if not matched:
            active.append({"box": nb, "ttl": hold})


def _pad(box, w, h, ratio=0.18, minpx=8):
    x1, y1, x2, y2 = box
    px = max(minpx, int((x2 - x1) * ratio))
    py = max(minpx, int((y2 - y1) * ratio))
    return (
        max(0, x1 - px),
        max(0, y1 - py),
        min(w, x2 + px),
        min(h, y2 + py),
    )


def _apply(frame, box, mode, strength):
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    if mode == "gaussian":
        k = max(3, strength) | 1
        out = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        blocks = max(2, strength)
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        out = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = out


def process_video(
    src_path,
    out_path,
    detector,
    mode="pixelate",
    strength=14,
    sample_interval=None,
    detect_scale=1.0,
    min_conf=0.30,
    progress=None,
):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if sample_interval is None:
        sample_interval = max(1, int(fps / 4))
    hold = sample_interval * 2

    tmp_video = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(tmp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    active = []
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % sample_interval == 0:
                found = detector.detect(frame, scale=detect_scale, min_conf=min_conf)
                _merge(active, found, hold)

            for e in active:
                e["ttl"] -= 1
            active = [e for e in active if e["ttl"] > 0]

            for e in active:
                _apply(frame, _pad(e["box"], w, h), mode, strength)

            writer.write(frame)
            idx += 1
            if progress:
                progress(idx, total)
    finally:
        cap.release()
        writer.release()

    _remux(tmp_video, src_path, out_path)
    if os.path.exists(tmp_video):
        os.remove(tmp_video)
    return out_path


def _remux(processed_video, src_with_audio, out_path):
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", src_with_audio,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
