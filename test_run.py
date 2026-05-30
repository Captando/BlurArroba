"""Teste direto: roda o pipeline de blur de @ sobre o vídeo da pasta."""
import sys
import time

from detector import AtDetector
from processor import process_video

import os

SRC = sys.argv[1] if len(sys.argv) > 1 else "WhatsApp Video 2026-05-30 at 3.22.07 PM.mp4"
OUT = "out_blurred_gpu.mp4"
USE_GPU = os.environ.get("USE_GPU", "1") == "1"


def main():
    import torch
    print(f"      torch={torch.__version__} cuda_disponivel={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name(0)}")

    print(f"[1/3] Carregando detector (EasyOCR, gpu={USE_GPU})...")
    t0 = time.time()
    detector = AtDetector(langs=("en", "pt"), gpu=USE_GPU)
    print(f"      ok em {time.time() - t0:.1f}s")

    print(f"[2/3] Processando {SRC} ...")

    def on_progress(done, total):
        if total and done % 30 == 0:
            print(f"      frame {done}/{total} ({done / total * 100:.0f}%)")

    t0 = time.time()
    process_video(
        SRC,
        OUT,
        detector,
        mode="pixelate",
        strength=14,
        min_conf=0.30,
        detect_scale=1.0,
        progress=on_progress,
    )
    print(f"[3/3] Concluido em {time.time() - t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
