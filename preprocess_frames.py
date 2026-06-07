"""
preprocess_frames.py
---------------------
One-time preprocessing — extracts all MP4 videos to per-frame JPEG files.

RGB frames → JPEG (quality 95, fast load, small files, negligible quality loss)

Directory structure created:
    frames_cache/{domain}/rgb/{instance_id}/frame_{N:04d}.jpg

Behaviour:
    - Checks per-instance whether all frames already exist → skips if complete
    - Safe to re-run — idempotent
    - Processes all frames (not just 32) — preserves full temporal flexibility

Usage:
    python preprocess_frames.py                  # all domains
    python preprocess_frames.py --domain human   # one domain
    python preprocess_frames.py --domain cartoon --workers 4

Disk estimate: ~16 GB for all 3 domains (RGB only)
"""

import argparse
import gc
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import cv2
import imageio.v3 as iio
import pandas as pd

from config import get_default_config, Config


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def instance_is_complete(rgb_dir: Path) -> bool:
    """
    Returns True if the RGB frame directory exists and is non-empty.
    Uses presence of at least one file as proxy — avoids re-listing thousands of files.
    """
    if not rgb_dir.exists():
        return False
    return any(rgb_dir.iterdir())


# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def extract_rgb(mp4_path: Path, out_dir: Path, jpeg_quality: int = 95) -> int:
    """
    Extracts all frames from an RGB mp4 as JPEG files.
    Returns number of frames extracted.
    """
    ensure_dir(out_dir)
    vid = iio.imread(str(mp4_path), plugin="pyav")   # (T, H, W, C) uint8
    for i, frame in enumerate(vid):
        out_path = out_dir / f"frame_{i:04d}.jpg"
        if not out_path.exists():
            # frame is (H, W, C) uint8 RGB — cv2 expects BGR
            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
    return len(vid)


# ============================================================
# PER-INSTANCE EXTRACTION
# ============================================================

def process_instance(
    instance_id: str,
    video_dir:   Path,
    rgb_out:     Path,
    jpeg_quality: int = 95,
) -> dict:
    """
    Extracts all frames for one instance (RGB only).
    Returns a result dict for logging.
    """
    t0 = time.perf_counter()

    video_path = video_dir / f"{instance_id}.mp4"

    # Check source file exists
    if not video_path.exists():
        return {"id": instance_id, "status": "skipped_missing_source",
                "elapsed": 0.0, "frames": 0}

    # Check if already fully extracted
    if instance_is_complete(rgb_out):
        return {"id": instance_id, "status": "skipped_cached",
                "elapsed": 0.0, "frames": 0}

    try:
        n_rgb = extract_rgb(video_path, rgb_out, jpeg_quality)

        elapsed = time.perf_counter() - t0
        return {"id": instance_id, "status": "done",
                "elapsed": elapsed, "frames": n_rgb}
    except Exception as e:
        return {"id": instance_id, "status": f"error: {e}",
                "elapsed": time.perf_counter() - t0, "frames": 0}


# ============================================================
# MAIN PER-DOMAIN LOGIC
# ============================================================

def preprocess_domain(domain: str, cfg: Config, num_workers: int = 4):
    print(f"\n{'='*55}")
    print(f"  Preprocessing: {domain.upper()} (RGB only)")
    print(f"{'='*55}")

    video_dir = Path(cfg.paths.video_dir.format(
        domain=domain, Domain=domain.capitalize()
    ))
    base_out  = Path(cfg.paths.frames_cache_dir) / domain

    # Collect all instance IDs from both train and test CSVs
    instance_ids = set()
    for csv_attr in ["train_csv", "test_csv"]:
        csv_path = Path(
            getattr(cfg.paths, csv_attr).format(
                domain=domain, Domain=domain.capitalize()
            )
        )
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                instance_ids.add(Path(str(row.iloc[0])).stem)

    instance_ids = sorted(instance_ids)
    print(f"  Instances to process: {len(instance_ids)}")
    print(f"  Output root: {base_out}")
    print(f"  Workers: {num_workers}")

    total_start = time.perf_counter()
    done = skipped_cached = skipped_missing = errors = 0
    times = []

    def _job(iid):
        return process_instance(
            instance_id  = iid,
            video_dir    = video_dir,
            rgb_out      = base_out / "rgb" / iid,
            jpeg_quality = 95,
        )

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_job, iid): iid for iid in instance_ids}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            status = result["status"]

            if status == "done":
                done += 1
                times.append(result["elapsed"])
                avg = sum(times) / len(times)
                print(
                    f"  [{i:04d}/{len(instance_ids)}] {result['id']:30s} "
                    f"{result['frames']} frames  "
                    f"{result['elapsed']:.1f}s  avg={avg:.1f}s"
                )
            elif status == "skipped_cached":
                skipped_cached += 1
            elif status == "skipped_missing_source":
                skipped_missing += 1
                print(f"  [WARN] Missing source video: {result['id']}")
            else:
                errors += 1
                print(f"  [ERROR] {result['id']}: {status}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  Done:             {done}")
    print(f"  Skipped (cached): {skipped_cached}")
    print(f"  Skipped (missing):{skipped_missing}")
    print(f"  Errors:           {errors}")
    print(f"  Total time:       {total_elapsed/60:.1f} min")
    if times:
        print(f"  Avg per instance: {sum(times)/len(times):.1f}s")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-time frame extraction from MP4 to JPEG (RGB only)"
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        help="Domain to process (human|animal|cartoon). Omit for all."
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel extraction threads (default: 4)"
    )
    args = parser.parse_args()

    cfg     = get_default_config()
    domains = [args.domain] if args.domain else cfg.domains.all_domains

    print(f"\nRGB frame extraction — output: {cfg.paths.frames_cache_dir}")
    print(f"Domains: {domains}")

    for domain in domains:
        preprocess_domain(domain, cfg, num_workers=args.workers)

    print("\n[DONE] All domains processed.")
    print("Update config.py: set use_frame_cache=True or point dataloader at frames_cache_dir.")