#!/usr/bin/env python3
"""Convert a recorded MP4 to a compact GIF with a two-pass FFmpeg palette.

This only converts existing footage; it does not create or validate a rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing recorded MP4")
    parser.add_argument("output", type=Path, help="Destination .gif")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-width", type=int, default=800)
    parser.add_argument("--colors", type=int, default=128, help="Palette size, 4–256")
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, help="Optional clip duration in seconds")
    parser.add_argument("--force", action="store_true", help="Replace an existing GIF")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Input does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files")
    if args.output.suffix.lower() != ".gif":
        parser.error("Output must have a .gif extension")
    if args.output.exists() and not args.force:
        parser.error(f"Output exists; use --force to replace it: {args.output}")
    if not 1 <= args.fps <= 50 or not 32 <= args.max_width <= 1920:
        parser.error("Use fps 1–50 and max-width 32–1920")
    if not 4 <= args.colors <= 256 or args.start < 0:
        parser.error("Use colors 4–256 and nonnegative start")
    if args.duration is not None and args.duration <= 0:
        parser.error("Duration must be positive")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("FFmpeg is required but was not found in PATH")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    video_input = ["-ss", str(args.start), "-i", str(args.input.resolve())]
    duration = [] if args.duration is None else ["-t", str(args.duration)]
    # Quotes protect the expression's comma in FFmpeg; no shell is involved.
    scale = f"fps={args.fps},scale=w='min({args.max_width},iw)':h=-2:flags=lanczos"

    try:
        with tempfile.TemporaryDirectory(prefix="q1-gif-") as directory:
            palette = str(Path(directory) / "palette.png")
            subprocess.run(
                base + video_input + duration
                + ["-vf", f"{scale},palettegen=max_colors={args.colors}:stats_mode=diff",
                   "-frames:v", "1", "-threads", "1", palette],
                check=True,
            )
            subprocess.run(
                base + video_input + ["-i", palette] + duration
                + ["-filter_complex",
                   f"[0:v]{scale}[clip];[clip][1:v]paletteuse=dither=sierra2_4a:diff_mode=rectangle",
                   "-an", "-loop", "0", "-gifflags", "+transdiff",
                   str(args.output.resolve())],
                check=True,
            )
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"FFmpeg conversion failed with exit code {error.returncode}") from error

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"GIF: {args.output} ({size_mb:.2f} MiB, {args.fps} FPS, max width {args.max_width}px)")


if __name__ == "__main__":
    main()
