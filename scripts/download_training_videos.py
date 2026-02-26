"""
Download training (and optional demo) videos from a curated list.

Reads config/video_sources.yaml (and optionally config/github_sources.yaml): for each exercise,
downloads URLs (YouTube via yt-dlp, or direct HTTP/HTTPS) into data/downloaded_videos/<exercise>/
as <exercise>_<quality>_<index>.mp4 so that scripts/video_to_skeleton.py can process them.

When YouTube is not available, add direct video URLs in video_sources.yaml or in
config/github_sources.yaml under direct_downloads (see docs/TRAINING_VIDEOS.md).

Requires: pip install yt-dlp  (for YouTube). Direct HTTP uses stdlib only.
"""

import os
import sys
import argparse
import re
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import yaml
except ImportError:
    print("[Error] PyYAML required. Install with: pip install pyyaml")
    sys.exit(1)


def load_sources(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_youtube(url: str) -> bool:
    return "youtube.com" in url or "youtu.be" in url


def safe_filename(exercise: str, quality: str, index: int) -> str:
    return f"{exercise}_{quality}_{index:02d}.mp4"


def download_direct(url: str, out_path: str, quiet: bool = False) -> bool:
    """Download a direct HTTP/HTTPS URL to out_path."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; RehabDownload/1.0)"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        with open(out_path, "wb") as f:
            f.write(data)
        return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
    except Exception as e:
        if not quiet:
            print(f"  Failed: {e}")
        return False


def download_one(url: str, out_path: str, quiet: bool = False) -> bool:
    url = (url or "").strip()
    if not url:
        return False
    if is_youtube(url):
        if yt_dlp is None:
            print("  [Error] yt-dlp required for YouTube. Install with: pip install yt-dlp")
            return False
        # Single-file format so download works without ffmpeg (22=720p, 18=360p mp4)
        opts = {
            "outtmpl": out_path,
            "format": "22/18/best[ext=mp4]/best",
            "quiet": quiet,
            "no_warnings": quiet,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            return os.path.isfile(out_path)
        except Exception as e:
            if not quiet:
                print(f"  Failed: {e}")
            return False
    # Direct HTTP/HTTPS (e.g. Zenodo, GitHub release asset, any .mp4/.avi/.webm link)
    return download_direct(url, out_path, quiet)


def main():
    parser = argparse.ArgumentParser(description="Download training videos from config/video_sources.yaml (and optionally GitHub/direct sources)")
    parser.add_argument("--config", type=str, default="config/video_sources.yaml")
    parser.add_argument("--also-github", type=str, default="", help="Also load URLs from this config (e.g. config/github_sources.yaml) direct_downloads section")
    parser.add_argument("--output", type=str, default="data/downloaded_videos")
    parser.add_argument("--exercises", type=str, nargs="*", help="Only these exercise keys (default: all with URLs)")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"[Error] Config not found: {args.config}")
        sys.exit(1)

    data = load_sources(args.config)
    exercises = dict(data.get("exercises") or {})
    # Merge direct_downloads from GitHub/direct config when videos are not on YouTube
    if args.also_github and os.path.isfile(args.also_github):
        gh = load_sources(args.also_github)
        direct = gh.get("direct_downloads") or {}
        for ex, cfg in direct.items():
            if not isinstance(cfg, dict):
                continue
            urls = cfg.get("urls") or []
            if not urls:
                continue
            if ex not in exercises:
                exercises[ex] = {"quality": (cfg.get("quality") or "good").lower(), "urls": []}
            for u in urls:
                if isinstance(u, dict):
                    u = u.get("url") or ""
                if u and u not in (exercises[ex].get("urls") or []):
                    exercises[ex].setdefault("urls", []).append(u)
        if direct and not args.quiet:
            print(f"[Info] Merged direct_downloads from {args.also_github}")
    if args.exercises:
        exercises = {k: v for k, v in exercises.items() if k in args.exercises}

    total = 0
    for exercise, cfg in exercises.items():
        if not isinstance(cfg, dict):
            continue
        urls = cfg.get("urls") or []
        quality = (cfg.get("quality") or "good").lower()
        if quality not in ("good", "moderate", "poor"):
            quality = "good"
        if not urls:
            continue
        out_dir = os.path.join(args.output, exercise)
        os.makedirs(out_dir, exist_ok=True)
        existing = [f for f in os.listdir(out_dir) if f.endswith(".mp4")]
        # Next index from existing files named <ex>_<quality>_NN.mp4
        indices = []
        for f in existing:
            m = re.match(rf"^{re.escape(exercise)}_(\w+)_(\d+)\.mp4$", f)
            if m:
                indices.append(int(m.group(2)))
        start_idx = max(indices, default=0) + 1
        for i, url in enumerate(urls):
            url = (url or "").strip()
            if not url:
                continue
            out_name = safe_filename(exercise, quality, start_idx + i)
            out_path = os.path.join(out_dir, out_name)
            if os.path.isfile(out_path):
                print(f"Skip (exists): {exercise}/{out_name}")
                total += 1
                continue
            print(f"Downloading: {exercise} <- {url[:60]}...")
            if download_one(url, out_path, quiet=args.quiet):
                print(f"  -> {exercise}/{out_name}")
                total += 1
            else:
                print(f"  -> failed")

    print(f"\nDone. {total} videos in {args.output}")
    if total > 0:
        print("Next: python scripts/video_to_skeleton.py --input data/downloaded_videos --output data/custom")
        print("Then: python train.py --dataset custom")

if __name__ == "__main__":
    main()
