"""
Skeleton-Based Rehabilitation Exercise Quality Grading

Layout: Header | Section 1 (Demo video) | Section 2 (Step images/poses) | Section 3 (Live camera) | Footer
Includes: repetition count, attention-based live feedback (text + audio TTS).
"""

import os
import sys
import argparse
import time
import warnings
import webbrowser
import threading

# Reduce console noise: set before importing TensorFlow, Keras, MediaPipe
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=no INFO, 2=no INFO/WARNING
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom ops message
os.environ["GLOG_minloglevel"] = "2"       # MediaPipe/C++: 2=ERROR only (no INFO/WARNING)

# Suppress known third-party warnings (pygame pkg_resources, tf.placeholder, etc.)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="(tensorflow|keras).*")
warnings.filterwarnings("ignore", message=".*placeholder.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# TensorFlow/Keras Python logging (placeholder deprecation, etc.)
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    logging.getLogger("keras").setLevel(logging.ERROR)
except Exception:
    pass

# Redirect only stderr so C++/TF/MediaPipe warnings are hidden (stdout left so pygame print works on Windows)
def _silent_stdio():
    """Redirect stderr to devnull. Returns (saved_stdout, saved_stderr, stdout_fd, stderr_fd) to restore later."""
    stderr_fd = sys.stderr.fileno()
    stdout_fd = sys.stdout.fileno()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_stderr = os.dup(stderr_fd)
    saved_stdout = os.dup(stdout_fd)
    os.dup2(devnull, stderr_fd)  # only stderr -> devnull (stdout unchanged for pygame)
    os.close(devnull)
    return saved_stdout, saved_stderr, stdout_fd, stderr_fd

def _restore_stdio(saved_stdout, saved_stderr, stdout_fd, stderr_fd):
    os.dup2(saved_stderr, stderr_fd)
    os.close(saved_stdout)
    os.close(saved_stderr)

_STDIO_SAVED = _silent_stdio()
try:
    import cv2
    import numpy as np
    import pygame
    from config import load_config
except Exception:
    _restore_stdio(*_STDIO_SAVED)
    raise

# Fixed layout - no cut-off, clear proportions
WIN_W = 960
WIN_H = 720
HEADER_H = 64
FOOTER_H = 124
VIDEO_Y = HEADER_H
VIDEO_H = WIN_H - HEADER_H - FOOTER_H
PAD = 24
# Three sections: 1=Demo video, 2=Step images/poses, 3=Live camera
DEMO_FRAC = 0.28   # fraction of content width for demo
STEPS_FRAC = 0.28  # fraction for step-by-step images
LIVE_FRAC = 0.44   # fraction for live camera (largest)
DEMO_DIR = "data/demos"
STEP_IMAGES_DIR = "data/step_images"  # per-exercise folders: step_01.jpg, step_02.png, etc.
DEMO_EXTENSIONS = (".mp4", ".webm", ".avi")
DEMO_YT_MAX_DURATION_SEC = 60  # download only first 60s for in-app playback
# Show only the exercise clip: skip intro, then loop this duration
DEMO_SKIP_INTRO_SEC = 5   # default skip (intro/title)
DEMO_CLIP_DURATION_SEC = 45  # default clip length to loop
DEMO_FLIP_FOR_TEXT = True  # flip demo frame so on-video text is not mirrored
# Per-exercise (skip_sec, clip_duration_sec) for videos with long intros (~5 min). Add keys only for those.
DEMO_EXERCISE_CLIP = {
    "shoulder_rotation": (300, 120),   # skip 5 min intro, then 2 min exercise loop
    # "deep_squat": (300, 90), "sit_to_stand": (300, 90),  # add if those demos have long intros
}
def get_demo_clip(exercise_key: str):
    """Return (skip_sec, clip_duration_sec) for demo playback; uses default if not in DEMO_EXERCISE_CLIP."""
    return DEMO_EXERCISE_CLIP.get(exercise_key, (DEMO_SKIP_INTRO_SEC, DEMO_CLIP_DURATION_SEC))


# ----- Audio feedback (TTS): speak feedback messages instead of text-only -----
_tts_last_msg = None
_tts_last_time = 0.0
TTS_COOLDOWN_SEC = 2.2  # min seconds between speaking the same message again

def _speak_worker(text: str):
    """Run in thread: speak text via pyttsx3 (offline TTS)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass

def speak_feedback(text: str):
    """Queue feedback for audio output (TTS). Throttled by message and cooldown."""
    global _tts_last_msg, _tts_last_time
    if not text or not text.strip():
        return
    now = time.time()
    if text.strip() == _tts_last_msg and (now - _tts_last_time) < TTS_COOLDOWN_SEC:
        return
    _tts_last_msg = text.strip()
    _tts_last_time = now
    t = threading.Thread(target=_speak_worker, args=(text.strip(),), daemon=True)
    t.start()

# 15 exercises from UI-PRMD, KIMORE + 8 custom (novel) exercises
EXERCISE_NAMES = {
    "deep_squat": "Deep squat",
    "hurdle_step": "Hurdle step",
    "inline_lunge": "Inline lunge",
    "side_lunge": "Side lunge",
    "sit_to_stand": "Sit to stand",
    "standing_leg_raise": "Standing leg raise",
    "shoulder_abduction": "Shoulder abduction",
    "shoulder_extension": "Shoulder extension",
    "shoulder_rotation": "Shoulder rotation",
    "shoulder_scaption": "Shoulder scaption",
    "hip_abduction": "Hip abduction",
    "trunk_rotation": "Trunk rotation",
    "squat": "Squat",
    "leg_raise": "Leg raise",
    "reach_and_retrieve": "Reach and retrieve",
    # Custom novel exercises (not in UI-PRMD, KIMORE, NTU)
    "wall_pushup": "Wall push-up",
    "heel_raise": "Heel raise",
    "bird_dog": "Bird dog",
    "glute_bridge": "Glute bridge",
    "clamshell": "Clamshell",
    "chin_tuck": "Chin tuck",
    "marching_in_place": "Marching in place",
    "step_up": "Step-up",
}
# Dropdown order: sorted by display name (ascending)
EXERCISE_ORDER = sorted(EXERCISE_NAMES.keys(), key=lambda ex: EXERCISE_NAMES.get(ex, ex))
# Exercises shown in UI dropdown (exclude e.g. heel_raise)
DROPDOWN_EXERCISES = [ex for ex in EXERCISE_ORDER if ex != "heel_raise"]

# Optional: YouTube demo/tutorial URL per exercise. Replace with real links; empty = no button.
# Example: "deep_squat": "https://www.youtube.com/watch?v=..."
EXERCISE_YOUTUBE = {
    "deep_squat": "https://www.youtube.com/results?search_query=deep+squat+rehabilitation+exercise",
    "hurdle_step": "https://www.youtube.com/results?search_query=hurdle+step+rehabilitation",
    "inline_lunge": "https://www.youtube.com/results?search_query=inline+lunge+rehab",
    "side_lunge": "https://www.youtube.com/results?search_query=side+lunge+exercise",
    "sit_to_stand": "https://www.youtube.com/results?search_query=sit+to+stand+exercise",
    "standing_leg_raise": "https://www.youtube.com/results?search_query=standing+leg+raise",
    "shoulder_abduction": "https://www.youtube.com/results?search_query=shoulder+abduction+exercise",
    "shoulder_extension": "https://www.youtube.com/results?search_query=shoulder+extension+rehab",
    "shoulder_rotation": "https://www.youtube.com/results?search_query=shoulder+rotation+exercise",
    "shoulder_scaption": "https://www.youtube.com/results?search_query=shoulder+scaption",
    "hip_abduction": "https://www.youtube.com/results?search_query=hip+abduction+exercise",
    "trunk_rotation": "https://www.youtube.com/results?search_query=trunk+rotation+exercise",
    "squat": "https://www.youtube.com/results?search_query=squat+rehabilitation",
    "leg_raise": "https://www.youtube.com/results?search_query=leg+raise+exercise",
    "reach_and_retrieve": "https://www.youtube.com/results?search_query=reach+and+retrieve+exercise",
    "wall_pushup": "https://www.youtube.com/results?search_query=wall+push+up+rehab",
    "heel_raise": "https://www.youtube.com/results?search_query=heel+raise+exercise",
    "bird_dog": "https://www.youtube.com/results?search_query=bird+dog+exercise",
    "glute_bridge": "https://www.youtube.com/results?search_query=glute+bridge+exercise",
    "clamshell": "https://www.youtube.com/results?search_query=clamshell+exercise",
    "chin_tuck": "https://www.youtube.com/results?search_query=chin+tuck+exercise",
    "marching_in_place": "https://www.youtube.com/results?search_query=marching+in+place+rehab",
    "step_up": "https://www.youtube.com/results?search_query=step+up+exercise",
}

# Clean color scheme
BG = (22, 26, 32)
HEADER_BG = (32, 38, 48)
FOOTER_BG = (32, 38, 48)
PANEL_BG = (42, 50, 62)
GREEN = (76, 217, 100)
AMBER = (255, 193, 7)
RED = (255, 82, 82)
WHITE = (255, 255, 255)
GRAY = (170, 180, 190)
DARK = (55, 65, 80)

DD_W = 260
DD_ITEM_H = 26
DD_VISIBLE_ITEMS = 10   # max items visible in dropdown; rest via scrollbar
DD_LIST_H = DD_VISIBLE_ITEMS * DD_ITEM_H + 12
DD_SB_W = 14            # scrollbar width

# Only run quality prediction when pose is clearly moving (full sequence + recent window)
# Balanced: high enough to avoid idle/jitter, low enough so real exercise passes
MOTION_THRESHOLD = 0.018
RECENT_MOTION_FRAMES = 10
# When model has joint attention: gate quality/reps by motion in attended body parts (graph-based pose)
ATTENTION_MOTION_THRESHOLD = 0.002


def get_demo_path(exercise: str):
    """Return path to demo video in data/demos (populate via scripts/download_kaggle_demos.py)."""
    base_local = os.path.join(DEMO_DIR, exercise)
    for ext in DEMO_EXTENSIONS:
        p = base_local + ext
        if os.path.isfile(p):
            return p
    return None


def get_step_image_paths(exercise: str):
    """Return sorted list of paths to step images for exercise: data/step_images/<exercise>/step_01.jpg, etc."""
    import glob
    folder = os.path.join(STEP_IMAGES_DIR, exercise)
    if not os.path.isdir(folder):
        return []
    paths = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        paths.extend(glob.glob(os.path.join(folder, "*" + ext)))
    # Sort by filename so step_01, step_02, ... or 1.jpg, 2.png order
    paths.sort(key=lambda p: (os.path.basename(p).lower(), p))
    return paths


def _is_youtube_url(url: str) -> bool:
    return url and ("youtube.com/watch" in url or "youtu.be/" in url)


def _get_youtube_demo_url(exercise: str) -> str:
    """Return a direct YouTube watch URL for demo (from video_sources.yaml)."""
    try:
        import yaml
        vs_path = os.path.join(os.path.dirname(__file__), "config", "video_sources.yaml")
        if os.path.isfile(vs_path):
            with open(vs_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for ex, cfg in (data.get("exercises") or {}).items():
                if ex != exercise or not isinstance(cfg, dict):
                    continue
                for u in (cfg.get("urls") or []):
                    u = (u or "").strip()
                    if _is_youtube_url(u):
                        return u
    except Exception:
        pass
    return ""


def _download_yt_demo_to_cache(exercise: str, url: str, cache_path: str, max_sec: int, result_list: list):
    """Run in thread: download first max_sec of YouTube video to cache_path. Appends (exercise, success)."""
    try:
        import yt_dlp
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # Single-file format so download works without ffmpeg; skip -t truncation if no ffmpeg
        opts = {
            "outtmpl": cache_path,
            "format": "22/18/best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=2)
            opts["postprocessor_args"] = {"ffmpeg": ["-t", str(max_sec)]}
        except Exception:
            pass
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        ok = os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0
        result_list.append((exercise, ok))
    except Exception:
        result_list.append((exercise, False))


def get_font(size, bold=False):
    for name in ["Segoe UI", "Arial"]:
        try:
            return pygame.font.SysFont(name, size, bold=bold)
        except Exception:
            pass
    return pygame.font.Font(None, size)


def truncate(font, text, max_w):
    if font.size(text)[0] <= max_w:
        return text
    while text and font.size(text + "…")[0] > max_w:
        text = text[:-1]
    return text + "…" if text else ""


def scale_fit(surface, dest_w, dest_h):
    """Scale surface to fit inside dest_w x dest_h, maintain aspect ratio."""
    sw, sh = surface.get_size()
    if sw <= 0 or sh <= 0:
        return surface
    scale = min(dest_w / sw, dest_h / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    return pygame.transform.smoothscale(surface, (nw, nh))


def draw_header(screen, score, exercise, dd_open, dd_scroll, font_m, font_s, w, h):
    pygame.draw.rect(screen, HEADER_BG, (0, 0, w, HEADER_H))
    pygame.draw.line(screen, DARK, (0, HEADER_H), (w, HEADER_H), 1)

    # Score block (show "—" when no active assessment, i.e. score decayed to idle)
    sx, sy = PAD, 10
    lbl = font_s.render("QUALITY", True, GRAY)
    if score < 0.05:
        pct = font_m.render("—", True, GRAY)
        st, sc = "Perform exercise", GRAY
    else:
        pct = font_m.render(f"{int(score * 100)}%", True, WHITE)
        if score >= 0.75:
            st, sc = "GOOD", GREEN
        elif score >= 0.45:
            st, sc = "MODERATE", AMBER
        else:
            st, sc = "INCORRECT", RED
    status = font_s.render(st, True, sc)
    screen.blit(lbl, (sx, sy))
    screen.blit(pct, (sx, sy + 18))
    screen.blit(status, (sx, sy + 38))

    # Exercise dropdown
    dx = w - DD_W - PAD
    dy = 12
    dh = 40
    pygame.draw.rect(screen, PANEL_BG, (dx, dy, DD_W, dh))
    pygame.draw.rect(screen, DARK, (dx, dy, DD_W, dh), 1)
    name = truncate(font_s, EXERCISE_NAMES.get(exercise, exercise), DD_W - 44)
    txt = font_s.render(name, True, WHITE)
    screen.blit(txt, (dx + 12, dy + (dh - txt.get_height()) // 2))
    arr = font_s.render("▾" if dd_open else "▸", True, GRAY)
    screen.blit(arr, (dx + DD_W - arr.get_width() - 10, dy + (dh - arr.get_height()) // 2))

    if dd_open:
        ex_list = DROPDOWN_EXERCISES
        n = len(ex_list)
        dly = dy + dh + 4
        pygame.draw.rect(screen, PANEL_BG, (dx, dly, DD_W, DD_LIST_H))
        pygame.draw.rect(screen, DARK, (dx, dly, DD_W, DD_LIST_H), 1)
        # Visible slice of items
        list_w = DD_W - DD_SB_W - 24
        for vi in range(DD_VISIBLE_ITEMS):
            i = dd_scroll + vi
            if i >= n:
                break
            ex = ex_list[i]
            iy = dly + 6 + vi * DD_ITEM_H
            lab = truncate(font_s, EXERCISE_NAMES[ex], list_w)
            col = GREEN if ex == exercise else WHITE
            it = font_s.render(lab, True, col)
            screen.blit(it, (dx + 12, iy))
        # Vertical scrollbar
        sb_x = dx + DD_W - DD_SB_W
        pygame.draw.rect(screen, DARK, (sb_x, dly, DD_SB_W, DD_LIST_H))
        if n > DD_VISIBLE_ITEMS:
            thumb_frac = DD_VISIBLE_ITEMS / n
            thumb_h = max(20, int(DD_LIST_H * thumb_frac))
            range_ = n - DD_VISIBLE_ITEMS
            thumb_y = dly + int((dd_scroll / range_) * (DD_LIST_H - thumb_h)) if range_ > 0 else dly
            pygame.draw.rect(screen, GRAY, (sb_x + 2, thumb_y, DD_SB_W - 4, thumb_h))
            pygame.draw.rect(screen, WHITE, (sb_x + 2, thumb_y, DD_SB_W - 4, thumb_h), 1)


def draw_left_panel(screen, left_x, left_y, left_w, left_h, demo_surf, exercise, font_m, font_s, youtube_url, downloading_exercise=None):
    """Draw section 1: reference/demo video."""
    pygame.draw.rect(screen, PANEL_BG, (left_x, left_y, left_w, left_h))
    pygame.draw.rect(screen, DARK, (left_x, left_y, left_w, left_h), 1)
    title = font_s.render("1. Reference / Demo", True, GRAY)
    screen.blit(title, (left_x + 12, left_y + 8))
    ex_name = truncate(font_m, EXERCISE_NAMES.get(exercise, exercise), left_w - 24)
    ex_txt = font_m.render(ex_name, True, WHITE)
    screen.blit(ex_txt, (left_x + 12, left_y + 28))
    content_y = left_y + 52
    content_h = left_h - 52
    if demo_surf is not None:
        scaled = scale_fit(demo_surf, left_w - 16, content_h - 8)
        sx = left_x + (left_w - scaled.get_width()) // 2
        sy = content_y + (content_h - scaled.get_height()) // 2
        screen.blit(scaled, (sx, sy))
    elif downloading_exercise:
        msg = font_m.render("Loading demo…", True, WHITE)
        screen.blit(msg, (left_x + (left_w - msg.get_width()) // 2, content_y + (content_h - msg.get_height()) // 2 - 12))
        sub = font_s.render("Playing YouTube clip in app", True, GRAY)
        screen.blit(sub, (left_x + (left_w - sub.get_width()) // 2, content_y + (content_h - sub.get_height()) // 2 + 16))
    else:
        msg = font_s.render("No demo video for this exercise.", True, GRAY)
        screen.blit(msg, (left_x + 20, content_y + 24))
        hint = font_s.render(f"Add: {DEMO_DIR}/{exercise}.mp4", True, GRAY)
        screen.blit(hint, (left_x + 20, content_y + 48))
        if youtube_url:
            btn_y = content_y + content_h - 44
            btn_rect = pygame.Rect(left_x + 20, btn_y, left_w - 40, 36)
            pygame.draw.rect(screen, (220, 60, 60), btn_rect)
            pygame.draw.rect(screen, DARK, btn_rect, 1)
            yt_txt = font_s.render("Watch on YouTube", True, WHITE)
            screen.blit(yt_txt, (btn_rect.centerx - yt_txt.get_width() // 2, btn_rect.centery - yt_txt.get_height() // 2))
            return btn_rect
    return None


def draw_middle_panel(screen, mid_x, mid_y, mid_w, mid_h, exercise, step_paths, current_step_index, font_m, font_s):
    """Draw the step-by-step images/poses panel. Shows one image at a time or strip; current_step_index cycles 0..len(step_paths)-1."""
    pygame.draw.rect(screen, PANEL_BG, (mid_x, mid_y, mid_w, mid_h))
    pygame.draw.rect(screen, DARK, (mid_x, mid_y, mid_w, mid_h), 1)
    title = font_s.render("2. Steps / Pose reference", True, GRAY)
    screen.blit(title, (mid_x + 12, mid_y + 8))
    ex_name = truncate(font_m, EXERCISE_NAMES.get(exercise, exercise), mid_w - 24)
    ex_txt = font_m.render(ex_name, True, WHITE)
    screen.blit(ex_txt, (mid_x + 12, mid_y + 28))
    content_y = mid_y + 52
    content_h = mid_h - 52
    if step_paths:
        idx = current_step_index % max(1, len(step_paths))
        path = step_paths[idx]
        try:
            img = cv2.imread(path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_rgb = np.rot90(img_rgb)
                step_surf = pygame.surfarray.make_surface(img_rgb)
            else:
                step_surf = None
        except Exception:
            step_surf = None
        if step_surf is not None:
            scaled = scale_fit(step_surf, mid_w - 16, content_h - 28)
            sx = mid_x + (mid_w - scaled.get_width()) // 2
            sy = content_y + (content_h - 28 - scaled.get_height()) // 2
            screen.blit(scaled, (sx, sy))
        step_lbl = font_s.render(f"Step {idx + 1} of {len(step_paths)}", True, GRAY)
        screen.blit(step_lbl, (mid_x + (mid_w - step_lbl.get_width()) // 2, content_y + content_h - 24))
    else:
        msg = font_s.render("No step images for this exercise.", True, GRAY)
        screen.blit(msg, (mid_x + 20, content_y + 24))
        hint = font_s.render(f"Add: {STEP_IMAGES_DIR}/{exercise}/", True, GRAY)
        screen.blit(hint, (mid_x + 20, content_y + 44))
        sub = font_s.render("step_01.jpg, step_02.png, ...", True, GRAY)
        screen.blit(sub, (mid_x + 20, content_y + 62))


def draw_footer(screen, reps, target, feedback, font_m, font_s, w, h):
    fy = h - FOOTER_H
    pygame.draw.rect(screen, FOOTER_BG, (0, fy, w, FOOTER_H))
    pygame.draw.line(screen, DARK, (0, fy), (w, fy), 1)

    fx = PAD
    lbl = font_s.render("Repetitions", True, GRAY)
    val = font_m.render(f"{reps} / {target}", True, GREEN)
    screen.blit(lbl, (fx, fy + 12))
    screen.blit(val, (fx, fy + 32))

    # Progress bar - clearly visible
    bar_x, bar_y = fx, fy + 52
    bar_w, bar_h = w - 2 * PAD, 26
    prog = min(reps / max(1, target), 1.0)
    pygame.draw.rect(screen, DARK, (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.rect(screen, (70, 80, 95), (bar_x + 2, bar_y + 2, bar_w - 4, bar_h - 4))
    if prog > 0:
        fw = int((bar_w - 6) * prog)
        pygame.draw.rect(screen, GREEN, (bar_x + 3, bar_y + 3, fw, bar_h - 6))

    # Feedback text (e.g. "Perform the exercise") - kept well inside footer
    fb = truncate(font_s, feedback, w - 2 * PAD - 180)
    fb_txt = font_s.render(fb, True, WHITE)
    screen.blit(fb_txt, (fx, fy + 84))


def draw_summary(screen, ex, target, done, avg, dur, font_l, font_m, w, h):
    overlay = pygame.Surface((w, h))
    overlay.fill((12, 16, 22))
    overlay.set_alpha(245)
    screen.blit(overlay, (0, 0))
    cw, ch = 420, 380
    cx, cy = (w - cw) // 2, (h - ch) // 2
    pygame.draw.rect(screen, PANEL_BG, (cx, cy, cw, ch))
    pygame.draw.rect(screen, DARK, (cx, cy, cw, ch), 2)
    t = font_l.render("Session Summary", True, WHITE)
    screen.blit(t, ((w - t.get_width()) // 2, cy + 24))
    items = [
        ("Exercise", EXERCISE_NAMES.get(ex, ex)),
        ("Target", str(target)),
        ("Completed", str(done)),
        ("Score", f"{avg:.0f}%"),
        ("Time", f"{dur:.0f} sec"),
    ]
    for i, (k, v) in enumerate(items):
        y = cy + 80 + i * 48
        screen.blit(font_m.render(k + ":", True, GRAY), (cx + 32, y))
        screen.blit(font_m.render(str(v), True, WHITE), (cx + cw - 120, y))
    hnt = font_m.render("SPACE = New  |  ESC = Quit", True, GRAY)
    screen.blit(hnt, ((w - hnt.get_width()) // 2, h - 60))


def _draw_loading_screen(screen, w, h, msg="Loading..."):
    """Show a loading message centered on screen."""
    screen.fill(BG)
    font = get_font(28)
    txt = font.render(msg, True, WHITE)
    screen.blit(txt, ((w - txt.get_width()) // 2, (h - txt.get_height()) // 2))
    pygame.display.flip()


def _draw_loading_ui(screen, w, h, msg, font_m, font_s, left_w, mid_w=0):
    """Draw full UI shell with loading message in the right (camera) panel so layout appears immediately."""
    if mid_w <= 0:
        mid_w = max(280, int(w * STEPS_FRAC))
    screen.fill(BG)
    # Header (no score, no dropdown open)
    pygame.draw.rect(screen, HEADER_BG, (0, 0, w, HEADER_H))
    pygame.draw.line(screen, DARK, (0, HEADER_H), (w, HEADER_H), 1)
    screen.blit(font_s.render("QUALITY", True, GRAY), (PAD, 10))
    screen.blit(font_m.render("—", True, GRAY), (PAD, 28))
    screen.blit(font_s.render("Loading…", True, GRAY), (PAD, 46))
    dx = w - DD_W - PAD
    pygame.draw.rect(screen, PANEL_BG, (dx, 12, DD_W, 40))
    screen.blit(font_s.render("Exercise", True, GRAY), (dx + 12, 22))
    # Panel 1: Demo
    pygame.draw.rect(screen, PANEL_BG, (0, VIDEO_Y, left_w, VIDEO_H))
    screen.blit(font_s.render("1. Reference / Demo", True, GRAY), (12, VIDEO_Y + 8))
    # Panel 2: Step images
    pygame.draw.rect(screen, PANEL_BG, (left_w, VIDEO_Y, mid_w, VIDEO_H))
    screen.blit(font_s.render("2. Steps / Pose", True, GRAY), (left_w + 12, VIDEO_Y + 8))
    # Panel 3: loading message
    rx = left_w + mid_w + (w - left_w - mid_w) // 2 - 120
    ry = VIDEO_Y + VIDEO_H // 2 - 20
    screen.blit(font_m.render(msg, True, WHITE), (rx, ry))
    # Footer
    fy = h - FOOTER_H
    pygame.draw.rect(screen, FOOTER_BG, (0, fy, w, FOOTER_H))
    screen.blit(font_s.render("Repetitions", True, GRAY), (PAD, fy + 12))
    screen.blit(font_m.render("0 / 10", True, GRAY), (PAD, fy + 32))
    pygame.display.flip()


def _load_backend_worker(result_dict, config, model_path, use_simplified):
    """Run in thread: heavy imports, camera, model. Sets result_dict['ready'] and entries or result_dict['error']."""
    try:
        from src.pose_extraction import PoseExtractor
        from src.preprocessing import PoseBuffer, sequence_motion_energy, recent_motion_energy, attention_weighted_motion
        from src.inference import InferenceEngine
        from src.models.st_gcn import build_rehab_grading_model, build_simplified_model
        from src.joint_analysis import detect_errors, get_feedback_from_score
        from src.motion_rep_counter import MotionRepCounter
        from src.repetition_counter import RepetitionCounter
        from src.skeleton_highlight import highlight_joints
        from src.session_manager import save_session, export_for_power_bi
        from src.ui.components import lerp, draw_rounded_rect

        rt = config.get("realtime", {})
        pose_cfg = config.get("pose", {})
        model_cfg = config.get("model", {})
        seq_cfg = config.get("sequence", {})
        cam_id = rt.get("camera_id", 0)
        seq_len = seq_cfg.get("frame_buffer_size", 64)
        num_joints = model_cfg.get("num_joints", 33)

        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            result_dict["error"] = "Could not open webcam."
            return
        cam_w = rt.get("camera_width", 320)
        cam_h = rt.get("camera_height", 240)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        extractor = PoseExtractor(
            min_detection_confidence=pose_cfg.get("min_detection_confidence", 0.5),
            min_tracking_confidence=pose_cfg.get("min_tracking_confidence", 0.5),
            model_complexity=pose_cfg.get("model_complexity", 1),
        )
        buffer = PoseBuffer(maxlen=seq_len)
        if model_path and os.path.exists(model_path):
            engine = InferenceEngine(model_path=model_path)
        else:
            model = build_simplified_model(num_joints=num_joints, in_channels=3, sequence_length=seq_len)
            engine = InferenceEngine(model=model)
        motion_ctr = MotionRepCounter(threshold=0.002, min_peak_distance=8)
        score_ctr = RepetitionCounter(threshold=0.35, min_peak_distance=10, window_size=12)

        result_dict["cap"] = cap
        result_dict["extractor"] = extractor
        result_dict["buffer"] = buffer
        result_dict["engine"] = engine
        result_dict["motion_ctr"] = motion_ctr
        result_dict["score_ctr"] = score_ctr
        result_dict["sequence_motion_energy"] = sequence_motion_energy
        result_dict["recent_motion_energy"] = recent_motion_energy
        result_dict["attention_weighted_motion"] = attention_weighted_motion
        result_dict["detect_errors"] = detect_errors
        result_dict["get_feedback_from_score"] = get_feedback_from_score
        result_dict["highlight_joints"] = highlight_joints
        result_dict["save_session"] = save_session
        result_dict["export_for_power_bi"] = export_for_power_bi
        result_dict["lerp"] = lerp
        result_dict["inference_interval_frames"] = rt.get("inference_interval_frames", 4)
        result_dict["ready"] = True
    except Exception as e:
        result_dict["error"] = str(e)
        result_dict["ready"] = False


def run_app(model_path=None, use_simplified=False, config_path=None):
    config = load_config(config_path)
    rt = config.get("realtime", {})
    pose_cfg = config.get("pose", {})
    model_cfg = config.get("model", {})
    seq_cfg = config.get("sequence", {})

    target_reps = rt.get("default_target_reps", 10)
    cam_id = rt.get("camera_id", 0)
    seq_len = seq_cfg.get("frame_buffer_size", 64)
    num_joints = model_cfg.get("num_joints", 33)

    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    pygame.display.set_caption("Rehab Exercise Quality Grading")
    clock = pygame.time.Clock()
    font_l = get_font(26, bold=True)
    font_m = get_font(20)
    font_s = get_font(16)
    left_w = max(280, int(w * DEMO_FRAC))
    mid_w = max(280, int(w * STEPS_FRAC))

    # Show full UI shell immediately so user sees the app layout
    _draw_loading_ui(screen, w, h, "Loading model & camera…", font_m, font_s, left_w, mid_w)

    load_result = {}
    t = threading.Thread(
        target=_load_backend_worker,
        args=(load_result, config, model_path, use_simplified),
        daemon=True,
    )
    t.start()

    # Wait for backend; keep UI responsive and show loading state
    while not load_result.get("ready") and load_result.get("error") is None:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
        _draw_loading_ui(screen, w, h, "Loading model & camera…", font_m, font_s, left_w, mid_w)
        clock.tick(20)

    if load_result.get("error"):
        _draw_loading_screen(screen, w, h, "Error: " + load_result["error"][:50])
        pygame.display.flip()
        time.sleep(3)
        pygame.quit()
        return

    cap = load_result["cap"]
    extractor = load_result["extractor"]
    buffer = load_result["buffer"]
    engine = load_result["engine"]
    motion_ctr = load_result["motion_ctr"]
    score_ctr = load_result["score_ctr"]
    sequence_motion_energy = load_result["sequence_motion_energy"]
    recent_motion_energy = load_result["recent_motion_energy"]
    attention_weighted_motion_fn = load_result["attention_weighted_motion"]
    inference_interval = load_result.get("inference_interval_frames", 4)
    detect_errors = load_result["detect_errors"]
    get_feedback_from_score = load_result["get_feedback_from_score"]
    highlight_joints = load_result["highlight_joints"]
    save_session = load_result["save_session"]
    export_for_power_bi = load_result["export_for_power_bi"]
    lerp = load_result["lerp"]

    score = display_score = 0.0
    frame_count = 0
    exercise = "deep_squat"
    dd_open = False
    # Demo video: local file, or cached YouTube clip, or background-download from YouTube
    demo_cap = None
    demo_exercise = None
    demo_frame_surf = None
    demo_clip_start_time = None  # real-time start for time-based playback
    downloading_demo = None  # exercise name while downloading
    demo_ready_list = []    # thread appends exercise when download done

    def open_demo_for(ex):
        nonlocal demo_cap, demo_exercise, demo_frame_surf, demo_clip_start_time, downloading_demo
        if demo_cap is not None:
            demo_cap.release()
            demo_cap = None
        demo_exercise = ex
        demo_frame_surf = None
        path = get_demo_path(ex)
        if path:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                demo_cap = cap
                demo_clip_start_time = time.time()
                skip_sec, _ = get_demo_clip(ex)
                demo_cap.set(cv2.CAP_PROP_POS_MSEC, skip_sec * 1000)
            downloading_demo = None
            return
        yt_url = _get_youtube_demo_url(ex)
        if yt_url:
            cache_path = os.path.join(DEMOS_CACHE_DIR, ex + ".mp4")
            if not os.path.isfile(cache_path) or os.path.getsize(cache_path) == 0:
                downloading_demo = ex
                t = threading.Thread(
                    target=_download_yt_demo_to_cache,
                    args=(ex, yt_url, cache_path, DEMO_YT_MAX_DURATION_SEC, demo_ready_list),
                    daemon=True,
                )
                t.start()
            else:
                cap = cv2.VideoCapture(cache_path)
                if cap.isOpened():
                    demo_cap = cap
                    demo_clip_start_time = time.time()
                    skip_sec, _ = get_demo_clip(ex)
                    demo_cap.set(cv2.CAP_PROP_POS_MSEC, skip_sec * 1000)
        else:
            downloading_demo = None
    open_demo_for(exercise)
    step_paths = get_step_image_paths(exercise)
    step_cycle_start = time.time()  # for cycling step images every few seconds
    youtube_btn_rect = None  # set each frame by draw_left_panel when no demo + has URL
    dd_scroll = 0           # scroll offset for dropdown (0 = top)
    session_on = False
    session_scores = []
    session_fb = []
    show_summary = False
    summary_data = None
    start_time = None

    feedback_msg = "Perform the exercise"
    last_spoken_feedback = None  # only speak when feedback changes
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    if show_summary:
                        running = False
                    else:
                        show_summary = True
                        if session_on and start_time:
                            dur = time.time() - start_time
                            avg = (sum(session_scores) / len(session_scores)) * 100 if session_scores else 0
                            save_session(exercise, target_reps, max(motion_ctr.reps, score_ctr.count), session_scores, session_fb, dur)
                            export_for_power_bi()
                            summary_data = (exercise, target_reps, max(motion_ctr.reps, score_ctr.count), avg, dur)
                elif ev.key == pygame.K_SPACE:
                    if show_summary:
                        show_summary = False
                        motion_ctr.reset()
                        score_ctr.reset()
                        session_on = True
                        start_time = time.time()
                        session_scores.clear()
                        session_fb.clear()
                    else:
                        session_on = not session_on
                        if session_on:
                            motion_ctr.reset()
                            score_ctr.reset()
                            start_time = time.time()
                            session_scores.clear()
                            session_fb.clear()
                elif ev.key == pygame.K_r:
                    motion_ctr.reset()
                    score_ctr.reset()
                    engine.reset_reps()
            elif ev.type == pygame.MOUSEWHEEL and dd_open:
                ex_list = EXERCISE_ORDER
                n = len(ex_list)
                if n > DD_VISIBLE_ITEMS:
                    dd_scroll = max(0, min(n - DD_VISIBLE_ITEMS, dd_scroll - ev.y))
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                # YouTube button (left panel)
                if youtube_btn_rect and youtube_btn_rect.collidepoint(mx, my):
                    url = EXERCISE_YOUTUBE.get(exercise, "")
                    if url:
                        webbrowser.open(url)
                dx = w - DD_W - PAD
                if dx <= mx <= dx + DD_W and 12 <= my <= 52:
                    dd_open = not dd_open
                    if dd_open:
                        dd_scroll = 0
                elif dd_open:
                    dly = 56
                    ex_list = DROPDOWN_EXERCISES
                    n = len(ex_list)
                    # Click on list area (items, not scrollbar)
                    if dx <= mx < dx + DD_W - DD_SB_W and dly <= my < dly + DD_LIST_H:
                        vi = (my - dly - 6) // DD_ITEM_H
                        idx = dd_scroll + vi
                        if 0 <= idx < n:
                            exercise = ex_list[idx]
                            motion_ctr.set_exercise(exercise)
                            open_demo_for(exercise)
                            step_paths = get_step_image_paths(exercise)
                            dd_open = False
                    # Click on scrollbar track/thumb: jump to position
                    elif dx + DD_W - DD_SB_W <= mx <= dx + DD_W and dly <= my < dly + DD_LIST_H and n > DD_VISIBLE_ITEMS:
                        thumb_frac = DD_VISIBLE_ITEMS / n
                        thumb_h = max(20, int(DD_LIST_H * thumb_frac))
                        range_ = n - DD_VISIBLE_ITEMS
                        rel = (my - dly - thumb_h / 2) / max(1, DD_LIST_H - thumb_h)
                        dd_scroll = max(0, min(range_, int(rel * range_)))
                    else:
                        dd_open = False

        w, h = screen.get_size()

        # If background YouTube demo download finished, open cached file or clear loading state
        while demo_ready_list:
            ex_ready, success = demo_ready_list.pop(0)
            if ex_ready != exercise:
                continue
            if success:
                open_demo_for(exercise)
            else:
                downloading_demo = None
            break

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame, landmarks = extractor.extract(frame)
        has_pose = landmarks is not None

        has_valid_motion = False
        if has_pose:
            buffer.add(landmarks)
            frame_count += 1
            # Only count reps during an active session so idle/jitter doesn't increase count
            if session_on:
                motion_ctr.update(landmarks)
            if buffer.is_ready() and (frame_count % max(1, inference_interval) == 0):
                seq = buffer.get_sequence()
                motion_energy = sequence_motion_energy(seq)
                recent_energy = recent_motion_energy(seq, last_n_frames=RECENT_MOTION_FRAMES)
                motion_passed = motion_energy >= MOTION_THRESHOLD and recent_energy >= MOTION_THRESHOLD
                # Run model; with joint attention, gate quality/reps by motion in attended body parts (graph pose)
                score, joint_weights = engine.predict(seq)
                has_valid_motion = True  # we have a new score to show
                if joint_weights is not None:
                    att_motion = attention_weighted_motion_fn(seq, joint_weights, last_n_frames=RECENT_MOTION_FRAMES)
                    attention_motion_ok = att_motion >= ATTENTION_MOTION_THRESHOLD
                else:
                    attention_motion_ok = motion_passed  # fallback when model has no joint attention
                if session_on and attention_motion_ok:
                    score_ctr.update(score, frame_count)
                    session_scores.append(score)
                if attention_motion_ok:
                    errors, err_joints = detect_errors(landmarks, exercise)
                    if errors:
                        feedback_msg = errors[0]
                        frame = highlight_joints(frame, landmarks, err_joints)
                    else:
                        feedback_msg = get_feedback_from_score(score)
                    if session_on:
                        session_fb.append(feedback_msg)
                else:
                    feedback_msg = "Perform the exercise"
        else:
            feedback_msg = "Move into camera view"
        # Audio: speak feedback when it changes (TTS)
        if feedback_msg != last_spoken_feedback:
            last_spoken_feedback = feedback_msg
            speak_feedback(feedback_msg)

        if has_valid_motion:
            # Lerp quickly toward model score so quality % actually increases when you do well
            display_score = lerp(display_score, score, 0.28)
        else:
            # When idle: decay slowly so last score stays visible (was 0.14)
            display_score = lerp(display_score, 0.0, 0.06)
        reps = max(motion_ctr.reps, score_ctr.count)

        # Demo video: time-based playback so it runs at real speed (independent of app FPS)
        if demo_cap is not None and demo_cap.isOpened() and demo_clip_start_time is not None and demo_exercise is not None:
            skip_sec, clip_sec = get_demo_clip(demo_exercise)
            elapsed = time.time() - demo_clip_start_time
            position_sec = elapsed % clip_sec  # loop within exercise clip
            seek_msec = (skip_sec + position_sec) * 1000
            demo_cap.set(cv2.CAP_PROP_POS_MSEC, seek_msec)
            ret, demo_f = demo_cap.read()
            if ret:
                if DEMO_FLIP_FOR_TEXT:
                    demo_f = cv2.flip(demo_f, 1)
                demo_f = cv2.cvtColor(demo_f, cv2.COLOR_BGR2RGB)
                demo_f = np.rot90(demo_f)
                try:
                    demo_frame_surf = pygame.surfarray.make_surface(demo_f)
                except Exception:
                    demo_frame_surf = pygame.Surface((demo_f.shape[1], demo_f.shape[0]))
                    demo_frame_surf.fill((50, 55, 65))

        # Live camera surface (right panel)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        try:
            surf = pygame.surfarray.make_surface(frame_rgb)
        except Exception:
            surf = pygame.Surface((frame.shape[1], frame.shape[0]))
            surf.fill((50, 55, 65))

        left_w = max(280, int(w * DEMO_FRAC))
        mid_w = max(280, int(w * STEPS_FRAC))
        right_w = w - left_w - mid_w
        right_surf = scale_fit(surf, right_w, VIDEO_H)
        step_index = int((time.time() - step_cycle_start) / 3) % max(1, len(step_paths)) if step_paths else 0

        screen.fill(BG)
        # Section 1: Demo video
        youtube_btn_rect = draw_left_panel(
            screen, 0, VIDEO_Y, left_w, VIDEO_H,
            demo_frame_surf, exercise, font_m, font_s,
            EXERCISE_YOUTUBE.get(exercise, ""),
            downloading_exercise=downloading_demo,
        )
        # Section 2: Step images / pose reference
        draw_middle_panel(screen, left_w, VIDEO_Y, mid_w, VIDEO_H, exercise, step_paths, step_index, font_m, font_s)
        # Section 3: Live camera
        rw, rh = right_surf.get_size()
        rx = left_w + mid_w + (right_w - rw) // 2
        ry = VIDEO_Y + (VIDEO_H - rh) // 2
        screen.blit(right_surf, (rx, ry))
        you_lbl = font_s.render("3. You (Live)", True, GRAY)
        screen.blit(you_lbl, (left_w + mid_w + 12, VIDEO_Y + 8))

        if show_summary and summary_data:
            draw_summary(screen, *summary_data, font_l, font_m, w, h)
        else:
            draw_header(screen, display_score, exercise, dd_open, dd_scroll, font_m, font_s, w, h)
            draw_footer(screen, reps, target_reps, feedback_msg, font_m, font_s, w, h)
            st = "● Recording" if session_on else "SPACE Start · R Reset · ESC End"
            st_txt = font_s.render(st, True, GRAY)
            screen.blit(st_txt, (w - st_txt.get_width() - PAD, h - FOOTER_H + 14))

        pygame.display.flip()
        clock.tick(30)

    if demo_cap is not None:
        demo_cap.release()
    cap.release()
    extractor.close()
    pygame.quit()


if __name__ == "__main__":
    _restore_stdio(*_STDIO_SAVED)  # Restore so --help and errors are visible
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/rehab_model.keras")
    ap.add_argument("--simplified", action="store_true")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    _STDIO_SAVED = _silent_stdio()  # Quiet again before run_app
    try:
        run_app(model_path=args.model, use_simplified=args.simplified, config_path=args.config)
    except Exception:
        _restore_stdio(*_STDIO_SAVED)
        raise
