"""
Skeleton-Based Rehabilitation Exercise Quality Grading

Clean 4-zone layout: Header | Video | Footer
"""

import os
import sys
import argparse
import time
import warnings

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

# Only run quality prediction when pose is clearly moving (full sequence + recent window)
MOTION_THRESHOLD = 0.028
RECENT_MOTION_FRAMES = 16  # Must be moving in last N frames too (stops score rising after you sit)


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


def draw_header(screen, score, exercise, dd_open, font_m, font_s, w, h):
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
        n = len(EXERCISE_NAMES)
        dl_h = n * DD_ITEM_H + 12
        dly = dy + dh + 4
        pygame.draw.rect(screen, PANEL_BG, (dx, dly, DD_W, dl_h))
        pygame.draw.rect(screen, DARK, (dx, dly, DD_W, dl_h), 1)
        for i, ex in enumerate(EXERCISE_NAMES.keys()):
            iy = dly + 6 + i * DD_ITEM_H
            lab = truncate(font_s, EXERCISE_NAMES[ex], DD_W - 20)
            col = GREEN if ex == exercise else WHITE
            it = font_s.render(lab, True, col)
            screen.blit(it, (dx + 12, iy))


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


def _draw_loading_screen(screen, w, h, msg="Loading model and camera..."):
    """Show a loading message so the window appears quickly."""
    screen.fill(BG)
    font = get_font(28)
    txt = font.render(msg, True, WHITE)
    screen.blit(txt, ((w - txt.get_width()) // 2, (h - txt.get_height()) // 2))
    pygame.display.flip()


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
    # Fullscreen: show window immediately so user sees something
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    w, h = screen.get_size()
    pygame.display.set_caption("Rehab Exercise Quality Grading")
    _draw_loading_screen(screen, w, h)
    clock = pygame.time.Clock()

    # Heavy imports only after window is visible (TensorFlow, MediaPipe take 20–40 s)
    from src.pose_extraction import PoseExtractor
    from src.preprocessing import PoseBuffer, sequence_motion_energy, recent_motion_energy
    from src.inference import InferenceEngine
    from src.models.st_gcn import build_rehab_grading_model, build_simplified_model
    from src.joint_analysis import detect_errors, get_feedback_from_score
    from src.motion_rep_counter import MotionRepCounter
    from src.repetition_counter import RepetitionCounter
    from src.skeleton_highlight import highlight_joints
    from src.session_manager import save_session, export_for_power_bi
    from src.ui.components import lerp, draw_rounded_rect

    _draw_loading_screen(screen, w, h, "Opening camera...")

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    motion_ctr = MotionRepCounter(threshold=0.005, min_peak_distance=8)
    score_ctr = RepetitionCounter(threshold=0.5)

    font_l = get_font(26, bold=True)
    font_m = get_font(20)
    font_s = get_font(16)

    score = display_score = 0.0
    frame_count = 0
    exercise = "deep_squat"
    dd_open = False
    session_on = False
    session_scores = []
    session_fb = []
    show_summary = False
    summary_data = None
    start_time = None

    feedback_msg = "Perform the exercise"
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
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                dx = w - DD_W - PAD
                if dx <= mx <= dx + DD_W and 12 <= my <= 52:
                    dd_open = not dd_open
                elif dd_open:
                    dly = 56
                    dl_h = len(EXERCISE_NAMES) * DD_ITEM_H + 12
                    if dx <= mx <= dx + DD_W and dly <= my <= dly + dl_h:
                        idx = (my - dly - 6) // DD_ITEM_H
                        ex_list = list(EXERCISE_NAMES.keys())
                        if 0 <= idx < len(ex_list):
                            exercise = ex_list[idx]
                            motion_ctr.set_exercise(exercise)
                            dd_open = False
                    else:
                        dd_open = False

        w, h = screen.get_size()

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
            if session_on:
                motion_ctr.update(landmarks)
            if buffer.is_ready() and (frame_count % 2 == 0):
                seq = buffer.get_sequence()
                motion_energy = sequence_motion_energy(seq)
                recent_energy = recent_motion_energy(seq, last_n_frames=RECENT_MOTION_FRAMES)
                if motion_energy >= MOTION_THRESHOLD and recent_energy >= MOTION_THRESHOLD:
                    has_valid_motion = True
                    score = engine.predict(seq)
                    if session_on:
                        session_scores.append(score)
                        score_ctr.update(score, frame_count)
                    errors, err_joints = detect_errors(landmarks, exercise)
                    if errors:
                        feedback_msg = errors[0]
                        frame = highlight_joints(frame, landmarks, err_joints)
                    else:
                        feedback_msg = get_feedback_from_score(score)
                    session_fb.append(feedback_msg)
                else:
                    feedback_msg = "Perform the exercise"
        else:
            feedback_msg = "Move into camera view"

        if has_valid_motion:
            display_score = lerp(display_score, score, 0.12)
        else:
            # When idle: decay quickly so "—" appears soon after you stop moving
            display_score = lerp(display_score, 0.0, 0.14)
        reps = max(motion_ctr.reps, score_ctr.count)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.rot90(frame_rgb)
        try:
            surf = pygame.surfarray.make_surface(frame_rgb)
        except Exception:
            surf = pygame.Surface((frame.shape[1], frame.shape[0]))
            surf.fill((50, 55, 65))
        surf = scale_fit(surf, w, VIDEO_H)

        screen.fill(BG)
        vw, vh = surf.get_size()
        vx = (w - vw) // 2
        vy = VIDEO_Y + (VIDEO_H - vh) // 2
        screen.blit(surf, (vx, vy))

        if show_summary and summary_data:
            draw_summary(screen, *summary_data, font_l, font_m, w, h)
        else:
            draw_header(screen, display_score, exercise, dd_open, font_m, font_s, w, h)
            draw_footer(screen, reps, target_reps, feedback_msg, font_m, font_s, w, h)
            st = "● Recording" if session_on else "SPACE Start · R Reset · ESC End"
            st_txt = font_s.render(st, True, GRAY)
            screen.blit(st_txt, (w - st_txt.get_width() - PAD, h - FOOTER_H + 14))

        pygame.display.flip()
        clock.tick(30)

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
