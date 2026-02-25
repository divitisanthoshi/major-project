# Custom Dataset Guide – Novel Rehab Exercises

---

## For Project Lead: Teammate in Another Location

**If a teammate is collecting data remotely (no project access):**

1. **Send them** `docs/DATA_COLLECTION_FOR_TEAM.md` – they only need a phone or laptop camera.
2. **They record videos** with naming `exercise_quality_number.mp4` (e.g. `wall_pushup_good_01.mp4`).
3. **They upload** the folder (e.g. zip) to Google Drive / OneDrive and share the link.
4. **You run** the converter on the downloaded videos:

   ```powershell
   python scripts/video_to_skeleton.py --input path/to/downloaded/custom_data --output data/custom
   ```

5. **Then train:** `python train.py --dataset custom`

---

## Using Images (e.g. from Google) Instead of Videos

If you have **static images** (screenshots, stock photos) instead of videos:

1. **Organize** images in folders: `images/wall_pushup/`, `images/heel_raise/`, etc.
2. **Name** files: `wall_pushup_good_01.jpg` or `wall_pushup_02.png` (quality in name → score)
3. **Run** the image converter:

   ```powershell
   python scripts/images_to_skeleton.py --input path/to/images --output data/custom
   ```

**Limitation:** Images provide static poses only. The model expects motion over time. Each image is repeated 64× to form a sequence, so results are weaker than real video. Use images for augmentation or proof-of-concept; videos are better for final training.

---

## Novelty Statement

**"We introduce a custom dataset of real-world rehabilitation exercises that are not present in any of the three benchmark datasets (UI-PRMD, KIMORE, NTU RGB+D)."**

| Dataset | Exercises | Our Novel Additions |
|---------|-----------|---------------------|
| **UI-PRMD** | Deep squat, Hurdle step, Inline lunge, Side lunge, Sit to stand, Standing leg raise, Shoulder abduction/extension/rotation/scaption | ❌ Not in ours |
| **KIMORE** | Squat, Hip abduction, Sit to stand, Leg raise, Trunk rotation | ❌ Not in ours |
| **NTU RGB+D** | 60 daily/medical actions (sit down, stand up, reach, etc.) | ❌ Not rehab-specific |
| **Our custom** | Wall push-up, Heel raise, Bird dog, Glute bridge, Clamshell, Chin tuck, Marching in place, Step-up | ✅ **Novel** |

---

## 8 Novel Rehabilitation Exercises (Not in UI-PRMD, KIMORE, NTU)

These exercises are widely used in clinical practice but **absent** from the three public datasets:

| # | Exercise | Key joints | Clinical use |
|---|----------|------------|--------------|
| 1 | **Wall push-up** | Shoulders, elbows, torso | Post-surgery, stroke, low fitness |
| 2 | **Heel raise (Calf raise)** | Ankles, knees | Ankle rehab, DVT prevention |
| 3 | **Bird dog** | Spine, hips, shoulders | Low back pain, core stability |
| 4 | **Glute bridge** | Hips, spine | Core, hip, postpartum |
| 5 | **Clamshell** | Hips (external rotation) | Gluteus medius, hip rehab |
| 6 | **Chin tuck** | Neck (cervical) | Cervical posture, headache |
| 7 | **Marching in place** | Hips, knees | Stroke, balance, gait |
| 8 | **Step-up** | Hips, knees, ankles | ACL, knee rehab, functional mobility |

---

## Point-by-Point Capture Instructions

### 1. Prerequisites

- Webcam (720p or better recommended)
- Good lighting (face the light; avoid backlight)
- Full-body visibility (camera 2–3 m away)
- Python environment with project dependencies

### 2. Setup

```powershell
cd c:\Users\santh\OneDrive\Desktop\MPP-2
```

The recorder creates exercise folders automatically when you use `--exercise`.

### 3. Camera Position by Exercise

| Exercise | Camera view | Distance |
|----------|-------------|----------|
| Wall push-up | **Front** (facing camera, hands visible on wall behind) or **Side** | 2 m |
| Heel raise | **Front** or **Side** (ankles visible) | 2 m |
| Bird dog | **Side** (quadruped posture visible) | 2 m |
| Glute bridge | **Side** (hips and legs visible) | 2 m |
| Clamshell | **Front** (side-lying, top leg moving) | 2 m |
| Chin tuck | **Front** (head/neck clear) | 1.5 m |
| Marching in place | **Front** | 2 m |
| Step-up | **Front** or **Side** (step/stool visible) | 2 m |

### 4. Recording Workflow (Step-by-Step)

#### Step 4.1 – Start recorder

```powershell
# Record to exercise-specific subfolder
python scripts/record_custom_data.py --output data/custom --exercise wall_pushup
```

#### Step 4.2 – Perform one rep, then stop

1. **Position**: Stand in frame, full body visible.
2. **Press SPACE** – Start recording.
3. **Perform 1–2 reps** of the exercise (aim for ~2–4 s per rep).
4. **Press SPACE** – Stop and save (creates `seq_000.npy`, etc.).
5. Repeat for more samples (10–20 per quality level).

#### Step 4.3 – Record multiple quality levels

- **Good (0.7–1.0)**: Correct form, controlled movement.
- **Moderate (0.4–0.7)**: Slight errors (e.g. limited range).
- **Poor (0.0–0.4)**: Noticeable errors (e.g. compensation, imbalance).

Use different performers or intentionally vary form to get spread.

### 5. Create `labels.csv`

In each exercise folder (e.g. `data/custom/wall_pushup/`):

```
file,score
seq_000.npy,0.95
seq_001.npy,0.85
seq_002.npy,0.45
seq_003.npy,0.65
...
```

- `file`: exact filename
- `score`: 0.0–1.0 (quality)

### 6. Recommended Samples per Exercise

| Exercise | Min samples | Recommended |
|----------|-------------|-------------|
| Per exercise | 30 | 50–100 |
| Good (0.7–1.0) | 15 | 25–40 |
| Moderate (0.4–0.7) | 10 | 15–30 |
| Poor (0.0–0.4) | 5 | 10–20 |

### 7. Training with Custom Data

**Option A – Custom only**

```powershell
python train.py --dataset custom --model stgcn --output models/rehab_custom.keras
```

**Option B – All datasets (UI-PRMD + KIMORE + NTU + Custom)**

The `--dataset all` mode now automatically includes your custom data:

```powershell
python train.py --dataset all --model stgcn
```

If no real data exists, synthetic fallback is used. Custom samples are merged when present.

**Option C – Custom + synthetic (when custom is small)**

If custom has few samples, the loader uses them; add more by capturing more sequences.

---

## Exercise Movement Descriptions (For Performers)

### 1. Wall push-up

- Stand ~arm’s length from wall, hands at shoulder height.
- Bend elbows, bring chest toward wall, then push back.
- Keep body straight; control the movement.

### 2. Heel raise

- Stand with feet hip-width apart.
- Rise onto toes, pause, lower slowly.
- Keep knees slightly bent.

### 3. Bird dog

- Start on hands and knees (quadruped).
- Extend one arm forward and opposite leg back; hold briefly.
- Return and repeat on other side.

### 4. Glute bridge

- Lie on back, knees bent, feet flat.
- Lift hips toward ceiling, squeeze glutes.
- Lower with control.

### 5. Clamshell

- Lie on side, knees bent 45°, heels together.
- Lift top knee while keeping feet together.
- Lower with control.

### 6. Chin tuck

- Sit or stand with neutral spine.
- Draw chin straight back (no nodding).
- Hold 2–3 s, release.

### 7. Marching in place

- Stand tall, march in place.
- Lift knees to a comfortable height.
- Keep torso stable.

### 8. Step-up

- Use a sturdy step (e.g. 15–20 cm).
- Step up with one foot, bring other foot up.
- Step down with control; alternate legs.

---

## File Structure After Capture

```
data/custom/
├── wall_pushup/
│   ├── seq_000.npy
│   ├── seq_001.npy
│   ├── ...
│   └── labels.csv
├── heel_raise/
│   ├── seq_000.npy
│   └── labels.csv
├── bird_dog/
├── glute_bridge/
├── clamshell/
├── chin_tuck/
├── marching_in_place/
└── step_up/
```

---

## Quick Reference

| Step | Command / Action |
|------|-------------------|
| 1 | `python scripts/record_custom_data.py --output data\custom --exercise wall_pushup` |
| 2 | SPACE = Start/Stop recording (saves to `data/custom/wall_pushup/`) |
| 3 | Add `labels.csv` in each exercise folder with `file,score` |
| 4 | `python train.py --dataset custom` or `python train.py --dataset all` |
