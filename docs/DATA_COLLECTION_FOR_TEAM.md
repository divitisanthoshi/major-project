# Data Collection Guide – For Team Members (No Code Required)

**Share this document with your teammate who will collect the custom dataset.**  
They only need a phone or laptop camera – no project code or Python.

---

## What You Need to Do

Record **short videos** of people performing rehabilitation exercises.  
Use your **phone** or **laptop webcam** – no special software.

---

## 8 Exercises to Record

| # | Exercise | How to Do It | Camera View |
|---|----------|--------------|-------------|
| 1 | **Wall push-up** | Stand arms-length from wall, hands shoulder-height. Bend elbows, chest toward wall, push back. | Front or side |
| 2 | **Heel raise** | Stand, feet hip-width. Rise onto toes, pause, lower slowly. | Front or side |
| 3 | **Bird dog** | On hands and knees. Extend one arm forward + opposite leg back. Hold, return. Repeat other side. | Side |
| 4 | **Glute bridge** | Lie on back, knees bent. Lift hips up, squeeze glutes, lower slowly. | Side |
| 5 | **Clamshell** | Lie on side, knees bent. Lift top knee (feet together). Lower. Repeat. | Front |
| 6 | **Chin tuck** | Sit or stand. Draw chin straight back (like making a double chin). Hold 2–3 sec. | Front |
| 7 | **Marching in place** | Stand tall. March in place, knees up comfortably. Keep torso steady. | Front |
| 8 | **Step-up** | Use a step/stool (15–20 cm). Step up, other foot follows. Step down. Alternate legs. | Front or side |

---

## Recording Rules

1. **Full body visible** – Person must be fully in frame (head to feet).
2. **Good lighting** – Face the light; avoid strong backlight.
3. **Steady camera** – Phone on a stand or laptop on a table.
4. **Distance** – Camera about 2–3 meters from the person.
5. **Each video** – One exercise, 2–5 reps, about 5–15 seconds long.

---

## File Naming (IMPORTANT)

Name each video file exactly like this:

```
<exercise>_<quality>_<number>.mp4
```

| Part | Options | Examples |
|------|---------|----------|
| `<exercise>` | wall_pushup, heel_raise, bird_dog, glute_bridge, clamshell, chin_tuck, marching_in_place, step_up | `wall_pushup` |
| `<quality>` | good, moderate, poor | `good` |
| `<number>` | 01, 02, 03... | `01` |

### Examples

| Filename | Meaning |
|----------|---------|
| `wall_pushup_good_01.mp4` | Wall push-up, correct form, sample 1 |
| `heel_raise_moderate_03.mp4` | Heel raise, slight errors, sample 3 |
| `chin_tuck_poor_02.mp4` | Chin tuck, noticeable errors, sample 2 |
| `marching_in_place_good_05.mp4` | Marching, correct form, sample 5 |

### Quality Guide (for naming)

- **good** – Correct form, controlled movement.
- **moderate** – Slight errors (e.g., limited range, small compensations).
- **poor** – Clear errors (e.g., wrong posture, unstable).

---

## How Many to Record

| Per exercise | Minimum | Recommended |
|--------------|---------|-------------|
| Total | 20 videos | 40–60 videos |
| good | 8 | 15–25 |
| moderate | 8 | 12–20 |
| poor | 4 | 8–15 |

**Different people** (ages, body types) improve the dataset.

---

## Organize Before Sending

Create folders by exercise and put videos inside:

```
custom_data/
├── wall_pushup/
│   ├── wall_pushup_good_01.mp4
│   ├── wall_pushup_good_02.mp4
│   ├── wall_pushup_moderate_01.mp4
│   └── ...
├── heel_raise/
│   ├── heel_raise_good_01.mp4
│   └── ...
├── bird_dog/
├── glute_bridge/
├── clamshell/
├── chin_tuck/
├── marching_in_place/
└── step_up/
```

---

## Send to the Project Lead

1. Zip the `custom_data` folder.
2. Upload to **Google Drive**, **OneDrive**, or **WeTransfer**.
3. Share the link with the project lead.

The project lead will convert these videos into the format needed for training.

---

## Summary Checklist

- [ ] Record 8 exercises (wall_pushup, heel_raise, bird_dog, glute_bridge, clamshell, chin_tuck, marching_in_place, step_up)
- [ ] Use naming: `exercise_quality_number.mp4` (e.g. `wall_pushup_good_01.mp4`)
- [ ] Full body visible, good lighting, steady camera
- [ ] 5–15 seconds per video, 2–5 reps per video
- [ ] Aim for 20+ videos per exercise (mix of good, moderate, poor)
- [ ] Organize in exercise folders, zip, and share via cloud link
