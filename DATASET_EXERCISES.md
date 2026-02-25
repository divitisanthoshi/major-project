# Dataset Exercise Reference

Exact number and names of exercises in each supported dataset.

---

## 1. UI-PRMD Dataset

**University of Idaho – Physical Rehabilitation Movement Dataset**

**Total: 10 exercises**

| # | Exercise Name |
|---|---------------|
| 1 | Deep squat |
| 2 | Hurdle step |
| 3 | Inline lunge |
| 4 | Side lunge |
| 5 | Sit to stand |
| 6 | Standing active straight leg raise |
| 7 | Standing shoulder abduction |
| 8 | Standing shoulder extension |
| 9 | Standing shoulder internal-external rotation |
| 10 | Standing shoulder scaption |

**Source:** [webpages.uidaho.edu/ui-prmd](https://webpages.uidaho.edu/ui-prmd/)  
**Data:** 10 subjects × 10 repetitions per exercise. Optical (Vicon) + Kinect. Joint positions and angles.

---

## 2. KIMORE Dataset

**KInematic Assessment of MOvement and Clinical Scores for Remote Monitoring of Physical REhabilitation**

**Total: 5 exercises**

| # | Exercise Name |
|---|---------------|
| 1 | Squat |
| 2 | Hip abduction |
| 3 | Sit to stand |
| 4 | Leg raise |
| 5 | Trunk rotation |

**Note:** KIMORE uses an exercise index (Ex_idx) in its structure. The exact labels may vary by implementation. The dataset targets stroke, cardiac, and injury rehabilitation with kinematic joint data from Kinect V2.

**Source:** IEEE Xplore – “The KIMORE Dataset” (2019)

---

## 3. NTU RGB+D Dataset

**NTU RGB+D 60 – Large-scale 3D Human Activity Analysis**

**Total: 60 actions** (not all rehabilitation-focused)

### Daily Actions (40 actions)

| # | Action Name |
|---|-------------|
| A1 | Drink water |
| A2 | Eat meal |
| A3 | Brush teeth |
| A4 | Brush hair |
| A5 | Drop |
| A6 | Pick up |
| A7 | Throw |
| A8 | Sit down |
| A9 | Stand up |
| A10 | Clapping |
| A11 | Reading |
| A12 | Writing |
| A13 | Tear up paper |
| A14 | Put on jacket |
| A15 | Take off jacket |
| A16 | Put on a shoe |
| A17 | Take off a shoe |
| A18 | Put on glasses |
| A19 | Take off glasses |
| A20 | Put on a hat/cap |
| A21 | Take off a hat/cap |
| A22 | Cheer up |
| A23 | Hand waving |
| A24 | Kicking something |
| A25 | Reach into pocket |
| A26 | Hopping |
| A27 | Jump up |
| A28 | Phone call |
| A29 | Play with phone/tablet |
| A30 | Type on a keyboard |
| A31 | Point to something |
| A32 | Taking a selfie |
| A33 | Check time (from watch) |
| A34 | Rub two hands |
| A35 | Nod head/bow |
| A36 | Shake head |
| A37 | Wipe face |
| A38 | Salute |
| A39 | Put palms together |
| A40 | Cross hands in front |

### Medical Conditions (9 actions)

| # | Action Name |
|---|-------------|
| A41 | Sneeze/cough |
| A42 | Staggering |
| A43 | Falling down |
| A44 | Headache |
| A45 | Chest pain |
| A46 | Back pain |
| A47 | Neck pain |
| A48 | Nausea/vomiting |
| A49 | Fan self |

### Mutual / Two-person (11 actions)

| # | Action Name |
|---|-------------|
| A50 | Punch/slap |
| A51 | Kicking |
| A52 | Pushing |
| A53 | Pat on back |
| A54 | Point finger |
| A55 | Hugging |
| A56 | Giving object |
| A57 | Touch pocket |
| A58 | Shaking hands |
| A59 | Walking towards |
| A60 | Walking apart |

**Source:** NTU ROSE Lab – [rose1.ntu.edu.sg/dataset/actionRecognition](https://rose1.ntu.edu.sg/dataset/actionRecognition/)  
**Rehabilitation overlap:** Sit down (A8), Stand up (A9), Reach into pocket (A25), and the medical-condition actions are most relevant for rehab.

---

## Summary

| Dataset   | Number of Exercises/Actions | Rehabilitation-focused |
|-----------|----------------------------|-------------------------|
| UI-PRMD   | 10                         | Yes                     |
| KIMORE    | 5                          | Yes                     |
| NTU RGB+D | 60                         | Partially (e.g., A8, A9, A41–A49) |
