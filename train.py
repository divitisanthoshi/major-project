"""
Training Script for Rehabilitation Exercise Quality Grading Model

Supports:
- UI-PRMD, KIMORE, NTU RGB+D, Custom datasets
- Full ST-GCN or simplified LSTM model
- Early stopping, validation split
- Model checkpointing and TensorBoard
"""

import os
import argparse
import sys
import json
from datetime import datetime

# Ensure project root in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import load_config
from src.models.st_gcn import build_rehab_grading_model, build_simplified_model
from src.datasets.ui_prmd import UIPRMDLoader
from src.datasets.kimore import KimoreLoader
from src.datasets.ntu_rgbd import NTURGBDLoader
from src.datasets.custom_webcam import CustomWebcamDataset
from src.datasets.synthetic_rehab import generate_realistic_synthetic


def get_loader(name: str, config: dict):
    """Return appropriate dataset loader."""
    base = config.get("sequence", {})
    seq_len = base.get("frame_buffer_size", 64)
    stride = base.get("stride", 8)
    num_joints = config.get("model", {}).get("num_joints", 33)

    paths = config.get("datasets", {})
    if name == "ui_prmd":
        path = paths.get("ui_prmd_path", "data/ui_prmd")
        return UIPRMDLoader(path, sequence_length=seq_len, stride=stride, num_joints=num_joints)
    elif name == "kimore":
        path = paths.get("kimore_path", "data/kimore")
        return KimoreLoader(path, sequence_length=seq_len, stride=stride, num_joints=num_joints)
    elif name == "ntu":
        path = paths.get("ntu_rgbd_path", "data/ntu_rgbd")
        return NTURGBDLoader(path, sequence_length=seq_len, stride=stride, num_joints=num_joints)
    elif name == "custom":
        path = paths.get("custom_path", "data/custom")
        return CustomWebcamDataset(path, sequence_length=seq_len, stride=stride, num_joints=num_joints)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_all_datasets(config: dict, seq_len: int = 64, num_joints: int = 33, include_custom: bool = True) -> tuple:
    """
    Load and combine UI-PRMD, KIMORE, NTU RGB+D, and (optionally) custom datasets.
    Returns (X, y, metadata) where metadata is {dataset_name: count}.
    """
    all_sequences = []
    all_scores = []
    metadata = {}

    loaders = [
        ("UI-PRMD", "ui_prmd"),
        ("KIMORE", "kimore"),
        ("NTU RGB+D", "ntu"),
    ]
    if include_custom:
        loaders.append(("Custom", "custom"))

    for name, loader_key in loaders:
        loader = get_loader(loader_key, config)
        X, y = loader.load()
        n = len(X)
        metadata[name] = n
        if n > 0:
            all_sequences.append(X)
            all_scores.append(y)
            print(f"[Train] {name}: loaded {n} sequences")
        else:
            print(f"[Train] {name}: no data found (path may not exist)")

    if len(all_sequences) == 0:
        return np.zeros((0, seq_len, num_joints, 3), dtype=np.float32), np.zeros(0, dtype=np.float32), metadata

    X_combined = np.concatenate(all_sequences, axis=0)
    y_combined = np.concatenate(all_scores, axis=0)

    # Shuffle combined data
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_combined))
    X_combined = X_combined[idx]
    y_combined = y_combined[idx]

    metadata["total_combined"] = len(X_combined)
    sources = "UI-PRMD, KIMORE, NTU RGB+D" + (", Custom" if include_custom else "")
    print(f"[Train] Combined: {len(X_combined)} sequences from {sources}")
    return X_combined, y_combined, metadata


def load_all_three_datasets(config: dict, seq_len: int = 64, num_joints: int = 33) -> tuple:
    """Legacy: Load UI-PRMD + KIMORE + NTU (no custom)."""
    return load_all_datasets(config, seq_len, num_joints, include_custom=False)


def generate_synthetic_data(n_samples: int = 1500, seq_len: int = 64, num_joints: int = 33) -> tuple:
    """Generate realistic synthetic data with exercise-like motion when real datasets unavailable."""
    return generate_realistic_synthetic(
        n_samples=n_samples,
        seq_len=seq_len,
        num_joints=num_joints,
        n_exercises=15,
        seed=42,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["all", "ui_prmd", "kimore", "ntu", "custom", "synthetic"],
                        help="'all' trains on UI-PRMD + KIMORE + NTU RGB+D combined")
    parser.add_argument("--model", type=str, default="stgcn", choices=["stgcn", "simplified"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", type=str, default="models/rehab_model.keras")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    seq_cfg = config.get("sequence", {})

    seq_len = seq_cfg.get("frame_buffer_size", 64)
    num_joints = model_cfg.get("num_joints", 33)
    in_channels = model_cfg.get("in_channels", 3)

    # Load data
    train_metadata = {"dataset_mode": args.dataset, "datasets_used": {}}
    if args.dataset == "all":
        print("[Train] Loading all datasets: UI-PRMD, KIMORE, NTU RGB+D, Custom...")
        result = load_all_datasets(config, seq_len=seq_len, num_joints=num_joints, include_custom=True)
        X, y = result[0], result[1]
        train_metadata["datasets_used"] = result[2]
        if len(X) == 0:
            print("[Train] No data found in any dataset. Using realistic synthetic (exercise-like motion).")
            X, y = generate_synthetic_data(n_samples=1500, seq_len=seq_len, num_joints=num_joints)
            train_metadata["datasets_used"] = {"UI-PRMD": 0, "KIMORE": 0, "NTU RGB+D": 0, "synthetic_realistic": 1500}
    elif args.dataset == "synthetic":
        print("[Train] Using realistic synthetic data (exercise-like motion, 15 types).")
        X, y = generate_synthetic_data(n_samples=1500, seq_len=seq_len, num_joints=num_joints)
        train_metadata["datasets_used"] = {"synthetic_realistic": len(X)}
    else:
        loader = get_loader(args.dataset, config)
        X, y = loader.load()
        train_metadata["datasets_used"] = {args.dataset: len(X) if len(X) > 0 else 0}
        if len(X) == 0:
            print("[Train] No data found. Using realistic synthetic.")
            X, y = generate_synthetic_data(n_samples=1500, seq_len=seq_len, num_joints=num_joints)
            train_metadata["datasets_used"]["synthetic_realistic"] = 1500

    print(f"[Train] Loaded {len(X)} sequences, shape {X.shape}")

    # Build model
    if args.model == "stgcn":
        model = build_rehab_grading_model(
            num_joints=num_joints,
            in_channels=in_channels,
            sequence_length=seq_len,
            stgcn_channels=model_cfg.get("stgcn_channels", [64, 64, 128, 256]),
            lstm_units=model_cfg.get("lstm_units", 128),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            attention_units=model_cfg.get("attention_units", 64),
            dropout=model_cfg.get("dropout", 0.3),
        )
    else:
        model = build_simplified_model(
            num_joints=num_joints,
            in_channels=in_channels,
            sequence_length=seq_len,
            lstm_units=model_cfg.get("lstm_units", 128),
            dropout=model_cfg.get("dropout", 0.3),
        )

    # Training
    batch_size = args.batch_size or train_cfg.get("batch_size", 32)
    epochs = args.epochs or train_cfg.get("epochs", 100)
    lr = train_cfg.get("learning_rate", 0.001)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 15),
            restore_best_weights=True,
            monitor="val_loss",
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    log_dir = "logs"
    if os.path.exists("logs"):
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir))

    val_split = train_cfg.get("validation_split", 0.2)
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    # Build manifest
    manifest = {
        "model_path": os.path.abspath(args.output),
        "trained_at": datetime.now().isoformat(),
        "dataset_mode": train_metadata["dataset_mode"],
        "datasets_used": train_metadata["datasets_used"],
        "total_samples": int(len(X)),
        "exercises": [
            "deep_squat", "hurdle_step", "inline_lunge", "side_lunge", "sit_to_stand",
            "standing_leg_raise", "shoulder_abduction", "shoulder_extension",
            "shoulder_rotation", "shoulder_scaption", "hip_abduction", "trunk_rotation",
            "squat", "leg_raise", "reach_and_retrieve",
        ],
        "model_type": args.model,
        "sequence_length": seq_len,
        "num_joints": num_joints,
        "epochs_run": len(history.history.get("loss", [])),
        "final_val_loss": float(history.history.get("val_loss", [None])[-1]) if history.history.get("val_loss") else None,
    }

    # Save model and manifest
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    model.save(args.output)
    manifest_path = args.output.rsplit(".", 1)[0] + "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Train] Model saved to {args.output}")
    print(f"[Train] Training manifest saved to {manifest_path}")
    print("\n--- Training provenance ---")
    print(f"  Datasets: {json.dumps(manifest['datasets_used'], indent=4)}")
    print(f"  Total samples: {manifest['total_samples']}")
    print("----------------------------")


if __name__ == "__main__":
    main()
