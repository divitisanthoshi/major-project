"""Dataset loaders for rehabilitation exercise grading."""

from src.datasets.base_loader import BaseDatasetLoader
from src.datasets.custom_webcam import CustomWebcamDataset
from src.datasets.ui_prmd import UIPRMDLoader
from src.datasets.kimore import KimoreLoader
from src.datasets.ntu_rgbd import NTURGBDLoader

__all__ = [
    "BaseDatasetLoader",
    "CustomWebcamDataset",
    "UIPRMDLoader",
    "KimoreLoader",
    "NTURGBDLoader",
]
