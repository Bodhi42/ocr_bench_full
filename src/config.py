from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_PATH = DATA_DIR / "annotations" / "annotations_cvat.xml"

PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
RESULTS_DIR = PROJECT_ROOT / "results"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

IOU_THRESHOLDS = [0.3, 0.5, 0.7]
DEFAULT_IOU_THRESHOLD = 0.5
