import os
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("DETECTOR_DATA", BASE / "data"))
DATA.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA / "corpus.db"
REPORTS = BASE / "reports"; REPORTS.mkdir(exist_ok=True)
MANIFEST = Path(os.environ.get("DETECTOR_MANIFEST", BASE / "models" / "manifest.json"))
REMOTE_MANIFEST_URL = os.environ.get("DETECTOR_MANIFEST_URL", "")  # http(s):// or file:// path
WEB_SEARCH_API_KEY = os.environ.get("WEB_SEARCH_API_KEY", "")
LIGHT_MODE = os.environ.get("DETECTOR_LIGHT", "0") == "1"   # tests: skip heavy models
