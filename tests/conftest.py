import os, shutil, tempfile, sys, pathlib
d = tempfile.mkdtemp()
shutil.copy(pathlib.Path(__file__).parent.parent / "models/manifest.json", d + "/manifest.json")
os.environ["DETECTOR_DATA"] = d; os.environ["DETECTOR_MANIFEST"] = d + "/manifest.json"
os.environ.setdefault("DETECTOR_LIGHT", "1")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
