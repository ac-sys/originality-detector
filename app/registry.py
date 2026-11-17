"""Model registry & update mechanism: versioned detectors in a JSON manifest, remote version check,
download (pre-fetch weights), evaluation gate, atomic activation (hot-swap) and rollback."""
import json, copy, time, urllib.request, threading
from .config import MANIFEST, REMOTE_MANIFEST_URL
class Registry:
    def __init__(self, path=MANIFEST):
        self.path = path; self.lock = threading.Lock(); self.manifest = json.loads(path.read_text())
    def save(self):
        tmp = self.path.with_suffix(".tmp"); tmp.write_text(json.dumps(self.manifest, indent=2)); tmp.replace(self.path)
    def spec(self, did):
        for d in self.manifest["detectors"]:
            if d["id"] == did: return d
        raise KeyError(did)
    def fetch_remote(self, url=None):
        url = url or REMOTE_MANIFEST_URL
        if not url: return None
        if url.startswith("file://"):
            return json.loads(open(url[7:]).read())
        with urllib.request.urlopen(url, timeout=20) as r: return json.loads(r.read())
    def check(self, url=None):
        remote = self.fetch_remote(url)
        if remote is None: return {"configured": False, "updates": []}
        known = {d["id"] for d in self.manifest["detectors"]}
        ups = [d for d in remote.get("detectors", []) if d["id"] not in known]
        cal_new = remote.get("calibration", {}).get("version") not in (None, self.manifest["calibration"]["version"])
        return {"configured": True, "updates": ups, "calibration_update": cal_new, "remote_active": remote.get("active")}
    def stage(self, remote):
        """Register new detector specs and pre-download weights (does not activate)."""
        from huggingface_hub import snapshot_download
        known = {d["id"] for d in self.manifest["detectors"]}
        added = []
        for d in remote.get("detectors", []):
            if d["id"] not in known:
                snapshot_download(d["hf_id"]); self.manifest["detectors"].append(d); added.append(d["id"])
        self.save(); return added
    def activate(self, active=None, calibration=None, reason=""):
        with self.lock:
            self.manifest["history"].append({"time": time.time(), "active": copy.deepcopy(self.manifest["active"]),
                                             "calibration": copy.deepcopy(self.manifest["calibration"]),
                                             "similarity": copy.deepcopy(self.manifest.get("similarity")), "reason": reason})
            if active: self.manifest["active"].update(active)
            if calibration: self.manifest["calibration"] = calibration
            self.save()
    def rollback(self):
        with self.lock:
            if not self.manifest["history"]: raise ValueError("Nothing to roll back to")
            prev = self.manifest["history"].pop()
            self.manifest["active"], self.manifest["calibration"] = prev["active"], prev["calibration"]
            if prev.get("similarity") is not None: self.manifest["similarity"] = prev["similarity"]
            elif "similarity" in prev: self.manifest.pop("similarity", None)
            self.save(); return prev
