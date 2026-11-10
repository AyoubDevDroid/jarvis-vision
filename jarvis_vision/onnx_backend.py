"""Backend leger sans PyTorch : YOLO11 exporte en ONNX + tracker IoU simple."""

from pathlib import Path
import urllib.request

import numpy as np

COCO_VEHICLES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Associe les boites d'une image a l'autre par recouvrement (IoU)."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, detections):
        """detections : [(x1, y1, x2, y2, label, conf)] -> meme liste + track_id."""
        pairs = sorted(
            ((_iou(t["box"], d[:4]), tid, i)
             for tid, t in self.tracks.items()
             for i, d in enumerate(detections)),
            reverse=True,
        )
        assigned, used_tracks, out = {}, set(), []
        for score, tid, i in pairs:
            if score < self.iou_threshold:
                break
            if tid in used_tracks or i in assigned:
                continue
            assigned[i] = tid
            used_tracks.add(tid)

        for i, det in enumerate(detections):
            tid = assigned.get(i)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
            self.tracks[tid] = {"box": det[:4], "missed": 0}
            out.append((*det, tid))

        active = {o[6] for o in out}
        for tid in list(self.tracks):
            if tid not in active:
                self.tracks[tid]["missed"] += 1
                if self.tracks[tid]["missed"] > self.max_missed:
                    del self.tracks[tid]
        return out


class OnnxVehicleDetector:
    def __init__(self, model_path: str = "yolo11n.onnx", imgsz: int = 640, conf: float = 0.35):
        import cv2
        import onnxruntime as ort

        path = Path(model_path)
        if not path.exists():
            print(f"Telechargement de {path.name}...")
            urllib.request.urlretrieve(DEFAULT_URL, path)
        self.cv2 = cv2
        self.session = ort.InferenceSession(str(path), providers=ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = imgsz
        self.conf = conf
        self.tracker = IouTracker()

    def __call__(self, frame):
        cv2 = self.cv2
        h, w = frame.shape[:2]
        scale = self.imgsz / max(h, w)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        pred = self.session.run(None, {self.input_name: blob})[0][0].T  # (8400, 84)
        scores = pred[:, 4:]
        class_ids = scores.argmax(1)
        confs = scores.max(1)
        keep = (confs >= self.conf) & np.isin(class_ids, list(COCO_VEHICLES))
        if not keep.any():
            return self.tracker.update([])
        boxes, confs, class_ids = pred[keep, :4], confs[keep], class_ids[keep]
        xywh = np.stack([
            (boxes[:, 0] - boxes[:, 2] / 2 - left) / scale,
            (boxes[:, 1] - boxes[:, 3] / 2 - top) / scale,
            boxes[:, 2] / scale,
            boxes[:, 3] / scale,
        ], 1)
        idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), confs.tolist(), class_ids.tolist(), self.conf, 0.5)
        detections = []
        for i in np.array(idx).flatten():
            x, y, bw, bh = xywh[i]
            detections.append((
                int(max(0, x)), int(max(0, y)), int(min(w, x + bw)), int(min(h, y + bh)),
                COCO_VEHICLES[int(class_ids[i])], float(confs[i]),
            ))
        return self.tracker.update(detections)
