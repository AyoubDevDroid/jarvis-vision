"""Application video Jarvis : detection des vehicules, plaques et trafic."""

import argparse
from collections import deque
from pathlib import Path
import sys
import threading
import time

from jarvis_vision.plates import PlateVoter, display, is_valid, normalize
from jarvis_vision.watchlist import Watchlist, notify_telegram
from jarvis_vision.traffic import (
    centered_traffic_region,
    count_vehicles_in_region,
    densest_region_index,
    is_traffic_saturated,
)

VEHICLE_COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (0, 0, 255),
    "bus": (255, 140, 0),
    "truck": (0, 165, 255),
    "bicycle": (0, 220, 255),
}
PLATE_CLASSES = {"car", "motorcycle", "bus", "truck", "van"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HUD Jarvis de detection de vehicules et d'alerte trafic.")
    p.add_argument("--source", default="0", help="Index de webcam, chemin video ou URL RTSP (defaut : 0).")
    p.add_argument("--backend", choices=["ultralytics", "onnx"], default="ultralytics",
                   help="ultralytics (PyTorch, ByteTrack) ou onnx (leger, sans PyTorch).")
    p.add_argument("--model", help="Poids YOLO (defaut : yolo11n.pt, ou yolo11n.onnx en mode onnx).")
    p.add_argument("--ocr-model", default="cct-xs-v2-global-model",
                   help="Modele OCR fast-plate-ocr (ex. european-plates-mobile-vit-v2-model).")
    p.add_argument("--plate-detector", default="yolo-v9-t-384-license-plate-end2end",
                   help="Detecteur de plaques open-image-models.")
    p.add_argument("--output", help="Chemin facultatif pour enregistrer la video annotee.")
    p.add_argument("--threshold", type=int, default=4, help="Vehicules dans la ROI avant alerte (defaut : 4).")
    p.add_argument("--confidence", type=float, default=0.35, help="Confiance minimale YOLO (defaut : 0.35).")
    p.add_argument("--imgsz", type=int, default=640, help="Taille d'entree YOLO (defaut : 640).")
    p.add_argument("--no-plates", action="store_true", help="Desactiver la lecture des plaques.")
    p.add_argument("--ocr-interval", type=int, default=8, help="Images entre deux lectures d'un meme vehicule (defaut : 8).")
    p.add_argument("--max-ocr-per-frame", type=int, default=2, help="Lectures OCR max par image (defaut : 2).")
    p.add_argument("--min-vehicle-width", type=int, default=100, help="Largeur min (px) d'un vehicule pour tenter l'OCR.")
    p.add_argument("--votes", type=int, default=3, help="Lectures identiques pour verrouiller une plaque (defaut : 3).")
    p.add_argument("--any-plate", action="store_true", help="Accepter les plaques non francaises.")
    p.add_argument("--watchlist", help="Base SQLite des vehicules de la flotte a signaler (ex. flotte.db).")
    p.add_argument("--tolerance", type=int, default=1, help="Erreurs OCR tolerees pour reconnaitre un vehicule de la flotte (defaut : 1).")
    p.add_argument("--cooldown", type=float, default=60, help="Secondes avant de re-signaler le meme vehicule (defaut : 60).")
    p.add_argument("--snapshots", help="Dossier ou enregistrer une photo de chaque passage.")
    p.add_argument("--verbose", action="store_true", help="Afficher chaque plaque verrouillee dans la console.")
    p.add_argument("--anonymize", action="store_true", help="Masquer le centre des plaques a l'ecran (pour publier une video).")
    p.add_argument("--no-display", action="store_true", help="Ne pas ouvrir de fenetre.")
    return p


def _source_value(value: str) -> int | str:
    return int(value) if value.isdecimal() else value


def _is_live(source: int | str) -> bool:
    return isinstance(source, int) or str(source).lower().startswith(("rtsp://", "http://", "https://"))


class LatestFrameReader:
    """Lit le flux en continu et ne garde que la derniere image (pas de retard accumule)."""

    def __init__(self, capture):
        self.capture = capture
        self.frame = None
        self.ok = True
        self.lock = threading.Lock()
        self.event = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.ok:
            ok, frame = self.capture.read()
            with self.lock:
                self.ok, self.frame = ok, frame
            self.event.set()

    def read(self):
        self.event.wait()
        self.event.clear()
        with self.lock:
            return self.ok, self.frame


def _vehicle_class_ids(names: dict[int, str] | list[str]) -> list[int]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    return [cid for cid, name in items if str(name).lower() in {*VEHICLE_COLORS, "van"}]


def _read_plate(alpr, crop, strict_fr: bool = False) -> str | None:
    best, best_conf = None, 0.0
    for result in alpr.predict(crop):
        if not result.ocr or not result.ocr.text:
            continue
        plate = normalize(result.ocr.text, strict_fr)
        if is_valid(plate, strict_fr) and result.detection.confidence > best_conf:
            best, best_conf = plate, result.detection.confidence
    return best


def _ascii(text: str) -> str:
    """OpenCV n'affiche pas les accents : 'Vehicule' plutot que 'V??hicule'."""
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def _draw_label(cv2, frame, text, x, y, color):
    text = _ascii(text)
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    top = max(0, y - th - base - 8)
    right = min(frame.shape[1], x + tw + 10)
    cv2.rectangle(frame, (x, top), (right, y), color, -1)
    cv2.putText(frame, text, (x + 5, max(th + 2, y - base - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def run(args: argparse.Namespace) -> int:
    if args.threshold < 1 or args.ocr_interval < 1 or args.max_ocr_per_frame < 1:
        raise ValueError("--threshold, --ocr-interval et --max-ocr-per-frame doivent etre superieurs a 0.")
    if not 0 < args.confidence <= 1:
        raise ValueError("--confidence doit etre compris entre 0 et 1.")

    try:
        import cv2
        if not args.no_plates:
            from fast_alpr import ALPR
        if args.backend == "ultralytics":
            from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Dependances absentes : python -m pip install -r requirements.txt") from error

    if args.backend == "onnx":
        from jarvis_vision.onnx_backend import OnnxVehicleDetector

        onnx_detector = OnnxVehicleDetector(args.model or "yolo11n.onnx", args.imgsz, args.confidence)

        def detect(frame):
            return onnx_detector(frame)
    else:
        yolo = YOLO(args.model or "yolo11n.pt")
        vehicle_ids = _vehicle_class_ids(yolo.names)
        if not vehicle_ids:
            raise ValueError("Le modele YOLO ne contient aucune classe de vehicule reconnue.")

        def detect(frame):
            result = yolo.track(
                frame, persist=True, verbose=False, conf=args.confidence,
                classes=vehicle_ids, imgsz=args.imgsz, tracker="bytetrack.yaml",
            )[0]
            out = []
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                    label = str(yolo.names[int(box.cls.item())]).lower()
                    track_id = int(box.id.item()) if box.id is not None else None
                    out.append((x1, y1, x2, y2, label, float(box.conf.item()), track_id))
            return out

    alpr = None
    if not args.no_plates:
        alpr = ALPR(detector_model=args.plate_detector, ocr_model=args.ocr_model)
    voter = PlateVoter(votes_to_lock=args.votes)
    strict_fr = not args.any_plate
    watchlist = Watchlist(args.watchlist, args.tolerance, args.cooldown) if args.watchlist else None
    if watchlist is not None:
        print(f"Flotte : {len(watchlist.vehicles)} vehicule(s) surveille(s).")
    if args.snapshots:
        Path(args.snapshots).mkdir(parents=True, exist_ok=True)
    fleet_hits: dict[int, str] = {}       # track_id -> libelle du vehicule de la flotte
    banner_text, banner_until = "", 0.0

    source = _source_value(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Impossible d'ouvrir la source video : {args.source}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    reader = LatestFrameReader(capture) if _is_live(source) else capture

    writer = None
    frame_number = 0
    recent_counts: deque[int] = deque(maxlen=10)  # lissage anti-clignotement
    fps, last_time = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                break
            frame = frame.copy() if reader is not capture else frame

            frame_number += 1
            height, width = frame.shape[:2]
            region = centered_traffic_region(width, height)
            vehicles = detect(frame)

            # 1) OCR sur l'image propre, AVANT de dessiner (plus besoin de copier l'image)
            if alpr is not None:
                candidates = [
                    v for v in vehicles
                    if v[6] is not None
                    and v[4] in PLATE_CLASSES
                    and voter.needs_read(v[6])
                    and (v[2] - v[0]) >= args.min_vehicle_width
                    and (frame_number + v[6]) % args.ocr_interval == 0  # etale la charge
                ]
                candidates.sort(key=lambda v: (v[2] - v[0]) * (v[3] - v[1]), reverse=True)
                for x1, y1, x2, y2, _, _, track_id in candidates[: args.max_ocr_per_frame]:
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    if crop.size:
                        plate = _read_plate(alpr, crop, strict_fr)
                        if not plate:
                            continue
                        voter.add(track_id, plate)
                        found = voter.get(track_id)
                        if not found or not found[1]:
                            continue  # pas encore verrouillee
                        locked_plate = found[0]
                        if args.verbose:
                            print(f"[{frame_number}] Plaque verrouillee : {locked_plate} (vehicule {track_id})")
                        hit = watchlist.match(locked_plate) if watchlist else None
                        if hit is None:
                            continue
                        fleet_plate, fleet_label = hit
                        fleet_hits[track_id] = fleet_label
                        banner_text = f"FLOTTE : {fleet_label} ({display(fleet_plate, args.anonymize, strict_fr)})"
                        banner_until = time.time() + 4
                        snapshot = ""
                        if args.snapshots:
                            stamp = time.strftime("%Y%m%d-%H%M%S")
                            snapshot = str(Path(args.snapshots) / f"{stamp}_{fleet_plate}.jpg")
                            cv2.imwrite(snapshot, crop)
                        if watchlist.record(fleet_plate, fleet_label, locked_plate, str(args.source), snapshot):
                            message = f"Jarvis : {fleet_label} ({fleet_plate}) detecte a {time.strftime('%H:%M:%S')}"
                            print(message)
                            notify_telegram(message)

            # 2) Dessin
            for x1, y1, x2, y2, label, conf, track_id in vehicles:
                color = VEHICLE_COLORS.get(label, (180, 0, 180))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                _draw_label(cv2, frame, f"{label} {conf:.0%}", x1, y1, color)
                if track_id is None:
                    continue
                voter.seen(track_id, frame_number)
                found = voter.get(track_id)
                if found:
                    plate, locked = found
                    text = f"PLAQUE: {display(plate, args.anonymize, strict_fr)}" + ("" if locked else " ?")
                    _draw_label(cv2, frame, text,
                                min(max(0, x2 + 5), max(0, width - 200)),
                                min(max(y1 + 30, 20), height - 2),
                                (30, 120, 30) if locked else (30, 30, 30))
                if track_id in fleet_hits:  # vehicule de la flotte : cadre jaune epais
                    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (0, 230, 255), 5)
                    _draw_label(cv2, frame, f"FLOTTE: {fleet_hits[track_id]}",
                                x1, min(height - 2, y2 + 24), (0, 140, 220))
            voter.purge(frame_number)
            for tid in [t for t in fleet_hits if t not in voter.last_seen]:
                del fleet_hits[tid]

            traffic_boxes = [(v[0], v[1], v[2], v[3]) for v in vehicles]
            recent_counts.append(count_vehicles_in_region(traffic_boxes, region))
            vehicle_count = round(sum(recent_counts) / len(recent_counts))
            saturated = is_traffic_saturated(vehicle_count, args.threshold)

            cv2.rectangle(frame, (region.x1, region.y1), (region.x2, region.y2),
                          (0, 0, 255) if saturated else (255, 200, 0), 2)
            alert = "RALENTIR - GARDER SES DISTANCES" if saturated else "CIRCULATION LOCALE FLUIDE"
            hud_color = (0, 0, 200) if saturated else (40, 90, 40)
            cv2.rectangle(frame, (10, 10), (min(width - 10, 470), 72), hud_color, -1)
            cv2.putText(frame, f"ZONE ROI : {vehicle_count} vehicule(s)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, alert, (20, 59),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)

            if saturated:
                dense = densest_region_index(traffic_boxes, region)
                if dense is not None:
                    section_width = (region.x2 - region.x1) / 3
                    ax = int(region.x1 + (dense + 0.5) * section_width)
                    cv2.arrowedLine(frame, (ax, max(75, region.y1 - 58)), (ax, region.y1 - 8),
                                    (0, 0, 255), 6, cv2.LINE_AA, tipLength=0.4)
                    cv2.putText(frame, "TRAFIC DENSE", (max(5, ax - 75), max(65, region.y1 - 65)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

            if time.time() < banner_until:
                banner_text = _ascii(banner_text)
                (tw, _), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                bx = max(10, (width - tw) // 2 - 15)
                cv2.rectangle(frame, (bx, height - 60), (min(width - 10, bx + tw + 30), height - 15), (0, 200, 255), -1)
                cv2.putText(frame, banner_text, (bx + 15, height - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
            last_time = now
            cv2.putText(frame, f"{fps:.0f} FPS", (width - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            if args.output and writer is None:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                src_fps = capture.get(cv2.CAP_PROP_FPS)
                src_fps = src_fps if src_fps > 0 else 30.0
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Impossible de creer la video : {output_path}")
            if writer is not None:
                writer.write(frame)
            if not args.no_display:
                cv2.imshow("Jarvis - HUD Moto", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if isinstance(reader, LatestFrameReader):
            reader.ok = False
        capture.release()
        if writer is not None:
            writer.release()
        if watchlist is not None:
            watchlist.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (RuntimeError, ValueError) as error:
        print(f"Erreur Jarvis : {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
