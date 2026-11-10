"""Base des vehicules de la flotte (SQLite) : liste, passages et notifications."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import urllib.parse
import urllib.request

from jarvis_vision.plates import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS vehicles (
    plate TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    note TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS sightings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    label TEXT NOT NULL,
    read_as TEXT NOT NULL,
    seen_at TEXT NOT NULL,
    source TEXT DEFAULT '',
    snapshot TEXT DEFAULT ''
);
"""


def _distance(a: str, b: str) -> int:
    """Distance de Levenshtein (nombre de caracteres a changer)."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class Watchlist:
    def __init__(self, path: str | Path, tolerance: int = 1, cooldown_s: float = 60.0):
        self.path = str(path)
        self.tolerance = tolerance
        self.cooldown_s = cooldown_s
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.last_alert: dict[str, float] = {}
        self.reload()

    # --- gestion de la liste -------------------------------------------------
    def reload(self) -> None:
        self.vehicles = dict(self.db.execute("SELECT plate, label FROM vehicles"))

    def add(self, plate: str, label: str, note: str = "") -> str:
        plate = normalize(plate, strict_fr=False)
        self.db.execute("INSERT OR REPLACE INTO vehicles VALUES (?, ?, ?)", (plate, label, note))
        self.db.commit()
        self.reload()
        return plate

    def remove(self, plate: str) -> bool:
        cur = self.db.execute("DELETE FROM vehicles WHERE plate = ?", (normalize(plate, strict_fr=False),))
        self.db.commit()
        self.reload()
        return cur.rowcount > 0

    # --- detection -----------------------------------------------------------
    def match(self, plate: str) -> tuple[str, str] | None:
        """Retourne (plaque_flotte, libelle) si la plaque lue correspond, sinon None.

        Tolere `tolerance` caractere(s) d'erreur OCR, mais seulement si une seule
        plaque de la flotte est assez proche (evite les confusions).
        """
        if plate in self.vehicles:
            return plate, self.vehicles[plate]
        if self.tolerance <= 0:
            return None
        close = [p for p in self.vehicles
                 if abs(len(p) - len(plate)) <= self.tolerance
                 and _distance(p, plate) <= self.tolerance]
        return (close[0], self.vehicles[close[0]]) if len(close) == 1 else None

    def record(self, plate: str, label: str, read_as: str, source: str = "", snapshot: str = "") -> bool:
        """Enregistre un passage. Retourne False si deja signale recemment (anti-doublon)."""
        now = time.time()
        if now - self.last_alert.get(plate, 0) < self.cooldown_s:
            return False
        self.last_alert[plate] = now
        self.db.execute(
            "INSERT INTO sightings (plate, label, read_as, seen_at, source, snapshot) VALUES (?, ?, ?, ?, ?, ?)",
            (plate, label, read_as, datetime.now().isoformat(timespec="seconds"), source, snapshot),
        )
        self.db.commit()
        return True

    def close(self) -> None:
        self.db.close()


def notify_telegram(text: str, photo_path: str = "") -> None:
    """Envoie une alerte Telegram en arriere-plan si TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID sont definis."""
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    def send():
        try:
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=10)
        except Exception as error:  # une alerte ratee ne doit jamais couper la video
            print(f"Telegram : {error}", file=sys.stderr)

    threading.Thread(target=send, daemon=True).start()


# --- ligne de commande : python -m jarvis_vision.watchlist ... ----------------
def main() -> int:
    p = argparse.ArgumentParser(description="Gerer la liste des vehicules de la flotte.")
    p.add_argument("--db", default="flotte.db", help="Fichier SQLite (defaut : flotte.db).")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="Ajouter un vehicule")
    a.add_argument("plate"); a.add_argument("label"); a.add_argument("--note", default="")
    r = sub.add_parser("remove", help="Retirer un vehicule"); r.add_argument("plate")
    sub.add_parser("list", help="Lister les vehicules")
    s = sub.add_parser("passages", help="Derniers passages"); s.add_argument("-n", type=int, default=20)
    e = sub.add_parser("export", help="Exporter les passages en CSV"); e.add_argument("csv_path")
    i = sub.add_parser("import", help="Importer des vehicules depuis un CSV (plaque;libelle;note)")
    i.add_argument("csv_path")
    args = p.parse_args()

    wl = Watchlist(args.db)
    if args.cmd == "add":
        print(f"Ajoute : {wl.add(args.plate, args.label, args.note)} ({args.label})")
    elif args.cmd == "remove":
        print("Retire." if wl.remove(args.plate) else "Plaque introuvable.")
    elif args.cmd == "list":
        for plate, label, note in wl.db.execute("SELECT * FROM vehicles ORDER BY label"):
            print(f"{plate:<12} {label:<25} {note}")
    elif args.cmd == "passages":
        rows = wl.db.execute(
            "SELECT seen_at, plate, label, read_as FROM sightings ORDER BY id DESC LIMIT ?", (args.n,))
        for seen_at, plate, label, read_as in rows:
            print(f"{seen_at}  {plate:<12} {label:<25} (lu : {read_as})")
    elif args.cmd == "export":
        rows = wl.db.execute("SELECT seen_at, plate, label, read_as, source, snapshot FROM sightings ORDER BY id")
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["date", "plaque", "vehicule", "lu_comme", "source", "photo"])
            w.writerows(rows)
        print(f"Exporte : {args.csv_path}")
    elif args.cmd == "import":
        n = 0
        with open(args.csv_path, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter=";"):
                if len(row) >= 2 and row[0].strip() and row[0].strip().lower() != "plaque":
                    wl.add(row[0], row[1], row[2] if len(row) > 2 else ""); n += 1
        print(f"{n} vehicule(s) importe(s).")
    wl.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
