"""Lecture fiable des plaques : nettoyage, validation FR et vote multi-images."""

import re
from collections import Counter, defaultdict

# SIV : AA-123-AA, sans les lettres I, O et U
SIV = re.compile(r"^[A-HJ-NP-TV-Z]{2}\d{3}[A-HJ-NP-TV-Z]{2}$")
FNI = re.compile(r"^\d{1,4}[A-Z]{1,3}(\d{2}|2A|2B)$")  # 123 ABC 69 (ancien)

# Confusions OCR frequentes (le SIV n'utilise jamais I, O, U)
TO_LETTER = {"0": "D", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1",
            "Z": "2", "A": "4", "S": "5", "G": "6", "T": "7", "B": "8"}


def normalize(text: str, strict_fr: bool = True) -> str:
    """Majuscules, sans tirets/espaces ; en mode FR, corrige 0/D, 8/B... selon la position SIV."""
    raw = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not strict_fr or len(raw) != 7:
        return raw
    fixed = [
        TO_DIGIT.get(c, c) if 2 <= i <= 4 else TO_LETTER.get(c, c)
        for i, c in enumerate(raw)
    ]
    candidate = "".join(fixed)
    return candidate if SIV.match(candidate) else raw


def is_valid(plate: str, strict_fr: bool = True) -> bool:
    if not strict_fr:
        return 4 <= len(plate) <= 10
    return bool(SIV.match(plate) or FNI.match(plate))


def display(plate: str, anonymize: bool = False, strict_fr: bool = True) -> str:
    """Formate la plaque (tirets SIV seulement en mode FR) ; anonymize masque le centre."""
    siv = strict_fr and bool(SIV.match(plate))
    if anonymize and len(plate) > 4:
        plate = plate[:2] + "*" * (len(plate) - 4) + plate[-2:]
    return f"{plate[:2]}-{plate[2:5]}-{plate[5:]}" if siv else plate


class PlateVoter:
    """Accumule les lectures par vehicule suivi et verrouille la plaque majoritaire."""

    def __init__(self, votes_to_lock: int = 3, max_age: int = 90):
        self.votes_to_lock = votes_to_lock
        self.max_age = max_age
        self.reads: dict[int, Counter] = defaultdict(Counter)
        self.locked: dict[int, str] = {}
        self.last_seen: dict[int, int] = {}

    def seen(self, track_id: int, frame_number: int) -> None:
        self.last_seen[track_id] = frame_number

    def needs_read(self, track_id: int) -> bool:
        return track_id not in self.locked

    def add(self, track_id: int, plate: str) -> None:
        if track_id in self.locked:
            return
        self.reads[track_id][plate] += 1
        best, count = self.reads[track_id].most_common(1)[0]
        if count >= self.votes_to_lock:
            self.locked[track_id] = best
            self.reads.pop(track_id, None)

    def get(self, track_id: int) -> tuple[str, bool] | None:
        """Retourne (plaque, verrouillee) ou None."""
        if track_id in self.locked:
            return self.locked[track_id], True
        if self.reads.get(track_id):
            return self.reads[track_id].most_common(1)[0][0], False
        return None

    def purge(self, frame_number: int) -> None:
        """Oublie les vehicules disparus depuis max_age images (evite la fuite memoire)."""
        stale = [t for t, f in self.last_seen.items() if frame_number - f > self.max_age]
        for track_id in stale:
            self.last_seen.pop(track_id, None)
            self.reads.pop(track_id, None)
            self.locked.pop(track_id, None)
