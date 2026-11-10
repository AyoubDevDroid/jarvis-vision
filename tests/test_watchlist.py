import tempfile
import unittest
from pathlib import Path

from jarvis_vision.watchlist import Watchlist


class WatchlistTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wl = Watchlist(Path(self.tmp.name) / "flotte.db", tolerance=1, cooldown_s=60)
        self.wl.add("AB-123-CD", "Ambulance 1")
        self.wl.add("EF-456-GH", "VSL 2")

    def tearDown(self):
        self.wl.close()
        self.tmp.cleanup()

    def test_exact_match(self):
        self.assertEqual(self.wl.match("AB123CD"), ("AB123CD", "Ambulance 1"))

    def test_tolerates_one_ocr_error(self):
        self.assertEqual(self.wl.match("AB128CD"), ("AB123CD", "Ambulance 1"))

    def test_rejects_unknown_plate(self):
        self.assertIsNone(self.wl.match("XY999ZZ"))

    def test_ambiguous_match_is_rejected(self):
        self.wl.add("AB124CD", "Ambulance 3")
        self.assertIsNone(self.wl.match("AB129CD"))  # proche de 2 plaques : on ne devine pas

    def test_record_has_cooldown(self):
        self.assertTrue(self.wl.record("AB123CD", "Ambulance 1", "AB123CD"))
        self.assertFalse(self.wl.record("AB123CD", "Ambulance 1", "AB123CD"))
        count = self.wl.db.execute("SELECT COUNT(*) FROM sightings").fetchone()[0]
        self.assertEqual(count, 1)

    def test_remove(self):
        self.assertTrue(self.wl.remove("EF-456-GH"))
        self.assertIsNone(self.wl.match("EF456GH"))


if __name__ == "__main__":
    unittest.main()
