import unittest

from jarvis_vision.plates import PlateVoter, display, is_valid, normalize


class PlateTests(unittest.TestCase):
    def test_normalize_fixes_siv_confusions(self):
        self.assertEqual(normalize("ab-l23-cd"), "AB123CD")
        self.assertEqual(normalize("A8 I23 C0"), "AB123CD")

    def test_foreign_plates_untouched(self):
        # plaque vietnamienne : aucune correction ni mise en forme francaise
        self.assertEqual(normalize("51G-123.45", strict_fr=False), "51G12345")
        self.assertEqual(normalize("50H6298", strict_fr=False), "50H6298")
        self.assertEqual(display("51G12345", strict_fr=False), "51G12345")
        self.assertEqual(display("51G12345", anonymize=True, strict_fr=False), "51****45")

    def test_validation(self):
        self.assertTrue(is_valid("AB123CD"))
        self.assertTrue(is_valid("123ABC69"))
        self.assertFalse(is_valid("AO123CD"))  # O interdit en SIV
        self.assertFalse(is_valid("HELLO"))

    def test_display(self):
        self.assertEqual(display("AB123CD"), "AB-123-CD")
        self.assertEqual(display("AB123CD", anonymize=True), "AB-***-CD")

    def test_vote_locks_majority_and_purges(self):
        voter = PlateVoter(votes_to_lock=3, max_age=10)
        for plate in ["AB123CD", "AB128CD", "AB123CD", "AB123CD"]:
            voter.add(1, plate)
        self.assertEqual(voter.get(1), ("AB123CD", True))
        self.assertFalse(voter.needs_read(1))
        voter.seen(1, 0)
        voter.purge(50)
        self.assertIsNone(voter.get(1))


if __name__ == "__main__":
    unittest.main()
