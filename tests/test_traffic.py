import unittest
from types import SimpleNamespace

from jarvis_vision.app import _read_plate, _source_value, _vehicle_class_ids
from jarvis_vision.traffic import (
    TrafficRegion,
    centered_traffic_region,
    count_vehicles_in_region,
    densest_region_index,
    is_traffic_saturated,
)


class TrafficAnalysisTests(unittest.TestCase):
    def test_centered_region_uses_expected_frame_proportions(self):
        self.assertEqual(
            centered_traffic_region(1280, 720),
            TrafficRegion(320, 324, 960, 684),
        )

    def test_counts_only_vehicle_centers_inside_region(self):
        region = TrafficRegion(10, 20, 90, 80)
        boxes = [
            (10, 20, 30, 40),
            (80, 60, 100, 90),
            (0, 0, 8, 8),
        ]
        self.assertEqual(count_vehicles_in_region(boxes, region), 2)

    def test_saturation_reaches_threshold(self):
        self.assertFalse(is_traffic_saturated(3, 4))
        self.assertTrue(is_traffic_saturated(4, 4))

    def test_saturation_threshold_must_be_positive(self):
        with self.assertRaises(ValueError):
            is_traffic_saturated(1, 0)

    def test_densest_section_identifies_where_traffic_is_concentrated(self):
        region = TrafficRegion(0, 0, 300, 100)
        boxes = [
            (10, 10, 30, 30),
            (30, 10, 50, 30),
            (50, 10, 70, 30),
            (220, 10, 240, 30),
        ]
        self.assertEqual(densest_region_index(boxes, region), 0)
        self.assertIsNone(densest_region_index([], region))

    def test_densest_section_rejects_invalid_column_count(self):
        with self.assertRaises(ValueError):
            densest_region_index([], TrafficRegion(0, 0, 100, 100), columns=0)

    def test_vehicle_class_ids_include_supported_coco_vehicles(self):
        names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
        self.assertEqual(_vehicle_class_ids(names), [2, 3, 5, 7, 1])

    def test_reads_highest_confidence_plate_result(self):
        alpr = SimpleNamespace(
            predict=lambda _crop: [
                SimpleNamespace(
                    detection=SimpleNamespace(confidence=0.6),
                    ocr=SimpleNamespace(text=" ab12 "),
                ),
                SimpleNamespace(
                    detection=SimpleNamespace(confidence=0.9),
                    ocr=SimpleNamespace(text="cd34"),
                ),
            ]
        )
        self.assertEqual(_read_plate(alpr, object()), "CD34")

    def test_missing_plate_result_returns_none(self):
        alpr = SimpleNamespace(
            predict=lambda _crop: [
                SimpleNamespace(
                    detection=SimpleNamespace(confidence=0.99),
                    ocr=SimpleNamespace(text=""),
                )
            ]
        )
        self.assertIsNone(_read_plate(alpr, object()))

    def test_source_value_accepts_camera_index_or_video_path(self):
        self.assertEqual(_source_value("0"), 0)
        self.assertEqual(_source_value("C:\\videos\\ride.mp4"), "C:\\videos\\ride.mp4")


if __name__ == "__main__":
    unittest.main()
