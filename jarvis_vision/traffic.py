"""Helpers purs pour l'analyse de la zone de circulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficRegion:
    x1: int
    y1: int
    x2: int
    y2: int

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


def centered_traffic_region(width: int, height: int) -> TrafficRegion:
    """Construit une zone centrale devant la camera, exprimee en pixels."""
    return TrafficRegion(
        x1=int(width * 0.25),
        y1=int(height * 0.45),
        x2=int(width * 0.75),
        y2=int(height * 0.95),
    )


def count_vehicles_in_region(
    boxes: list[tuple[float, float, float, float]],
    region: TrafficRegion,
) -> int:
    """Compte les vehicules dont le centre tombe dans la zone de circulation."""
    return sum(
        region.contains((x1 + x2) / 2, (y1 + y2) / 2)
        for x1, y1, x2, y2 in boxes
    )


def densest_region_index(
    boxes: list[tuple[float, float, float, float]],
    region: TrafficRegion,
    columns: int = 3,
) -> int | None:
    """Returns the busiest horizontal section without suggesting a maneuver."""
    if columns < 1:
        raise ValueError("Le nombre de sections doit etre superieur ou egal a 1.")

    counts = [0] * columns
    region_width = region.x2 - region.x1
    if region_width <= 0:
        raise ValueError("La region de circulation doit avoir une largeur positive.")

    for x1, y1, x2, y2 in boxes:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        if region.contains(center_x, center_y):
            index = min(int((center_x - region.x1) / region_width * columns), columns - 1)
            counts[index] += 1

    if not any(counts):
        return None
    return max(range(columns), key=counts.__getitem__)


def is_traffic_saturated(vehicle_count: int, threshold: int) -> bool:
    if threshold < 1:
        raise ValueError("Le seuil de saturation doit etre superieur ou egal a 1.")
    return vehicle_count >= threshold
