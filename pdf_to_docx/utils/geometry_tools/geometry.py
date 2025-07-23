from typing import List, Tuple

def get_bounds(block: List[Tuple[float, float]]):
    xs = [pt[0] for pt in block]
    ys = [pt[1] for pt in block]
    return min(xs), max(xs), min(ys), max(ys)

def intersect(blockA: List[Tuple[float, float]], blockB: List[Tuple[float, float]]) -> bool:
    xmin1, xmax1, ymin1, ymax1 = get_bounds(blockA)
    xmin2, xmax2, ymin2, ymax2 = get_bounds(blockB)

    return not (xmax1 < xmin2 or xmin1 > xmax2 or ymax1 < ymin2 or ymin1 > ymax2)

def compute_iou(blockA, blockB) -> float:
    x0 = max(min(p[0] for p in blockA), min(p[0] for p in blockB))
    y0 = max(min(p[1] for p in blockA), min(p[1] for p in blockB))
    x1 = min(max(p[0] for p in blockA), max(p[0] for p in blockB))
    y1 = min(max(p[1] for p in blockA), max(p[1] for p in blockB))

    inter_area = max(0, x1 - x0) * max(0, y1 - y0)
    area1 = (max(p[0] for p in blockA) - min(p[0] for p in blockA)) * (max(p[1] for p in blockA) - min(p[1] for p in blockA))
    area2 = (max(p[0] for p in blockB) - min(p[0] for p in blockB)) * (max(p[1] for p in blockB) - min(p[1] for p in blockB))
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area