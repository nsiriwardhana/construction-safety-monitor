def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_inside(inner_box, outer_box):
    cx, cy = box_center(inner_box)
    x1, y1, x2, y2 = outer_box
    return x1 <= cx <= x2 and y1 <= cy <= y2


def classify_worker(person_box, helmet_boxes, vest_boxes):
    has_helmet = any(center_inside(h, person_box) for h in helmet_boxes)
    has_vest = any(center_inside(v, person_box) for v in vest_boxes)

    if has_helmet and has_vest:
        return "safe", []

    missing = []
    if not has_helmet:
        missing.append("helmet")
    if not has_vest:
        missing.append("vest")

    return "unsafe", missing