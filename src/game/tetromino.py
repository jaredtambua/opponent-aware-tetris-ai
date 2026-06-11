import numpy as np
from typing import Tuple

# SRS spawn orientation (rot = 0)
SHAPES = {
    "I": np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=int),
    "T": np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
    "S": np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=int),
    "Z": np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=int),
    "J": np.array([[1, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
    "L": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
    "O": np.array([[1, 1], [1, 1]], dtype=int),
}


# SRS Wall Kick Tables
# indexing uses state indices {0,1,2,3} == {spawn, R, 180, L}
# for each (from_state, to_state) give a list of (dx, dy) to try in order.
# JLSTZ & T (3x3)
KICKS_JLSTZ = {
    (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
    (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
    (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
    (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
    (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
    (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
}

# I (4x4)
KICKS_I = {
    (0, 1): [(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
    (1, 0): [(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],
    (2, 1): [(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
    (2, 3): [(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
    (3, 0): [(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
    (0, 3): [(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],
}


def rotate_shape_clockwise(shape):
    return np.rot90(shape, -1)


def rotate_shape_counterclockwise(shape):
    return np.rot90(shape, 1)


def build_orientations(shapes):
    orientations = {}

    for label, base in shapes.items():
        rot0 = base
        rot1 = rotate_shape_clockwise(rot0)  # 90 CW
        rot2 = rotate_shape_clockwise(rot1)  # 180
        rot3 = rotate_shape_clockwise(rot2)  # 270 CW
        orientations[label] = [rot0, rot1, rot2, rot3]

    return orientations


def get_shape(label, rotation):
    rotation = int(rotation) % 4
    return ORIENTATIONS[label][rotation]


def get_kick_tb(label):
    if label == "I":
        return KICKS_I
    elif label == "O":
        return None  # O doesn't rotate meaningfully; no kicks.
    else:
        return KICKS_JLSTZ


# build once at import time
ORIENTATIONS = build_orientations(SHAPES)

# bitmask cache for fast collision checks
PIECE_ROW_MASKS = {}

for label, rots in ORIENTATIONS.items():
    PIECE_ROW_MASKS[label] = []
    for shape in rots:
        h, w = shape.shape
        row_masks = []
        for r in range(h):
            m = 0
            for c in range(w):
                if shape[r, c] != 0:
                    m |= 1 << c
            row_masks.append(m)
        PIECE_ROW_MASKS[label].append((row_masks, h, w))


def _try_rotate_to(board, label, x, y, rot, new_rot):
    # O piece: no visible rotation; just validate current spot.
    if label == "O":
        current_shape = get_shape(label, rot)
        ok = board.can_legally_place(current_shape, x, y)
        return (x, y, rot, current_shape, ok)

    target_shape = get_shape(label, new_rot)
    kicks = get_kick_tb(label)

    for dx, dy in kicks[(rot, new_rot)]:
        nx, ny = x + dx, y - dy
        if board.can_legally_place(target_shape, nx, ny):
            return (nx, ny, new_rot, target_shape, True)
    # no offset worked -> refuse rotation.
    return (x, y, rot, get_shape(label, rot), False)


def try_rotate_cw(board, label: str, x, y, rot):
    new_rot = (int(rot) + 1) % 4
    return _try_rotate_to(board, label, x, y, int(rot) % 4, new_rot)


def try_rotate_ccw(board, label: str, x, y, rot):
    new_rot = (int(rot) - 1) % 4
    return _try_rotate_to(board, label, x, y, int(rot) % 4, new_rot)


def _unique_preserve_order(pairs):
    seen = set()
    out = []
    for p in pairs:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def _build_180_kicks_from_90(kicks, rot, new_rot):
    rot = int(rot) % 4
    new_rot = int(new_rot) % 4
    if (rot + 2) % 4 != new_rot:
        return [(0, 0)]  # not a 180 request, safe fallback

    cw1 = (rot + 1) % 4
    ccw1 = (rot - 1) % 4

    candidates = []

    # always try no offset first
    candidates.append((0, 0))

    # CW+CW combined offsets
    if (rot, cw1) in kicks and (cw1, new_rot) in kicks:
        k1 = kicks[(rot, cw1)]
        k2 = kicks[(cw1, new_rot)]
        for dx1, dy1 in k1:
            for dx2, dy2 in k2:
                candidates.append((dx1 + dx2, dy1 + dy2))

    # CCW+CCW combined offsets
    if (rot, ccw1) in kicks and (ccw1, new_rot) in kicks:
        k1 = kicks[(rot, ccw1)]
        k2 = kicks[(ccw1, new_rot)]
        for dx1, dy1 in k1:
            for dx2, dy2 in k2:
                candidates.append((dx1 + dx2, dy1 + dy2))

    return _unique_preserve_order(candidates)


def try_rotate_180(board, label: str, x, y, rot):
    rot = int(rot) % 4
    new_rot = (rot + 2) % 4

    # O piece: no meaningful rotation; just validate current placement
    if label == "O":
        current_shape = get_shape(label, rot)
        ok = board.can_legally_place(current_shape, x, y)
        return (x, y, rot, current_shape, ok)

    target_shape = get_shape(label, new_rot)
    kicks = get_kick_tb(label)

    # Build 180 kick attempts from your existing 90 kick tables
    kick_list_180 = _build_180_kicks_from_90(kicks, rot, new_rot)

    for dx, dy in kick_list_180:
        # board is down-positive
        nx, ny = x + dx, y - dy
        if board.can_legally_place(target_shape, nx, ny):
            return (nx, ny, new_rot, target_shape, True)

    return (x, y, rot, get_shape(label, rot), False)
