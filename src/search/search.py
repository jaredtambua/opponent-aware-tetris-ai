from collections import deque

from src.game.tetromino import (
    PIECE_ROW_MASKS,
    KICKS_I,
    KICKS_JLSTZ,
    _build_180_kicks_from_90,
)
from src.game.bitboard import BitBoard


def _kick_table(label):
    if label == "I":
        return KICKS_I
    return KICKS_JLSTZ


def _settle_down(bb, label, x, y, rot):
    y = int(y)
    while bb.can_place(label, rot, x, y + 1):
        y += 1
    return y


def _try_rotate_to_bit(bb: "BitBoard", label, x, y, rot, new_rot):
    rot = int(rot) % 4
    new_rot = int(new_rot) % 4

    # O piece: no meaningful rotation; just validate current spot.
    if label == "O":
        ok = bb.can_place(label, rot, x, y)
        return x, y, rot, ok

    kicks = _kick_table(label)

    # for 90-degree turns, use the table directly.
    if (rot, new_rot) in kicks:
        kick_list = kicks[(rot, new_rot)]
    else:
        kick_list = [(0, 0)]

    for dx, dy in kick_list:
        # dy is "up-positive" in SRS; our y is "down-positive", so subtract.
        nx, ny = x + dx, y - dy
        if bb.can_place(label, new_rot, nx, ny):
            return nx, ny, new_rot, True

    return x, y, rot, False


def _try_rotate_180_bit(bb: "BitBoard", label, x, y, rot):
    rot = int(rot) % 4
    new_rot = (rot + 2) % 4

    if label == "O":
        ok = bb.can_place(label, rot, x, y)
        return x, y, rot, ok

    kicks = _kick_table(label)
    kick_list = _build_180_kicks_from_90(kicks, rot, new_rot)

    for dx, dy in kick_list:
        nx, ny = x + dx, y - dy
        if bb.can_place(label, new_rot, nx, ny):
            return nx, ny, new_rot, True

    return x, y, rot, False


def neighbors_no_down_bit(bb: "BitBoard", label, x, y, rot):
    rot = int(rot) % 4
    x, y = int(x), int(y)

    # left / right
    if bb.can_place(label, rot, x - 1, y):
        yield (x - 1, y, rot)
    if bb.can_place(label, rot, x + 1, y):
        yield (x + 1, y, rot)

    # rotations
    nx, ny, nrot, ok = _try_rotate_to_bit(bb, label, x, y, rot, (rot + 1) % 4)
    if ok:
        yield (int(nx), int(ny), int(nrot) % 4)

    nx, ny, nrot, ok = _try_rotate_to_bit(bb, label, x, y, rot, (rot - 1) % 4)
    if ok:
        yield (int(nx), int(ny), int(nrot) % 4)

    nx, ny, nrot, ok = _try_rotate_180_bit(bb, label, x, y, rot)
    if ok:
        yield (int(nx), int(ny), int(nrot) % 4)


# spin detection
def _cell_filled_or_wall(bb: "BitBoard", x, y):
    # outside the board counts as filled (wall/corner behaviour)
    if x < 0 or x >= bb.cols or y < 0 or y >= bb.rows:
        return True
    return ((bb.rows_bits[y] >> x) & 1) == 1


def tspin_corners_filled_bit(bb_before: "BitBoard", cx, cy):
    corners = (
        (cx - 1, cy - 1),
        (cx + 1, cy - 1),
        (cx - 1, cy + 1),
        (cx + 1, cy + 1),
    )
    return sum(1 for (x, y) in corners if _cell_filled_or_wall(bb_before, x, y))


def has_block_above_piece_bit(bb, label, rot, x, y):
    row_masks, h, _ = PIECE_ROW_MASKS[label][rot]

    for r in range(h):
        rm = row_masks[r]
        if rm == 0:
            continue

        by = y + r
        if by < 0:
            continue

        bits = rm
        while bits:
            lsb = bits & -bits
            local_x = lsb.bit_length() - 1
            bx = x + local_x

            for yy in range(0, by):
                if (bb.rows_bits[yy] >> bx) & 1:
                    return True

            bits -= lsb

    return False


def tspin_nose_front_empty_bit(bb: "BitBoard", left_x, top_y, rot):
    rot = int(rot) % 4
    if rot == 1:
        tx, ty = (left_x + 1) + 2, (top_y + 1)  # pivot + (2, 0)
    elif rot == 3:
        tx, ty = (left_x + 1) - 2, (top_y + 1)  # pivot + (-2, 0)
    else:
        return False  # only care about 1 and 3

    if tx < 0 or tx >= bb.cols or ty < 0 or ty >= bb.rows:
        return False

    return ((bb.rows_bits[ty] >> tx) & 1) == 0


def is_spinned_bit(bb: "BitBoard", label, x, y, rot):
    if label == "O":
        return False

    rot = int(rot) % 4
    x = int(x)
    y = int(y)

    can_left = bb.can_place(label, rot, x - 1, y)
    can_right = bb.can_place(label, rot, x + 1, y)
    can_nudge = can_left or can_right

    can_drop = bb.can_place(label, rot, x, y + 1)

    blocked_above = has_block_above_piece_bit(bb, label, rot, x, y)

    return (not can_nudge) and (not can_drop) and blocked_above


def tspin_attack_type_on_lock_bit(bb_before, label, left_x, top_y, rot, cleared):
    if label != "T" or cleared <= 0:
        return None

    cx = left_x + 1
    cy = top_y + 1

    filled = tspin_corners_filled_bit(bb_before, cx, cy)
    if filled < 3:
        return None

    if cleared == 3:
        return "tspin_triple" if filled == 4 else None

    nose_empty = tspin_nose_front_empty_bit(bb_before, left_x, top_y, rot)
    no_blocks_above = not has_block_above_piece_bit(
        bb_before, "T", int(rot) % 4, left_x, top_y
    )

    is_mini = (filled == 3) and no_blocks_above and nose_empty

    if cleared == 1:
        if is_mini:
            return "tspin_mini_single"
        return "tspin_single" if (filled == 3 and not no_blocks_above) else None

    if cleared == 2:
        if is_mini:
            return "tspin_mini_double"
        return "tspin_double" if (filled == 3 and not no_blocks_above) else None

    return None


def enumerate_hard_drop_landings(bb, label):
    out = set()

    x_min = -4
    x_max = bb.cols + 4

    for rot in range(4):
        row_masks, h, _w = PIECE_ROW_MASKS[label][rot]
        if h == 0:
            continue

        for x in range(x_min, x_max + 1):
            y = -h

            if not bb.can_place(label, rot, x, y):
                continue

            y = _settle_down(bb, label, x, y, rot)
            out.add((rot, x, y))

    return sorted(out)


# BFS all reachable landing end-states, but skip intermediate y values
def enumerate_all_landings_bit(bb, label, start_x, start_y, start_rot):
    rot0 = int(start_rot) % 4
    x0, y0 = int(start_x), int(start_y)

    # start must be legal
    if not bb.can_place(label, rot0, x0, y0):
        return []

    # canonicalise start by dropping it
    y0 = _settle_down(bb, label, x0, y0, rot0)
    start = (x0, y0, rot0)

    q = deque([start])
    visited = {start}
    landings = set()

    while q:
        x, y, rot = q.popleft()

        if not bb.can_place(label, rot, x, y + 1):
            spinned = is_spinned_bit(bb, label, x, y, rot)
            landings.add((x, y, rot, spinned))

        for nx, ny, nrot in neighbors_no_down_bit(bb, label, x, y, rot):
            # settle after move
            ny = _settle_down(bb, label, nx, ny, nrot)
            child = (int(nx), int(ny), int(nrot) % 4)

            if child in visited:
                continue
            if bb.can_place(label, child[2], child[0], child[1]):
                visited.add(child)
                q.append(child)

    return sorted(landings)


# ---------------- ALL HEURISTIC SEARCH HELPERS --------------------


def column_heights(bb):
    heights = [0] * bb.cols
    for c in range(bb.cols):
        for r in range(bb.rows):
            if (bb.rows_bits[r] >> c) & 1:
                heights[c] = bb.rows - r
                break
    return heights


def count_holes(bb):
    holes = 0
    for x in range(bb.cols):
        seen_block = False
        for y in range(bb.rows):
            filled = (bb.rows_bits[y] >> x) & 1
            if filled:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def aggregate_height(bb):
    return sum(column_heights(bb))


def open_hole_count(bb):
    heights = column_heights(bb)
    top_filled_y = []
    for h in heights:
        if h == 0:
            top_filled_y.append(float("inf"))
        else:
            top_filled_y.append(bb.rows - h)

    count = 0

    for x in range(bb.cols):
        seen_block = False
        for y in range(bb.rows):
            filled = ((bb.rows_bits[y] >> x) & 1) == 1

            if filled:
                seen_block = True
                continue

            if not seen_block:
                continue  # not a hole unless something is above

            open_left = (
                x - 2 >= 0 and top_filled_y[x - 1] < y and top_filled_y[x - 2] < y
            )
            open_right = (
                x + 2 < bb.cols and top_filled_y[x + 1] < y and top_filled_y[x + 2] < y
            )

            if open_left or open_right:
                count += 1

    return count


def tsd_detector(bb, heights=None):
    if heights is None:
        heights = column_heights(bb)

    rows = bb.rows
    count = 0

    for x in range(1, bb.cols - 1):
        hL = heights[x - 1]
        hR = heights[x + 1]
        if hL != hR:
            continue

        if hL == 0:
            continue

        top_y = rows - hL

        if top_y < 2:
            continue

        yy = top_y - 2
        overhang_left = ((bb.rows_bits[yy] >> (x - 1)) & 1) == 1
        overhang_right = ((bb.rows_bits[yy] >> (x + 1)) & 1) == 1

        if overhang_left or overhang_right:
            count += 1

    return count


def holes_and_depth(bb, heights):
    holes = 0
    depth = 0.0
    rows = bb.rows

    for c in range(bb.cols):
        h = heights[c]
        if h == 0:
            continue
        top = rows - h
        for r in range(top, rows):
            filled = (bb.rows_bits[r] >> c) & 1
            if filled == 0:
                holes += 1
                depth += r - top + 1

    return holes, depth


def bumpiness(bb, heights=None):
    if heights is None:
        heights = column_heights(bb)

    min_h = min(heights)
    well = heights.index(min_h)  # deepest well column (leftmost tie)

    bump = 0
    for i in range(len(heights) - 1):
        if i == well or i + 1 == well:
            continue
        bump += abs(heights[i] - heights[i + 1])
    return bump


def non_deepest_well_penalty(bb, heights=None, min_depth=2):
    if heights is None:
        heights = column_heights(bb)

    n = len(heights)
    if n < 2:
        return 0

    # columns to avoid as intended well when breaking ties, if possible.
    avoid_tiebreak_cols = {1, 8}

    min_height = min(heights)
    min_cols = [i for i, h in enumerate(heights) if h == min_height]

    preferred_min_cols = [i for i in min_cols if i not in avoid_tiebreak_cols]
    if preferred_min_cols:
        intended_well = max(preferred_min_cols)
    else:
        intended_well = max(min_cols)

    def well_depth_excluding_intended(i):
        if i == intended_well:
            return 0

        neighbors = []

        left = i - 1
        right = i + 1

        if left >= 0 and left != intended_well:
            neighbors.append(heights[left])
        if right < n and right != intended_well:
            neighbors.append(heights[right])

        if not neighbors:
            return 0

        return max(0, min(neighbors) - heights[i])

    penalty = 0

    for i in range(n):
        if i == intended_well:
            continue

        depth = well_depth_excluding_intended(i)
        if depth >= min_depth:
            penalty += depth

    return penalty


def placement_exceeds_top(label, landing):
    x, y, rot = landing
    row_masks, h, _w = PIECE_ROW_MASKS[label][int(rot) % 4]

    for r, row_mask in enumerate(row_masks):
        if row_mask == 0:
            continue
        if y + r < 0:
            return True

    return False


def buried_holes(rows_bits, rows, cols, col_heights):
    score = 0

    for c in range(cols):
        h = col_heights[c]
        if h == 0:
            continue

        top_r = rows - h
        seen_filled = False
        highest_hole_r = None

        # find highest hole in this column
        for r in range(top_r, rows):
            filled = (rows_bits[r] >> c) & 1
            if filled:
                seen_filled = True
            elif seen_filled:
                highest_hole_r = r
                break

        if highest_hole_r is None:
            continue

        # shallower hole -> larger penalty
        shallow_weight = rows - highest_hole_r

        # count filled cells above the hole within the stack region
        covered_cells = 0
        for r in range(top_r, highest_hole_r):
            if (rows_bits[r] >> c) & 1:
                covered_cells += 1

        score += shallow_weight * covered_cells

    return score


def row_transitions(bb):
    rows, cols = bb.rows, bb.cols
    trans = 0
    for r in range(rows):
        prev = 1  # left wall treated as filled
        row_bits = bb.rows_bits[r]
        for c in range(cols):
            cur = 1 if ((row_bits >> c) & 1) else 0
            if cur != prev:
                trans += 1
            prev = cur
        if prev == 0:
            trans += 1  # right wall
    return trans


def column_holes_via_transitions(bb):
    holes_per_col = []

    for x in range(bb.cols):
        transitions = 0

        prev = 1 if ((bb.rows_bits[bb.rows - 1] >> x) & 1) else 0

        for y in range(bb.rows - 2, -1, -1):
            cur = 1 if ((bb.rows_bits[y] >> x) & 1) else 0
            if cur != prev:
                transitions += 1
            prev = cur

        holes_per_col.append(transitions // 2)

    return holes_per_col


def hole_factor1(rows_bits, rows, cols, col_heights):
    score = 0

    for c in range(cols):
        h = col_heights[c]
        if h == 0:
            continue  # no blocks => no holes

        top_r = rows - h  # row index of the first filled cell in this column
        seen_filled = False

        for r in range(top_r, rows):
            filled = (rows_bits[r] >> c) & 1
            if filled:
                seen_filled = True
            else:
                if seen_filled:
                    depth = r - top_r
                    score += depth

    return score


def hole_factor2(rows_bits, rows, cols, col_heights, shallow_k=4, shallow_bonus=3):
    score = 0

    for c in range(cols):
        h = col_heights[c]
        if h == 0:
            continue

        top_r = rows - h
        seen_filled = False

        for r in range(top_r, rows):
            filled = (rows_bits[r] >> c) & 1
            if filled:
                seen_filled = True
            else:
                if seen_filled:
                    score += 1
                    depth = r - top_r
                    if depth <= shallow_k:
                        score += shallow_bonus

    return score


def hole_factor3(rows_bits, rows, cols, col_heights, window_rows=6):

    mask = (1 << cols) - 1

    first_hole_r = None
    min_surface_r = None

    for c in range(cols):
        h = col_heights[c]
        if h > 0:
            top_r = rows - h
            if min_surface_r is None or top_r < min_surface_r:
                min_surface_r = top_r

    if min_surface_r is None:
        return 0

    # detect first hole row by scanning each column; track min r where a hole occurs
    for c in range(cols):
        h = col_heights[c]
        if h == 0:
            continue
        top_r = rows - h
        seen_filled = False
        for r in range(top_r, rows):
            filled = (rows_bits[r] >> c) & 1
            if filled:
                seen_filled = True
            else:
                if seen_filled:
                    if first_hole_r is None or r < first_hole_r:
                        first_hole_r = r
                    break  # earliest hole in this column is enough for first_hole_r

    if first_hole_r is None:
        return 0  # no holes

    # choose band start
    start_r = min(first_hole_r, min_surface_r + window_rows)
    if start_r < 0:
        start_r = 0
    if start_r > rows:
        return 0

    # sum weighted empties from start_r downward
    band_len = rows - start_r
    score = 0
    for i, r in enumerate(range(start_r, rows)):
        filled_count = (rows_bits[r] & mask).bit_count()
        empty_count = cols - filled_count

        # weight empties near the top of the band more
        weight = band_len - i
        score += empty_count * weight

    return score


def height_variance(heights):
    if not heights:
        return 0.0
    mean = sum(heights) / len(heights)
    return sum((v - mean) ** 2 for v in heights) / len(heights)


def _filled_at_k_from_bottom(bb, x, k):
    if x < 0 or x >= bb.cols:
        return True
    y = bb.rows - 1 - k
    if y < 0 or y >= bb.rows:
        return True
    return ((bb.rows_bits[y] >> x) & 1) == 1
