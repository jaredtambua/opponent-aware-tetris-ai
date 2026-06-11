import numpy as np
import random
import time
from collections import deque

from src.game.constants import LABEL_TO_ID
from src.game.tetromino import (
    SHAPES,
    get_shape,
    try_rotate_cw,
    try_rotate_ccw,
    try_rotate_180,
)
from src.game.attack_table import compute_attack


class Game:
    def __init__(self, board, bag_rng=None):

        self.DEBUG_ONLY_I_PIECES = False

        self.start_time = time.time()
        self.total_attack_sent = 0

        self.board = board
        self.current_piece = None
        self.game_over = False
        self._bag = deque()

        # holding
        self.hold_label = None
        self._hold_used = False

        # for calculating garbage
        self.last_locked_piece = None
        self.lines_cleared_total = 0
        self.lines_cleared_last = 0

        # b2b + combo
        self.b2b_armed = False
        self.b2b_chain_len = 0
        self.attack_last = 0

        self.combo_armed = False
        self.combo_count = 0
        self.b2b_break_bonus = 0

        # garbage queues
        self.pending_garbage_packets = deque()

        self.incoming_attacks = deque()

        self.skip_garbage_apply_once = False

        # tuning
        self.MAX_GARBAGE_PER_PIECE = 8
        self.GARBAGE_DELAY_SEC = 0  # tweak to taste

        # lock-delay / lock-reset tracking
        self.on_ground = False
        self.grounded_ticks = 0
        self.lock_resets_used = 0
        self.LOCK_DELAY_TICKS = 2
        self.MAX_LOCK_RESETS = 6

        # T-spin tracking
        self.tspin_last = None

        # pps tracking
        self.pieces_locked = 0

        # bag rng
        self.bag_rng = bag_rng

    def find_spawn_xy_coords(self, shape):
        # center horizontally
        shape_height, shape_width = shape.shape
        left_x = (self.board.cols - shape_width) // 2

        # find the index of the topmost filled row inside the shape (the first row that contains any non-zero cell)
        top_filled_row = 0
        for r in range(shape_height):
            row_has_block = False
            for c in range(shape_width):
                if shape[r, c] != 0:
                    row_has_block = True
                    break
            if row_has_block:
                top_filled_row = r
                break

        # we start with a negative y so it appears partly above the board.
        top_y = -(top_filled_row + 2)

        return left_x, top_y

    def spawn(self, label, rotation=0):
        if self.game_over:
            return False

        rotation = int(rotation) % 4
        shape = get_shape(label, rotation)

        left_x, top_y = self.find_spawn_xy_coords(shape)

        if self.board.can_legally_place(shape, left_x, top_y):
            self.current_piece = [label, rotation, left_x, top_y, shape]

            self.on_ground = False
            self.grounded_ticks = 0
            self.lock_resets_used = 0

            return True

        # final piece placement above board (when game over)
        shape_h, _ = shape.shape

        best_y = None
        # search all possible vertical positions for this x from well above the board to the bottom
        for y in range(-shape_h, self.board.rows):
            if self.board.can_legally_place(shape, left_x, y):
                best_y = y

        if best_y is not None:
            self.current_piece = [label, rotation, left_x, best_y, shape]
            self.game_over = True
            return False

        self.current_piece = None
        self.game_over = True

        self.on_ground = False
        self.grounded_ticks = 0
        self.lock_resets_used = 0

        return False

    def _refill_bag(self):
        if getattr(self, "DEBUG_ONLY_I_PIECES", False):
            while len(self._bag) < 14:
                self._bag.append("I")
            return

        while len(self._bag) < 14:
            labels = list(SHAPES.keys())
            (self.bag_rng or random).shuffle(labels)
            self._bag.extend(labels)

    def spawn_random(self):
        self._refill_bag()
        piece = self._bag.popleft()  # take from front
        return self.spawn(piece)

    def rotate_clockwise(self):
        if self.game_over or not self.current_piece:
            return

        label, rotation, left_x, top_y, _shape = self.current_piece
        new_x, new_y, new_rotation, new_shape, ok = try_rotate_cw(
            self.board, label, left_x, top_y, rotation
        )

        if ok:
            self.current_piece = [label, new_rotation, new_x, new_y, new_shape]
            self._reset_lock_delay_on_action()

    def rotate_counterclockwise(self):
        if self.game_over or not self.current_piece:
            return

        label, rotation, left_x, top_y, _shape = self.current_piece
        new_x, new_y, new_rotation, new_shape, ok = try_rotate_ccw(
            self.board, label, left_x, top_y, rotation
        )

        if ok:
            self.current_piece = [label, new_rotation, new_x, new_y, new_shape]
            self._reset_lock_delay_on_action()

    def rotate_180(self):
        if self.game_over or not self.current_piece:
            return

        label, rotation, left_x, top_y, _shape = self.current_piece
        new_x, new_y, new_rotation, new_shape, ok = try_rotate_180(
            self.board, label, left_x, top_y, rotation
        )

        if ok:
            self.current_piece = [label, new_rotation, new_x, new_y, new_shape]
            self._reset_lock_delay_on_action()

    def _try_nudge(self, dx=0, dy=0):
        if self.game_over or not self.current_piece:
            return

        label, rotation, left_x, top_y, shape = self.current_piece
        target_x = left_x + dx
        target_y = top_y + dy

        if self.board.can_legally_place(shape, target_x, target_y):
            self.current_piece = [label, rotation, target_x, target_y, shape]

            if dx != 0:
                self._reset_lock_delay_on_action()
            return True

        return False

    def move_left(self):
        self._try_nudge(dx=-1)

    def move_right(self):
        self._try_nudge(dx=+1)

    def soft_drop(self):
        if self.game_over or not self.current_piece:
            return

        label, rotation, left_x, top_y, shape = self.current_piece

        if self.board.can_legally_place(shape, left_x, top_y + 1):
            top_y += 1

            self.current_piece = [label, rotation, left_x, top_y, shape]

            self.on_ground = False
            self.grounded_ticks = 0

        else:
            self.on_ground = True

    def gravity_tick(self):
        if self.game_over or not self.current_piece:
            return

        label, rotation, left_x, top_y, shape = self.current_piece

        if self.board.can_legally_place(shape, left_x, top_y + 1):
            top_y += 1

            self.current_piece = [label, rotation, left_x, top_y, shape]
            self.on_ground = False
            self.grounded_ticks = 0

        else:
            self.on_ground = True
            self.grounded_ticks += 1
            if self.grounded_ticks >= self.LOCK_DELAY_TICKS:
                self._lock()

    def hard_drop(self):
        if self.game_over or not self.current_piece:
            return

        label, rotation, left_x, top_y, shape = self.current_piece

        # keep moving down while the next row is legal.
        while self.board.can_legally_place(shape, left_x, top_y + 1):
            top_y += 1

        self.current_piece = [label, rotation, left_x, top_y, shape]
        self._lock()

    def _lock(self):
        label, rot, left_x, top_y, shape = self.current_piece
        piece_id = LABEL_TO_ID[label]

        topped_out = False
        shape_rows, shape_cols = shape.shape

        b2b_break_bonus = 0

        for r in range(shape_rows):
            for c in range(shape_cols):
                if shape[r, c] == 0:
                    continue

                board_row = top_y + r
                if board_row < 0:
                    topped_out = True
                    break
            if topped_out:
                break

        grid_before = self.board.grid.copy()

        spinned = self.check_if_spinned(left_x, top_y, rot)

        cleared = self.board.place_piece(shape, left_x, top_y, piece_id)
        self.pieces_locked += 1
        self.skip_garbage_apply_once = cleared > 0

        self.last_locked_piece = (label, rot, left_x, top_y)
        self.lines_cleared_last = cleared
        self.lines_cleared_total += cleared

        self.tspin_last = self._tspin_attack_type_on_lock(
            label=label,
            rot=rot,
            left_x=left_x,
            top_y=top_y,
            cleared=cleared,
            grid_before=grid_before,
            shape=shape,
        )

        if cleared > 0:
            if not self.combo_armed:
                # first clear: arm combo, but no bonus yet
                self.combo_armed = True
                self.combo_count = 0  # table index 0 (no extra yet for many types)
            else:
                # subsequent consecutive clears
                self.combo_count += 1
        else:
            # combo broken
            self.combo_armed = False
            self.combo_count = 0

        b2b_eligible = (cleared == 4) or spinned or (self.tspin_last is not None)

        if cleared > 0:
            if b2b_eligible:
                if not self.b2b_armed:
                    # first eligible clear: arm B2B but give no bonus
                    self.b2b_armed = True
                    self.b2b_chain_len = 0
                else:
                    # real B2B chain
                    self.b2b_chain_len += 1
            else:
                # chain broken
                if self.b2b_chain_len > 3:
                    b2b_break_bonus = self.b2b_chain_len

                self.b2b_chain_len = 0
                self.b2b_armed = False

        # B2B warmup
        effective_b2b = self.b2b_chain_len

        # combo warmup
        effective_combo = self.combo_count

        attack = self.compute_garbage_sent(
            effective_combo_count=effective_combo, effective_b2b_chain_len=effective_b2b
        )

        attack += b2b_break_bonus

        self.total_attack_sent += attack

        final_attack = self._apply_garbage_cancellation(attack)
        self.attack_last = final_attack

        if topped_out:
            self.game_over = True

        self.current_piece = None
        self._hold_used = False

    def get_pps(self):
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.pieces_locked / elapsed

    def get_apm(self):
        elapsed = time.time() - self.start_time
        if elapsed <= 0:
            return 0
        return (self.total_attack_sent / elapsed) * 60

    def check_if_spinned(self, x, y, rot):
        if not self.current_piece:
            return False

        label, _rot, _x, _y, shape = self.current_piece

        if label == "O":
            return False

        board = self.board
        rot = int(rot) % 4

        can_left = board.can_legally_place(shape, x - 1, y)
        can_right = board.can_legally_place(shape, x + 1, y)
        can_nudge = can_left or can_right

        can_drop = board.can_legally_place(shape, x, y + 1)

        # blocked above (reuses helper)
        blocked_above = not self._no_blocks_above_shape(shape, x, y, board.grid)

        return (not can_nudge) and (not can_drop) and blocked_above

    # hold logic as follows:
    # case 1 - we don't have a hold yet, so we hold the current piece, and spawn next piece from bag
    # case 2 - we are already holding a piece, so swap in place with piece on board
    def hold(self):
        if self.game_over or not self.current_piece or self._hold_used:
            return

        label, rot, left_x, top_y, shape = self.current_piece

        if self.hold_label is None:
            # store current and draw next piece
            self.hold_label = label
            self.current_piece = None
            self._hold_used = True
            self.spawn_random()

        else:
            # swap: current goes to hold; held piece spawns fresh at rot 0
            swap_in = self.hold_label
            self.hold_label = label
            self.current_piece = None
            self._hold_used = True
            self.spawn(swap_in, rotation=0)

    def ghost_position(self):
        if self.game_over or not self.current_piece:
            return None

        label, rot, left_x, top_y, shape = self.current_piece
        ghost_y = top_y

        while self.board.can_legally_place(shape, left_x, ghost_y + 1):
            ghost_y += 1
        return (left_x, ghost_y, shape)

    def peek_next(self, n=5):
        if not self._bag or n <= 0:
            return []

        labels = list(self._bag)
        return labels[:n]

    def compute_garbage_sent(self, effective_combo_count, effective_b2b_chain_len):
        cleared = self.lines_cleared_last
        if cleared == 0:
            return 0

        perfect_clear = self.board.grid.sum() == 0

        # if lock classified this as a T-spin type, use it.
        if self.tspin_last is not None:
            atk_type = self.tspin_last
        else:
            if cleared == 1:
                atk_type = "single"
            elif cleared == 2:
                atk_type = "double"
            elif cleared == 3:
                atk_type = "triple"
            elif cleared == 4:
                atk_type = "quad"
            else:
                return 0

        return compute_attack(
            atk_type=atk_type,
            combo_count=effective_combo_count,
            b2b_chain_len=effective_b2b_chain_len,
            all_clear=perfect_clear,
        )

    def cancel_incoming_garbage(self, lines):
        need = int(lines)
        cancelled = 0

        # cancel from pending packets first
        while need > 0 and self.pending_garbage_packets:
            pkt_lines, hole = self.pending_garbage_packets[0]
            take = min(pkt_lines, need)
            pkt_lines -= take
            need -= take
            cancelled += take

            if pkt_lines == 0:
                self.pending_garbage_packets.popleft()
            else:
                self.pending_garbage_packets[0][0] = pkt_lines

        # cancel from delayed incoming attacks
        while need > 0 and self.incoming_attacks:
            t, pkt_lines, hole = self.incoming_attacks[0]
            take = min(pkt_lines, need)
            pkt_lines -= take
            need -= take
            cancelled += take

            if pkt_lines == 0:
                self.incoming_attacks.popleft()
            else:
                # keep same delivery time + hole, just reduce lines
                self.incoming_attacks[0][1] = pkt_lines

        return cancelled

    def _apply_garbage_cancellation(self, attack):
        if attack <= 0:
            return attack

        cancelled = self.cancel_incoming_garbage(attack)
        return attack - cancelled

    def receive_attack(self, lines, hole_col, now: float, delay: float = None):
        if lines <= 0:
            return
        if delay is None:
            delay = self.GARBAGE_DELAY_SEC
        self.incoming_attacks.append([now + delay, int(lines), int(hole_col)])

    def process_incoming_attacks(self, now: float):
        while self.incoming_attacks and self.incoming_attacks[0][0] <= now:
            _t, lines, hole = self.incoming_attacks.popleft()
            self.pending_garbage_packets.append([lines, hole])

    def incoming_garbage_lines(self):
        delayed = sum(lines for _t, lines, _hole in self.incoming_attacks)
        pending = sum(lines for lines, _hole in self.pending_garbage_packets)
        return delayed + pending

    def pop_garbage_to_apply(self, cap):
        out = []
        remaining = int(cap)

        while remaining > 0 and self.pending_garbage_packets:
            pkt_lines, hole = self.pending_garbage_packets.popleft()
            use = min(pkt_lines, remaining)
            out.append((use, hole))
            remaining -= use

            leftover = pkt_lines - use
            if leftover > 0:
                # put remainder back at the FRONT with same hole
                self.pending_garbage_packets.appendleft([leftover, hole])

        return out

    def inject_pending_garbage(self, x, y: float):
        lines = int(x)
        if lines <= 0:
            return

        # clamp messiness
        y = float(y)
        if y < 0.0:
            y = 0.0
        elif y > 1.0:
            y = 1.0

        cols = int(getattr(self.board, "cols", 10))

        # intended hole column for this injection batch
        intended = random.randint(0, cols - 1)

        # merge consecutive same-hole lines
        cur_hole = None
        cur_count = 0

        for _ in range(lines):
            hole = intended
            if random.random() < y:
                hole = random.randint(0, cols - 1)

            if cur_hole is None:
                cur_hole = hole
                cur_count = 1
            elif hole == cur_hole:
                cur_count += 1
            else:
                self.pending_garbage_packets.append([cur_count, cur_hole])
                cur_hole = hole
                cur_count = 1

        # flush last packet
        if cur_hole is not None and cur_count > 0:
            self.pending_garbage_packets.append([cur_count, cur_hole])

    # use up attack
    def consume_attack(self):
        atk = self.attack_last
        self.attack_last = 0
        return atk

    # to count as a t-spin, at least 3 corners from the t center piece must be filled
    def _tspin_corners_filled(self, center_x, center_y, grid):
        rows, cols = grid.shape
        corners = [
            (center_x - 1, center_y - 1),
            (center_x + 1, center_y - 1),
            (center_x - 1, center_y + 1),
            (center_x + 1, center_y + 1),
        ]

        count = 0
        for x, y in corners:
            # off the left/right or below bottom = treated as filled
            if x < 0 or x >= cols or y >= rows:
                count += 1
                continue

            # avove the top of the visible board = treated as empty
            if y < 0:
                continue

            if grid[y, x] != 0:
                count += 1

        return count

    def _no_blocks_above_shape(self, shape, left_x, top_y, grid) -> bool:
        rows, cols = grid.shape
        sh_rows, sh_cols = shape.shape

        for r in range(sh_rows):
            for c in range(sh_cols):
                if shape[r, c] == 0:
                    continue

                x = left_x + c
                y = top_y + r
                above_y = y - 1

                # above visible board counts as no block above
                if above_y < 0:
                    continue

                if 0 <= x < cols and above_y < rows and grid[above_y, x] != 0:
                    return False

        return True

    def _nose_front_empty(self, left_x, top_y, rot, grid) -> bool:
        rows, cols = grid.shape
        rot = int(rot) % 4

        # T pivot in 3x3 frame
        cx, cy = left_x + 1, top_y + 1

        if rot == 1:
            tx, ty = cx + 2, cy
        elif rot == 3:
            tx, ty = cx - 2, cy
        else:
            return False  # only care about rot 1 and 3

        if tx < 0 or tx >= cols or ty < 0 or ty >= rows:
            return False

        return grid[ty, tx] == 0

    def _tspin_attack_type_on_lock(
        self, label, rot, left_x, top_y, cleared, grid_before, shape
    ):
        if label != "T" or cleared <= 0:
            return None

        cx = left_x + 1
        cy = top_y + 1

        filled = self._tspin_corners_filled(cx, cy, grid_before)
        if filled < 3:
            return None

        nose_empty = self._nose_front_empty(left_x, top_y, rot, grid_before)
        no_blocks_above = self._no_blocks_above_shape(shape, left_x, top_y, grid_before)

        is_mini = (filled == 3) and no_blocks_above and nose_empty

        if cleared == 3:
            return "tspin_triple" if filled == 4 else None

        if cleared == 1:
            if is_mini:
                return "tspin_mini_single"
            return "tspin_single" if (filled == 3 and not no_blocks_above) else None

        if cleared == 2:
            if is_mini:
                return "tspin_mini_double"
            return "tspin_double" if (filled == 3 and not no_blocks_above) else None

        return None

    def _reset_lock_delay_on_action(self):
        if self.on_ground and self.lock_resets_used < self.MAX_LOCK_RESETS:
            self.grounded_ticks = 0
            self.lock_resets_used += 1

    def force_to_landing(self, x, y, rot) -> bool:
        if self.game_over or not self.current_piece:
            return False

        label, _rot, _x, _y, _shape = self.current_piece
        rot = int(rot) % 4
        shape = get_shape(label, rot)

        if not self.board.can_legally_place(shape, x, y):
            return False

        self.current_piece = [label, rot, x, y, shape]
        return True
