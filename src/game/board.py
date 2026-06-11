import numpy as np
import random

from src.game.tetromino import PIECE_ROW_MASKS
from src.game.constants import GARBAGE_ID
from src.game.attack_table import compute_attack


class Board:
    def __init__(self, rows=20, cols=10):
        self.rows = rows
        self.cols = cols
        self.grid = np.zeros((rows, cols), dtype=int)

        # Bitboard shadow: one int per row. Bit x = 1 means column x is occupied.
        self.rows_bits = [0] * rows

    def _rebuild_rows_bits(self):
        bits = []
        for y in range(self.rows):
            row_mask = 0
            for x in range(self.cols):
                if self.grid[y, x] != 0:
                    row_mask |= 1 << x
            bits.append(row_mask)
        self.rows_bits = bits

    def is_row_full(self, row_i):
        return np.all(self.grid[row_i] != 0)

    def clear_full_rows(self):
        updated_grid = []
        for row in self.grid:
            if not all(row):
                updated_grid.append(row)

        cleared = self.rows - len(updated_grid)

        for _ in range(cleared):
            updated_grid.insert(0, np.zeros(self.cols, dtype=int))

        self.grid = np.array(updated_grid, dtype=int)
        self._rebuild_rows_bits()

        return cleared

    def can_legally_place(self, shape, left_x, top_y):
        shape_rows, shape_cols = shape.shape
        board_mask = (1 << self.cols) - 1  # e.g. 10 cols => 0b1111111111

        for r in range(shape_rows):
            # Build row mask for this shape row (only occupied cells)
            row_mask = 0
            for c in range(shape_cols):
                if shape[r, c] != 0:
                    row_mask |= 1 << c

            if row_mask == 0:
                continue  # nothing in this row, ignore completely (matches your original)

            board_row = top_y + r

            # Floor: if any occupied cells would be below the board, illegal
            if board_row >= self.rows:
                return False

            # Shift into board columns, handling negative x safely
            if left_x >= 0:
                shifted = row_mask << left_x
            else:
                k = -left_x
                # If any bits would fall off the left edge, illegal
                if row_mask & ((1 << k) - 1):
                    return False
                shifted = row_mask >> k

            # Right wall: if any bits go past the right edge, illegal
            if shifted & ~board_mask:
                return False

            # Above-top: walls apply, but no collision check (matches your original order)
            if board_row < 0:
                continue

            # Collision check (fast)
            if self.rows_bits[board_row] & shifted:
                return False

        return True

    def can_legally_place_mask(
        self, label: str, rot: int, left_x: int, top_y: int
    ) -> bool:
        row_masks, h, _w = PIECE_ROW_MASKS[label][int(rot) % 4]
        board_mask = (1 << self.cols) - 1

        for r in range(h):
            row_mask = row_masks[r]
            if row_mask == 0:
                continue

            board_row = top_y + r

            # floor
            if board_row >= self.rows:
                return False

            # shift with left-wall handling
            if left_x >= 0:
                shifted = row_mask << left_x
            else:
                k = -left_x
                if row_mask & ((1 << k) - 1):
                    return False
                shifted = row_mask >> k

            # right wall
            if shifted & ~board_mask:
                return False

            # above-top: walls apply, but no collision check
            if board_row < 0:
                continue

            # collision
            if self.rows_bits[board_row] & shifted:
                return False

        return True

    def can_legally_place_grid(self, shape, left_x, top_y):
        shape_rows, shape_cols = shape.shape
        for sr in range(shape_rows):
            for sc in range(shape_cols):
                if shape[sr, sc] == 0:
                    continue
                bc = left_x + sc
                br = top_y + sr
                if bc < 0 or bc >= self.cols or br >= self.rows:
                    return False
                if br < 0:
                    continue
                if self.grid[br, bc] != 0:
                    return False
        return True

    def place_piece(self, shape, left_x, top_y, piece_id):
        shape_rows, shape_cols = shape.shape

        for shape_row in range(shape_rows):
            for shape_col in range(shape_cols):

                if shape[shape_row, shape_col] == 0:
                    continue

                board_col = left_x + shape_col
                board_row = top_y + shape_row

                if board_row < 0:
                    continue

                if 0 <= board_col < self.cols and 0 <= board_row < self.rows:
                    self.grid[board_row, board_col] = piece_id
                else:
                    raise ValueError("Tried to place piece out-of-bounds.")

        # This rebuilds rows_bits inside clear_full_rows()
        return self.clear_full_rows()

    def add_garbage_rows(self, n_rows, hole_col=None):
        if n_rows <= 0:
            return

        for _ in range(n_rows):
            hole = random.randrange(self.cols) if hole_col is None else hole_col

            row = np.full(self.cols, GARBAGE_ID, dtype=int)
            row[hole] = 0

            self.grid = np.vstack([self.grid[1:], row])

        self._rebuild_rows_bits()

    def _row_mask_from_grid(self, y: int) -> int:
        m = 0
        for x in range(self.cols):
            if self.grid[y, x] != 0:
                m |= 1 << x
        return m

    def stack_height(self) -> int:
        # row 0 = top
        for y in range(self.rows):
            if self.rows_bits[y] != 0:  # any filled cell in this row
                return self.rows - y
        return 0
