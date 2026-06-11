import random

from collections import deque
from src.game.attack_table import compute_attack

from src.game.tetromino import PIECE_ROW_MASKS, KICKS_I, KICKS_JLSTZ


def _full_row_mask(cols):
    return (1 << cols) - 1


# compute-only board
# rows_bits[y] is an int where bit x = 1 means column x is occupied
class BitBoard:
    def __init__(self, rows, cols, rows_bits):
        self.rows = rows
        self.cols = cols
        self.rows_bits = rows_bits  # keep tuple for immutability

    # construction / conversion

    @staticmethod
    def empty(rows=20, cols=10):
        return BitBoard(rows, cols, tuple([0] * rows))

    @staticmethod
    def from_board(board):
        return BitBoard(board.rows, board.cols, tuple(board.rows_bits))

    def to_board_bits(self):
        return list(self.rows_bits)

    # core bit operations
    def is_row_full(self, y):
        return self.rows_bits[y] == _full_row_mask(self.cols)

    def clear_full_rows(self):
        full = _full_row_mask(self.cols)
        kept = [row for row in self.rows_bits if row != full]
        cleared = self.rows - len(kept)
        if cleared == 0:
            return self, 0
        new_rows = [0] * cleared + kept
        return BitBoard(self.rows, self.cols, tuple(new_rows)), cleared

    # placement legality
    def can_place(self, label, rot, left_x, top_y):
        row_masks, h, _w = PIECE_ROW_MASKS[label][rot % 4]
        board_mask = _full_row_mask(self.cols)

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

            # above-top: no collision check
            if board_row < 0:
                continue

            # collision
            if self.rows_bits[board_row] & shifted:
                return False

        return True

    # applying a placement
    def place(self, label, rot, left_x, top_y):
        if not self.can_place(label, rot, left_x, top_y):
            raise ValueError("Illegal placement")

        row_masks, h, _w = PIECE_ROW_MASKS[label][rot % 4]
        board_mask = _full_row_mask(self.cols)

        new_rows = list(self.rows_bits)

        for r in range(h):
            row_mask = row_masks[r]
            if row_mask == 0:
                continue

            board_row = top_y + r

            if left_x >= 0:
                shifted = row_mask << left_x
            else:
                shifted = row_mask >> (-left_x)

            shifted &= board_mask

            if board_row < 0:
                continue

            new_rows[board_row] |= shifted

        placed = BitBoard(self.rows, self.cols, tuple(new_rows))
        cleared_board, _cleared = placed.clear_full_rows()
        return cleared_board
