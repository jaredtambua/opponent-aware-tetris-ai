import random
import pygame
import numpy as np

from src.game.board import Board
from src.game.game import Game
from src.game.constants import ID_TO_COLOUR, LABEL_TO_ID
from src.game.tetromino import get_shape


from src.ai.fast_ai import TetrisAI
from src.ai.efficient_ai import TetrisAI2
from src.ai.fast_ai_opponent_aware import TetrisAI_Opp
from src.ai.efficient_ai_opponent_aware import TetrisAI2_Opp

# single modes
# from ai1_up_only import AI1UP
# from ai1_optimistic_only import AI1OPTIMISTIC
# from ai1_down_only import AI1Down

# from ai2_up_only import AI2UP
# from ai2_optimistic_only import AI2OPTIMISTIC
# from ai2_down_only import AI2Down

CELL_SIZE = 32
BOARD_ROWS = 22
BOARD_COLS = 10
SPAWN_BUFFER_ROWS = 2

MARGIN = 16
GAP = 12
LEFT_PANEL_W = 7 * CELL_SIZE
LEFT_PANEL_H = 6 * CELL_SIZE
RIGHT_PANEL_W = 7 * CELL_SIZE

FPS = 1000
GRAVITY_HZ = 1.0

BG_COLOUR = (0, 0, 0)
GRID_COLOUR = (60, 60, 60)
PANEL_BG = (18, 18, 18)
PANEL_OUT = (220, 220, 220)


GHOST_COLOUR = (180, 180, 180)


BOARD_W_PX = BOARD_COLS * CELL_SIZE
BOARD_H_PX = BOARD_ROWS * CELL_SIZE
EXTRA_TOP_PX = SPAWN_BUFFER_ROWS * CELL_SIZE

WINDOW_W = MARGIN + LEFT_PANEL_W + GAP + BOARD_W_PX + GAP + RIGHT_PANEL_W + MARGIN
WINDOW_H = MARGIN + EXTRA_TOP_PX + BOARD_H_PX + MARGIN + 60


SEED = 1
np.random.seed(SEED)
random.seed(SEED)

# garbage control
PRESSURE_LINES = 10
PRESSURE_MESSINESS = 1
PRESSURE_EVERY_PIECES = 40


def _enqueue_garbage_pressure(game, cols, lines, messiness, sim_time):
    y = max(0.0, min(1.0, float(messiness)))

    intended = random.randrange(cols)
    for _ in range(int(lines)):
        hole = intended
        if random.random() < y:
            hole = random.randrange(cols)

        game.receive_attack(1, hole, sim_time)


# AI control
AI_ENABLED = True


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PPSController:
    def __init__(self, fixed_pps):
        self.fixed_pps = fixed_pps
        self.interval = 1.0 / max(fixed_pps, 1e-6)
        self._timer = 0.0

    def update(self, dt):
        self._timer += dt

    def can_place(self) -> bool:
        return self._timer >= self.interval

    def on_place(self):
        self._timer -= self.interval


def run_ai_action_instant(
    game, ai, pps_controller, pieces_placed_ref, max_inputs_per_piece=64
):
    if game.game_over or not pps_controller.can_place():
        return False

    for _ in range(max_inputs_per_piece):
        if game.game_over or game.current_piece is None:
            return False

        action = ai.choose_action(game)

        if action == "hold":
            game.hold()
            if game.current_piece is None:
                return False

        elif action == "hard_drop":
            game.hard_drop()
            pps_controller.on_place()
            pieces_placed_ref[0] += 1
            return True

        elif action == "left":
            game.move_left()

        elif action == "right":
            game.move_right()

        elif action == "rot_cw":
            game.rotate_clockwise()

        elif action == "rot_ccw":
            game.rotate_counterclockwise()

        elif action == "rot_180":
            game.rotate_180()

        else:
            raise ValueError(f"Unknown AI action: {action}")

    raise RuntimeError(
        f"AI failed to reach hard_drop within {max_inputs_per_piece} inputs."
    )


# rendering helpers
def panel_inner(rect, pad=8):
    return pygame.Rect(
        rect.x + pad,
        rect.y + 28,
        rect.width - pad * 2,
        rect.height - 36,
    )


def draw_panel(surface, rect, title, font):
    pygame.draw.rect(surface, PANEL_BG, rect)
    pygame.draw.rect(surface, PANEL_OUT, rect, width=2)

    title_surf = font.render(title, True, PANEL_OUT)
    surface.blit(title_surf, (rect.x + 8, rect.y + 8))


def draw_grid(surface, origin_x, origin_y):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            x = origin_x + (col * CELL_SIZE)
            y = origin_y + (row * CELL_SIZE)
            pygame.draw.rect(
                surface, GRID_COLOUR, (x, y, CELL_SIZE, CELL_SIZE), width=1
            )


def draw_board(surface, board, origin_x, origin_y):
    for row in range(board.rows):
        for col in range(board.cols):
            value = board.grid[row, col]
            if not value:
                continue

            x = origin_x + (col * CELL_SIZE)
            y = origin_y + (row * CELL_SIZE)

            cell_colour = ID_TO_COLOUR.get(value)
            pygame.draw.rect(surface, cell_colour, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(
                surface, GRID_COLOUR, (x, y, CELL_SIZE, CELL_SIZE), width=1
            )


def draw_piece(surface, piece, origin_x, origin_y):
    if not piece:
        return

    label, rot, left_x, top_y, shape = piece
    piece_colour = ID_TO_COLOUR[LABEL_TO_ID[label]]

    h, w = shape.shape
    for shape_row in range(h):
        for shape_col in range(w):
            if shape[shape_row, shape_col] == 0:
                continue

            board_x = left_x + shape_col
            board_y = top_y + shape_row

            x = origin_x + (board_x * CELL_SIZE)
            y = origin_y + (board_y * CELL_SIZE)

            pygame.draw.rect(surface, piece_colour, (x, y, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(
                surface, GRID_COLOUR, (x, y, CELL_SIZE, CELL_SIZE), width=1
            )


def draw_piece_preview(surface, rect, label, rotation):
    shape = get_shape(label, int(rotation) % 4)

    rows, cols = shape.shape
    filled_rows = [r for r in range(rows) if any(shape[r, c] != 0 for c in range(cols))]
    filled_cols = [c for c in range(cols) if any(shape[r, c] != 0 for r in range(rows))]
    if not filled_rows or not filled_cols:
        return

    r0, r1 = min(filled_rows), max(filled_rows)
    c0, c1 = min(filled_cols), max(filled_cols)
    bbox_h = r1 - r0 + 1
    bbox_w = c1 - c0 + 1

    PAD = 10
    avail_w = max(1, rect.width - PAD * 2)
    avail_h = max(1, rect.height - PAD * 2)

    cell = min(avail_w // bbox_w, avail_h // bbox_h)
    cell = max(6, min(30, int(cell * 0.75)))

    draw_w = bbox_w * cell
    draw_h = bbox_h * cell
    start_x = rect.x + (rect.width - draw_w) // 2
    start_y = rect.y + (rect.height - draw_h) // 2

    colour = ID_TO_COLOUR[LABEL_TO_ID[label]]

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if shape[rr, cc] == 0:
                continue
            x = start_x + (cc - c0) * cell
            y = start_y + (rr - r0) * cell
            pygame.draw.rect(surface, colour, (x, y, cell, cell))
            pygame.draw.rect(surface, GRID_COLOUR, (x, y, cell, cell), width=1)


def draw_next_preview_list(surface, rect, next_labels, max_show=5):
    if not next_labels:
        return

    labels = next_labels[:max_show]
    slot_h = rect.height // max_show

    for i, label in enumerate(labels):
        slot = pygame.Rect(rect.x, rect.y + i * slot_h, rect.width, slot_h)
        draw_piece_preview(surface, slot, label, rotation=0)


def draw_ghost(surface, game, origin_x, origin_y):
    if not game.current_piece:
        return

    ghost = game.ghost_position()
    if not ghost:
        return

    left_x, ghost_y, shape = ghost
    label = game.current_piece[0]
    colour = ID_TO_COLOUR[LABEL_TO_ID[label]]
    darker = tuple(max(0, int(c * 0.4)) for c in colour)

    h, w = shape.shape
    for r in range(h):
        for c in range(w):
            if shape[r, c] == 0:
                continue

            bx = left_x + c
            by = ghost_y + r
            if by < 0:
                continue

            x = origin_x + (bx * CELL_SIZE)
            y = origin_y + (by * CELL_SIZE)

            pygame.draw.rect(surface, darker, (x, y, CELL_SIZE, CELL_SIZE))


def draw_counters(surface, game, rect, font):
    y = rect.bottom + 6
    x = rect.x + 8

    combo = getattr(game, "combo_count", 0)
    b2b = getattr(game, "b2b_chain_len", 0)
    total_atk = getattr(game, "total_attack_sent", 0)
    apm = game.get_apm() if hasattr(game, "get_apm") else 0.0
    pieces = getattr(game, "pieces_locked")

    surface.blit(font.render(f"COMBO: {combo}", True, PANEL_OUT), (x, y))
    y += font.get_height() + 2

    surface.blit(font.render(f"B2B: {b2b}", True, PANEL_OUT), (x, y))
    y += font.get_height() + 2

    surface.blit(font.render(f"ATK: {total_atk}", True, PANEL_OUT), (x, y))
    y += font.get_height() + 2

    pps = game.get_pps() if hasattr(game, "get_pps") else 0.0
    surface.blit(font.render(f"PPS: {pps:.2f}", True, PANEL_OUT), (x, y))
    y += font.get_height() + 2

    surface.blit(font.render(f"APM: {apm:.1f}", True, PANEL_OUT), (x, y))
    y += font.get_height() + 8

    incoming = (
        game.incoming_garbage_lines() if hasattr(game, "incoming_garbage_lines") else 0
    )
    surface.blit(font.render(f"IN: {incoming}", True, PANEL_OUT), (x, y))

    y += font.get_height() + 8
    pieces = getattr(game, "pieces_locked", 0)
    surface.blit(font.render(f"PIECES: {pieces}", True, PANEL_OUT), (x, y))

    # toggling AI
    # y += font.get_height() + 8
    # ai_on = "ON" if AI_ENABLED else "OFF"
    # surface.blit(font.render(f"AI: {ai_on} (TAB)", True, PANEL_OUT), (x, y))


# Controls (manual)
def handle_keydown(game, key) -> None:
    if key == pygame.K_LEFT:
        game.move_left()
    elif key == pygame.K_RIGHT:
        game.move_right()
    elif key == pygame.K_e:
        game.rotate_clockwise()
    elif key == pygame.K_w:
        game.rotate_counterclockwise()
    elif key == pygame.K_q:
        game.rotate_180()
    elif key == pygame.K_r:
        game.hold()
    elif key == pygame.K_SPACE:
        game.hard_drop()
    elif key == pygame.K_t:
        game.spawn_random()


# Main loop
def main():
    global AI_ENABLED

    pygame.init()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(
        "Tetris: Single Board vs Simulated Garbage Pressure (+AI)"
    )
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont(None, 28)

    board = Board(rows=BOARD_ROWS, cols=BOARD_COLS)
    game = Game(board)
    game.spawn_random()

    #  AI setup
    # ai = TetrisAI(ply=2, gamma=0.95, depth_beam_width=10, well_col=9)
    # ai = TetrisAI_Opp(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # ai = TetrisAI2(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # ai = TetrisAI2_Opp(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # single-modes
    # ai = AI1UP(ply=2, gamma=0.95, depth_beam_width=10, well_col=9)
    # ai = AI1OPTIMISTIC(ply=2, gamma=0.95, depth_beam_width=10, well_col=9)
    # ai = AI1Down(ply=2, gamma=0.95, depth_beam_width=10, well_col=9)

    # ai = AI2UP(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    ai = AI2OPTIMISTIC(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # ai = AI2Down(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    FIXED_PPS = 100
    ai_pps = PPSController(FIXED_PPS)
    ai_elapsed = 0.0
    ai_pieces_placed = [0]  # mutable ref for run_ai_action()

    # layout rects
    board_origin_x = MARGIN + LEFT_PANEL_W + GAP
    board_origin_y = MARGIN + EXTRA_TOP_PX

    hold_rect = pygame.Rect(MARGIN, board_origin_y, LEFT_PANEL_W, LEFT_PANEL_H)
    next_rect = pygame.Rect(
        MARGIN + LEFT_PANEL_W + GAP + BOARD_W_PX + GAP,
        board_origin_y,
        RIGHT_PANEL_W,
        BOARD_H_PX,
    )

    gravity_timer = 0.0
    gravity_interval = 1.0 / max(GRAVITY_HZ, 1e-6)

    # soft drop + DAS/ARR (manual)
    DAS = 0.100
    ARR = 0.01
    SOFT_DROP_SPEED = 10

    left_held = 0.0
    right_held = 0.0
    horiz_repeat_timer = 0.0

    sim_time = 0.0
    pieces_placed_for_pressure = 0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        sim_time += dt
        gravity_timer += dt

        had_piece_at_frame_start = bool(game.current_piece)

        # move delayed garbage into pending queue when due
        game.process_incoming_attacks(sim_time)

        # AI pps update
        # if AI_ENABLED and not game.game_over:
        #     ai_elapsed += dt
        #     avg_pps = (ai_pieces_placed[0] / ai_elapsed) if ai_elapsed > 0 else 0.0

        #     # same danger logic style as main.py (tweak if you like)
        #     in_danger = (board.stack_height() >= 10)
        #     ai_pps.update(dt, avg_pps, in_danger)

        if AI_ENABLED and not game.game_over:
            ai_elapsed += dt
            ai_pps.update(dt)

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_TAB:
                    AI_ENABLED = not AI_ENABLED

                else:
                    if not AI_ENABLED:
                        handle_keydown(game, event.key)

        # continuous movement (manual only)
        keys = pygame.key.get_pressed()

        if not AI_ENABLED:
            if keys[pygame.K_LEFT]:
                left_held += dt
                right_held = 0.0
            else:
                left_held = 0.0

            if keys[pygame.K_RIGHT]:
                right_held += dt
                left_held = 0.0
            else:
                right_held = 0.0

            direction = 0
            if left_held > 0 and right_held == 0:
                direction = -1
            elif right_held > 0 and left_held == 0:
                direction = 1
            else:
                horiz_repeat_timer = 0.0

            if direction != 0 and (left_held > DAS or right_held > DAS):
                horiz_repeat_timer += dt
                while horiz_repeat_timer >= ARR:
                    horiz_repeat_timer -= ARR
                    if direction == -1:
                        game.move_left()
                    else:
                        game.move_right()

            if keys[pygame.K_DOWN]:
                for _ in range(SOFT_DROP_SPEED):
                    game.soft_drop()

        # AI actions (only when enabled)
        # if AI_ENABLED and not game.game_over:
        #     run_ai_action(game, ai, ai_pps, ai_pieces_placed)

        if AI_ENABLED and not game.game_over:
            while ai_pps.can_place():
                placed = run_ai_action_instant(game, ai, ai_pps, ai_pieces_placed)
                if not placed:
                    break

        # gravity
        while gravity_timer >= gravity_interval:
            gravity_timer -= gravity_interval
            if not game.game_over:
                game.gravity_tick()

        # detect a placement (piece -> no piece)
        if had_piece_at_frame_start and not game.current_piece and not game.game_over:
            pieces_placed_for_pressure += 1

            if PRESSURE_EVERY_PIECES > 0 and (
                pieces_placed_for_pressure % PRESSURE_EVERY_PIECES == 0
            ):
                _enqueue_garbage_pressure(
                    game=game,
                    cols=board.cols,
                    lines=PRESSURE_LINES,
                    messiness=PRESSURE_MESSINESS,
                    sim_time=sim_time,
                )

        # apply pending garbage between pieces, then spawn next piece
        if not game.game_over and not game.current_piece:
            if game.skip_garbage_apply_once:
                game.skip_garbage_apply_once = False
            else:
                packets = game.pop_garbage_to_apply(game.MAX_GARBAGE_PER_PIECE)
                for lines, hole in packets:
                    board.add_garbage_rows(lines, hole_col=hole)

            game.spawn_random()

        # draw
        screen.fill(BG_COLOUR)

        draw_panel(screen, hold_rect, "HOLD", title_font)
        if game.hold_label is not None:
            draw_piece_preview(
                screen, panel_inner(hold_rect), game.hold_label, rotation=0
            )

        draw_counters(screen, game, hold_rect, title_font)

        draw_panel(screen, next_rect, "NEXT", title_font)
        draw_next_preview_list(screen, panel_inner(next_rect), game.peek_next(5))

        draw_ghost(screen, game, board_origin_x, board_origin_y)
        draw_grid(screen, board_origin_x, board_origin_y)
        draw_board(screen, board, board_origin_x, board_origin_y)
        draw_piece(screen, game.current_piece, board_origin_x, board_origin_y)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
