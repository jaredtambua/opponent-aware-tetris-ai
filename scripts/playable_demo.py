import os

import pygame
import sys
import random
import time


from src.game.board import Board
from src.game.game import Game
from src.game.constants import ID_TO_COLOUR, LABEL_TO_ID
from src.game.tetromino import get_shape

from src.ai.fast_ai import TetrisAI
from src.ai.efficient_ai import TetrisAI2

from src.ai.fast_ai_opponent_aware import TetrisAI_Opp
from src.ai.efficient_ai_opponent_aware import TetrisAI2_Opp

# board configurations
CELL_SIZE = 32
BOARD_ROWS = 20
BOARD_COLS = 10


# layout
MARGIN = 16
GAP = 12  # gap between panel and board
LEFT_PANEL_W = 7 * CELL_SIZE
LEFT_PANEL_H = 6 * CELL_SIZE
RIGHT_PANEL_W = 7 * CELL_SIZE

SPAWN_BUFFER_ROWS = 2

FPS = 100000
GRAVITY_HZ = 1.0
BG_COLOUR = (0, 0, 0)
GRID_COLOUR = (60, 60, 60)
TEXT_COLOUR = (230, 230, 230)
PANEL_BG = (18, 18, 18)
PANEL_OUT = (220, 220, 220)

# ghost
GHOST_COLOUR = (180, 180, 180)

# derived sizes
BOARD_W_PX = BOARD_COLS * CELL_SIZE
BOARD_H_PX = BOARD_ROWS * CELL_SIZE
EXTRA_TOP_PX = SPAWN_BUFFER_ROWS * CELL_SIZE


# size of one player's layout (panels + board)
SINGLE_PLAY_WIDTH = (
    MARGIN + LEFT_PANEL_W + GAP + BOARD_W_PX + GAP + RIGHT_PANEL_W + MARGIN
)

# two layouts side by side
WINDOW_W = SINGLE_PLAY_WIDTH * 2
WINDOW_H = MARGIN + EXTRA_TOP_PX + BOARD_H_PX + MARGIN + 60  # + HUD

# separator
SEPARATOR_WIDTH = 4
SEPARATOR_COLOUR = (200, 200, 200)

# Player 1 origins (left side)
BOARD1_ORIGIN_X = MARGIN + LEFT_PANEL_W + GAP
BOARD1_ORIGIN_Y = MARGIN + EXTRA_TOP_PX

HOLD1_RECT = pygame.Rect(MARGIN, BOARD1_ORIGIN_Y, LEFT_PANEL_W, LEFT_PANEL_H)
NEXT1_RECT = pygame.Rect(
    MARGIN + LEFT_PANEL_W + GAP + BOARD_W_PX + GAP,
    BOARD1_ORIGIN_Y,
    RIGHT_PANEL_W,
    BOARD_H_PX,
)

# Player 2 origins (right side) – just shifted by SINGLE_PLAY_WIDTH
BOARD2_ORIGIN_X = BOARD1_ORIGIN_X + SINGLE_PLAY_WIDTH
BOARD2_ORIGIN_Y = BOARD1_ORIGIN_Y

HOLD2_RECT = HOLD1_RECT.move(SINGLE_PLAY_WIDTH, 0)
NEXT2_RECT = NEXT1_RECT.move(SINGLE_PLAY_WIDTH, 0)


AI_P1_ENABLED = False
AI_P2_ENABLED = True


def draw_grid(surface, origin_x, origin_y):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            x = origin_x + (col * CELL_SIZE)
            y = origin_y + (row * CELL_SIZE)

            pygame.draw.rect(
                surface, GRID_COLOUR, (x, y, CELL_SIZE, CELL_SIZE), width=1
            )  # surface, colour, (px, py, width, height), line thickness


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

            # skip over empty values in our box
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


# def draw_hud(surface, font):
#     msg = "Arrows move | Up rotate | Down soft | Space hard | R new piece | Esc quit"
#     surface.blit(font.render(msg, True, TEXT_COLOUR), (MARGIN, MARGIN + BOARD_ROWS * CELL_SIZE + 12))


def draw_panel(surface, rect, title, font):
    # panel body
    pygame.draw.rect(surface, PANEL_BG, rect)
    pygame.draw.rect(surface, PANEL_OUT, rect, width=2)

    # title
    title_surf = font.render(title, True, PANEL_OUT)
    surface.blit(title_surf, (rect.x + 8, rect.y + 8))


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

    PAD = 10  # pixels inside the preview rect
    avail_w = max(1, rect.width - PAD * 2)
    avail_h = max(1, rect.height - PAD * 2)

    cell = min(avail_w // bbox_w, avail_h // bbox_h)
    cell = max(6, min(30, int(cell * 0.75)))

    # centre the piece in the rect
    draw_w = bbox_w * cell
    draw_h = bbox_h * cell
    start_x = rect.x + (rect.width - draw_w) // 2
    start_y = rect.y + (rect.height - draw_h) // 2

    colour = ID_TO_COLOUR[LABEL_TO_ID[label]]

    # draw each filled cell within the bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if shape[r, c] == 0:
                continue
            x = start_x + (c - c0) * cell
            y = start_y + (r - r0) * cell
            pygame.draw.rect(surface, colour, (x, y, cell, cell))
            pygame.draw.rect(surface, GRID_COLOUR, (x, y, cell, cell), width=1)


def panel_inner(rect, pad=12):
    return pygame.Rect(
        rect.x + pad, rect.y + 32, rect.width - 2 * pad, rect.height - 32 - pad
    )


def draw_next_preview_list(surface, rect, labels):
    MARGIN = 8  # outer padding inside the panel
    SPACING = 8  # vertical gap between previews
    n = max(1, len(labels))

    # compute a neat box height for each preview
    usable_h = rect.height - 2 * MARGIN - SPACING * (n - 1)
    box_h = usable_h // n

    x = rect.x + MARGIN
    y = rect.y + MARGIN
    w = rect.width - 2 * MARGIN

    for label in labels:
        box = pygame.Rect(x, y, w, box_h)
        draw_piece_preview(surface, box, label, rotation=0)
        y += box_h + SPACING


def draw_ghost(surface, game, origin_x, origin_y):
    ghost = game.ghost_position()
    if not ghost:
        return

    left_x, ghost_y, shape = ghost
    label = game.current_piece[0]  # same label as the active piece
    colour = ID_TO_COLOUR[LABEL_TO_ID[label]]
    darker = tuple(max(0, int(c * 0.4)) for c in colour)  # 40% brightness

    h, w = shape.shape

    for r in range(h):
        for c in range(w):
            if shape[r, c] == 0:
                continue

            bx = left_x + c
            by = ghost_y + r
            if by < 0:
                continue  # ignore above-top cells

            x = origin_x + (bx * CELL_SIZE)
            y = origin_y + (by * CELL_SIZE)

            pygame.draw.rect(surface, darker, (x, y, CELL_SIZE, CELL_SIZE))


def draw_counters(
    surface,
    game,
    rect,
    font,
    frozen_apm=None,
    frozen_pps=None,
):
    y = rect.bottom + 6
    x = rect.x + 8

    combo = game.combo_count
    b2b = game.b2b_chain_len
    total_atk = game.total_attack_sent
    apm = frozen_apm if frozen_apm is not None else game.get_apm()
    pps = frozen_pps if frozen_pps is not None else game.get_pps()
    pieces = getattr(game, "pieces_locked")

    combo_surf = font.render(f"COMBO: {combo}", True, PANEL_OUT)
    surface.blit(combo_surf, (x, y))
    y += combo_surf.get_height() + 2

    b2b_surf = font.render(f"B2B: {b2b}", True, PANEL_OUT)
    surface.blit(b2b_surf, (x, y))
    y += b2b_surf.get_height() + 2

    atk_surf = font.render(f"ATK: {total_atk}", True, PANEL_OUT)
    surface.blit(atk_surf, (x, y))
    y += atk_surf.get_height() + 2

    pps_surf = font.render(f"PPS: {pps:.2f}", True, PANEL_OUT)
    surface.blit(pps_surf, (x, y))
    y += pps_surf.get_height() + 2

    apm_surf = font.render(f"APM: {apm:.1f}", True, PANEL_OUT)
    surface.blit(apm_surf, (x, y))
    y += apm_surf.get_height() + 8

    incoming = game.incoming_garbage_lines()
    in_surf = font.render(f"IN: {incoming}", True, PANEL_OUT)
    surface.blit(in_surf, (x, y))
    y += in_surf.get_height() + 8

    pieces_surf = font.render(f"PIECES: {pieces}", True, PANEL_OUT)
    surface.blit(pieces_surf, (x, y))


def handle_keydown_p1(game, key):
    if key == pygame.K_LEFT:
        game.move_left()
    elif key == pygame.K_RIGHT:
        game.move_right()
    elif key == pygame.K_e:
        game.rotate_clockwise()
    elif key == pygame.K_w:
        game.rotate_counterclockwise()
    elif key == pygame.K_LSHIFT or key == pygame.K_q:
        game.rotate_180()
    elif key == pygame.K_r:
        game.hold()
    elif key == pygame.K_y:
        game.spawn_random()
    elif key == pygame.K_SPACE:
        game.hard_drop()


def handle_keydown_p2(game, key):
    if key == pygame.K_j:
        game.move_left()
    elif key == pygame.K_l:
        game.move_right()
    elif key == pygame.K_s:
        game.rotate_clockwise()
    elif key == pygame.K_a:
        game.rotate_counterclockwise()
    elif key == pygame.K_f:
        game.hard_drop()
    elif key == pygame.K_d:
        game.hold()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def apply_ai_action_instant(
    game, opponent_game, ai, pps_controller, pieces_placed, max_inputs_per_piece=64
):
    if not pps_controller.can_place():
        return pieces_placed

    for _ in range(max_inputs_per_piece):
        if game.game_over or game.current_piece is None:
            return pieces_placed

        action = ai.choose_action(game, opponent_game=opponent_game)

        if action == "hold":
            game.hold()
            if game.current_piece is None:
                return pieces_placed

        elif action == "hard_drop":
            game.hard_drop()
            pps_controller.on_place()
            return pieces_placed + 1

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


class PPSController:
    def __init__(self, fixed_pps):
        self.fixed_pps = fixed_pps
        self.interval = 1.0 / max(fixed_pps, 1e-6)
        self._timer = 0.0

    def update(self, dt):
        self._timer += dt

    def can_place(self):
        return self._timer >= self.interval

    def on_place(self):
        self._timer -= self.interval


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Tetris Single-Player Env")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)
    title_font = pygame.font.SysFont(None, 28)

    # SEEDED BAGS
    MATCH_SEED = 2
    bag_rng1 = random.Random(MATCH_SEED)
    bag_rng2 = random.Random(MATCH_SEED)

    board1 = Board(rows=BOARD_ROWS, cols=BOARD_COLS)
    game1 = Game(board1, bag_rng=bag_rng1)

    game1.spawn_random()  # start with a random piece

    board2 = Board(rows=BOARD_ROWS, cols=BOARD_COLS)
    game2 = Game(board2, bag_rng=bag_rng2)
    game2.spawn_random()  # start with a random piece

    # AI INSTANTIATION
    # ai_p1 = TetrisAI(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # ai_p1 = TetrisAI_Opp(ply=2)
    ai_p1 = TetrisAI2(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # ai_p1 = TetrisAI2_Opp(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # ai_p2 = TetrisAI(ply=2)
    # ai_p2 = TetrisAI_Opp(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # ai_p2 = TetrisAI2(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    ai_p2 = TetrisAI2_Opp(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # pps control
    ai_p1_pps = PPSController(30)
    ai_p2_pps = PPSController(3)

    p1_elapsed = 0
    p1_pieces_placed = 0

    p2_elapsed = 0
    p2_pieces_placed = 0

    gravity_timer = 0.0
    gravity_interval = 1.0 / max(GRAVITY_HZ, 1e-6)

    # garbage control
    sim_time = 0.0

    running = True

    softdrop_timer_p1 = 0.0
    softdrop_timer_p2 = 0.0
    softdrop_interval = 0.05  # 50 ms

    DAS = 0.100  # delay auto shift
    ARR = 0.01  # auto repeat rate
    SOFT_DROP_SPEED = 10

    p1_left_held = 0.0
    p1_right_held = 0.0
    p1_horiz_repeat_timer = 0.0

    p2_left_held = 0.0
    p2_right_held = 0.0
    p2_horiz_repeat_timer = 0.0

    match_over = False

    final_apm1 = None
    final_apm2 = None

    final_pps1 = None
    final_pps2 = None

    while running:
        if not match_over and (game1.game_over or game2.game_over):
            match_over = True

            final_apm1 = game1.get_apm()
            final_apm2 = game2.get_apm()

            final_pps1 = game1.get_pps()
            final_pps2 = game2.get_pps()
        delta_t = clock.tick(FPS) / 1000.0
        gravity_timer += delta_t

        # garbage controler
        sim_time += delta_t
        # Move delayed attacks into each player's pending queue when due
        game1.process_incoming_attacks(sim_time)
        game2.process_incoming_attacks(sim_time)

        # pps control
        # p1_elapsed += delta_t
        # p2_elapsed += delta_t
        # avg_p1_pps = (p1_pieces_placed / p1_elapsed) if p1_elapsed > 0 else 0
        # avg_p2_pps = (p2_pieces_placed / p2_elapsed) if p2_elapsed > 0 else 0

        # height_p1 = board1.stack_height()
        # in_danger_p1 = height_p1 >= 10

        # in_danger_p2 = (game2.incoming_garbage_lines() >= 6)

        # ai_p1_pps.update(delta_t, avg_p1_pps, in_danger_p1)
        # ai_p2_pps.update(delta_t, avg_p2_pps, in_danger_p2)

        p1_elapsed += delta_t
        p2_elapsed += delta_t

        ai_p1_pps.update(delta_t)
        ai_p2_pps.update(delta_t)

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_TAB:
                    global AI_P2_ENABLED
                    AI_P2_ENABLED = not AI_P2_ENABLED

                else:
                    if not AI_P1_ENABLED:
                        handle_keydown_p1(game1, event.key)
                    if not AI_P2_ENABLED:
                        handle_keydown_p2(game2, event.key)

        # continuous soft drop with speed limit
        keys = pygame.key.get_pressed()

        if not AI_P1_ENABLED:
            # P1: horizontal DAS + ARR
            if keys[pygame.K_LEFT]:
                p1_left_held += delta_t
                p1_right_held = 0.0  # prefer the last direction, cancel opposite
            else:
                p1_left_held = 0.0

            if keys[pygame.K_RIGHT]:
                p1_right_held += delta_t
                p1_left_held = 0.0
            else:
                p1_right_held = 0.0

            # choose which direction is active for auto-repeat
            p1_dir = 0

            if p1_left_held > 0 and p1_right_held == 0:
                p1_dir = -1

            elif p1_right_held > 0 and p1_left_held == 0:
                p1_dir = 1

            else:
                p1_horiz_repeat_timer = 0.0

            if p1_dir != 0:
                if (p1_left_held > DAS) or (p1_right_held > DAS):
                    p1_horiz_repeat_timer += delta_t

                    while p1_horiz_repeat_timer >= ARR:
                        p1_horiz_repeat_timer -= ARR

                        if p1_dir == -1:
                            game1.move_left()
                        else:
                            game1.move_right()

            if keys[pygame.K_DOWN]:
                for _ in range(SOFT_DROP_SPEED):
                    game1.soft_drop()

        if not AI_P2_ENABLED:
            # P2: horizontal DAS + ARR
            if keys[pygame.K_j]:
                p2_left_held += delta_t
                p2_right_held = 0.0
            else:
                p2_left_held = 0.0

            if keys[pygame.K_l]:
                p2_right_held += delta_t
                p2_left_held = 0.0
            else:
                p2_right_held = 0.0

            p2_dir = 0

            if p2_left_held > 0 and p2_right_held == 0:
                p2_dir = -1

            elif p2_right_held > 0 and p2_left_held == 0:
                p2_dir = 1

            else:
                p2_horiz_repeat_timer = 0.0

            if p2_dir != 0:
                if (p2_left_held > DAS) or (p2_right_held > DAS):
                    p2_horiz_repeat_timer += delta_t

                    while p2_horiz_repeat_timer >= ARR:
                        p2_horiz_repeat_timer -= ARR

                        if p2_dir == -1:
                            game2.move_left()
                        else:
                            game2.move_right()

            # P2 vertical movement
            if keys[pygame.K_k]:
                softdrop_timer_p2 += delta_t
                if softdrop_timer_p2 >= softdrop_interval:
                    softdrop_timer_p2 -= softdrop_interval
                    game2.soft_drop()
            else:
                softdrop_timer_p2 = 0.0

        # gravity tick (natural falling + lock delay)
        while gravity_timer >= gravity_interval:
            gravity_timer -= gravity_interval

            if not AI_P1_ENABLED and not game1.game_over:
                game1.gravity_tick()

            if not AI_P2_ENABLED and not game2.game_over:
                game2.gravity_tick()

        # resolve attacks into pending garbage
        atk1 = game1.consume_attack()
        atk2 = game2.consume_attack()

        if atk1 > 0 and not game2.game_over:
            hole = random.randrange(board2.cols)
            game2.receive_attack(atk1, hole, sim_time)

        if atk2 > 0 and not game1.game_over:
            hole = random.randrange(board1.cols)
            game1.receive_attack(atk2, hole, sim_time)

        # Player 1 - apply pending garbage between pieces
        if not game1.game_over and not game1.current_piece:
            if game1.skip_garbage_apply_once:
                game1.skip_garbage_apply_once = False
            else:
                packets = game1.pop_garbage_to_apply(game1.MAX_GARBAGE_PER_PIECE)
                for lines, hole in packets:
                    board1.add_garbage_rows(lines, hole_col=hole)

            game1.spawn_random()

        # Player 2
        if not game2.game_over and not game2.current_piece:
            if game2.skip_garbage_apply_once:
                game2.skip_garbage_apply_once = False
            else:
                packets = game2.pop_garbage_to_apply(game2.MAX_GARBAGE_PER_PIECE)
                for lines, hole in packets:
                    board2.add_garbage_rows(lines, hole_col=hole)

            game2.spawn_random()

        # if AI_P1_ENABLED and not game1.game_over:
        #     action = ai_p1.choose_action(game1, opponent_game=game2)

        #     if action == "hold":
        #         game1.hold()

        #     elif action == "hard_drop":
        #         if ai_p1_pps.can_place():
        #             game1.hard_drop()
        #             ai_p1_pps.on_place()
        #             p1_pieces_placed += 1

        # if AI_P2_ENABLED and not game2.game_over:
        #     next2 = game2.peek_next(1)
        #     next_label2 = next2[0] if next2 else None
        #     action = ai_p2.choose_action(game2, opponent_game=game1)

        #     if action == "hold":
        #         game2.hold()

        #     elif action == "hard_drop":
        #         if ai_p2_pps.can_place():
        #             game2.hard_drop()
        #             ai_p2_pps.on_place()
        #             p2_pieces_placed += 1

        #     elif action == "left":
        #         game2.move_left()
        #     elif action == "right":
        #         game2.move_right()
        #     elif action == "rot_cw":
        #         game2.rotate_clockwise()
        #     elif action == "rot_ccw":
        #         game2.rotate_counterclockwise()
        #     elif action == "rot_180":
        #         game2.rotate_180()

        # if AI_P1_ENABLED and not game1.game_over:
        if AI_P1_ENABLED and not match_over:
            while ai_p1_pps.can_place():
                before = p1_pieces_placed
                p1_pieces_placed = apply_ai_action_instant(
                    game1, game2, ai_p1, ai_p1_pps, p1_pieces_placed
                )
                if p1_pieces_placed == before:
                    break

        if AI_P2_ENABLED and not match_over:
            while ai_p2_pps.can_place():
                before = p2_pieces_placed
                p2_pieces_placed = apply_ai_action_instant(
                    game2, game1, ai_p2, ai_p2_pps, p2_pieces_placed
                )
                if p2_pieces_placed == before:
                    break

        # draw
        screen.fill(BG_COLOUR)

        # Player 1 (left)
        draw_panel(screen, HOLD1_RECT, "HOLD", title_font)
        if game1.hold_label is not None:
            draw_piece_preview(
                screen, panel_inner(HOLD1_RECT), game1.hold_label, rotation=0
            )

        draw_counters(
            screen,
            game1,
            HOLD1_RECT,
            title_font,
            frozen_apm=final_apm1,
            frozen_pps=final_pps1,
        )

        draw_panel(screen, NEXT1_RECT, "NEXT", title_font)
        inner1 = panel_inner(NEXT1_RECT)
        next_labels1 = game1.peek_next(5)
        draw_next_preview_list(screen, inner1, next_labels1)

        draw_ghost(screen, game1, BOARD1_ORIGIN_X, BOARD1_ORIGIN_Y)
        draw_grid(screen, BOARD1_ORIGIN_X, BOARD1_ORIGIN_Y)
        draw_board(screen, board1, BOARD1_ORIGIN_X, BOARD1_ORIGIN_Y)
        draw_piece(screen, game1.current_piece, BOARD1_ORIGIN_X, BOARD1_ORIGIN_Y)

        # Player 2 (right)
        draw_panel(screen, HOLD2_RECT, "HOLD", title_font)
        if game2.hold_label is not None:
            draw_piece_preview(
                screen, panel_inner(HOLD2_RECT), game2.hold_label, rotation=0
            )

        draw_counters(
            screen,
            game2,
            HOLD2_RECT,
            title_font,
            frozen_apm=final_apm2,
            frozen_pps=final_pps2,
        )

        draw_panel(screen, NEXT2_RECT, "NEXT", title_font)
        inner2 = panel_inner(NEXT2_RECT)
        next_labels2 = game2.peek_next(5)
        draw_next_preview_list(screen, inner2, next_labels2)

        draw_ghost(screen, game2, BOARD2_ORIGIN_X, BOARD2_ORIGIN_Y)
        draw_grid(screen, BOARD2_ORIGIN_X, BOARD2_ORIGIN_Y)
        draw_board(screen, board2, BOARD2_ORIGIN_X, BOARD2_ORIGIN_Y)
        draw_piece(screen, game2.current_piece, BOARD2_ORIGIN_X, BOARD2_ORIGIN_Y)

        # separator
        sep_x = SINGLE_PLAY_WIDTH - SEPARATOR_WIDTH // 2
        pygame.draw.rect(
            screen, SEPARATOR_COLOUR, (sep_x, 0, SEPARATOR_WIDTH, WINDOW_H)
        )

        winner_text = None

        if game1.game_over:
            winner_text = "PLAYER 2 WINS"

        elif game2.game_over:
            winner_text = "PLAYER 1 WINS"

        if winner_text:
            overlay = pygame.Surface((WINDOW_W, WINDOW_H))
            overlay.set_alpha(180)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))

            win_font = pygame.font.SysFont(None, 72)

            text = win_font.render(winner_text, True, (255, 255, 255))
            rect = text.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2))

            screen.blit(text, rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
