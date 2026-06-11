import random
import time

from src.game.board import Board
from src.game.game import Game

from src.ai.fast_ai import TetrisAI
from src.ai.efficient_ai import TetrisAI2
from src.ai.fast_ai_opponent_aware import TetrisAI_Opp
from src.ai.efficient_ai_opponent_aware import TetrisAI2_Opp

# from ai1_down_only import AI1Down
# from ai1_up_only import AI1UP
# from ai1_optimistic_only import AI1OPTIMISTIC

# from ai2_down_only import AI2Down
# from ai2_up_only import AI2UP
# from ai2_optimistic_only import AI2OPTIMISTIC

# two-board configuration
BOARD_ROWS = 20
BOARD_COLS = 10

# series config
NUM_GAMES = 100
START_SEED = 0
FPS = 1000
DT = 1.0 / FPS
DRAW_AFTER_P2_PIECES = 1000

# pps setting
FIXED_PPS = 100


# pps timing helper
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


# choose which ai p1 uses
def create_ai_p1():
    # ai1 controls
    # return AI1UP(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # return AI1Down(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # return AI1OPTIMISTIC(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai1 standard
    # return TetrisAI(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai1 opponent-aware
    return TetrisAI_Opp(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai2 controls
    # return AI2UP(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # return AI2Down(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # return AI2OPTIMISTIC(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # ai2 standard
    # return TetrisAI2(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # ai2 opponent-aware
    # return TetrisAI2_Opp(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)


# choose which ai p2 uses
def create_ai_p2():
    # ai1 controls
    # return AI1Down(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # return AI1UP(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # return AI1OPTIMISTIC(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai1 standard
    # return TetrisAI(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai1 opponent-aware
    # return TetrisAI_Opp(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai2 controls
    # return AI2Down(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # return AI2UP(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # return AI2OPTIMISTIC(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # ai2 standard
    return TetrisAI2(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # ai2 opponent-aware
    # return TetrisAI2_Opp(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)


# run ai inputs until it places one piece
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

        elif action is None:
            return pieces_placed

        else:
            raise ValueError(f"unknown ai action: {action}")

    raise RuntimeError(
        f"ai failed to reach hard_drop within {max_inputs_per_piece} inputs."
    )


# spawn next piece and apply pending garbage
def spawn_if_needed(game, board):
    if game.game_over or game.current_piece:
        return

    if game.skip_garbage_apply_once:
        game.skip_garbage_apply_once = False
    else:
        packets = game.pop_garbage_to_apply(game.MAX_GARBAGE_PER_PIECE)
        for lines, hole in packets:
            board.add_garbage_rows(lines, hole_col=hole)

    game.spawn_random()


# exchange pending attack between players
def resolve_attacks(game1, board1, game2, board2, sim_time, hole_rng):
    atk1 = game1.consume_attack()
    atk2 = game2.consume_attack()

    if atk1 > 0 and not game2.game_over:
        hole = hole_rng.randrange(board2.cols)
        game2.receive_attack(atk1, hole, sim_time)

    if atk2 > 0 and not game1.game_over:
        hole = hole_rng.randrange(board1.cols)
        game1.receive_attack(atk2, hole, sim_time)


# decide winner after gameover or timeout
def choose_winner(game1, game2):
    if game1.game_over and not game2.game_over:
        return "p2", "gameover"
    if game2.game_over and not game1.game_over:
        return "p1", "gameover"
    if game1.game_over and game2.game_over:
        return "draw", "gameover"

    if game1.total_attack_sent > game2.total_attack_sent:
        return "p1", "tiebreak_attack"
    if game2.total_attack_sent > game1.total_attack_sent:
        return "p2", "tiebreak_attack"

    h1 = game1.board.stack_height()
    h2 = game2.board.stack_height()

    if h1 < h2:
        return "p1", "tiebreak_height"
    if h2 < h1:
        return "p2", "tiebreak_height"

    return "draw", "tiebreak_draw"


# run one head-to-head match
def simulate_match(
    seed, draw_after_p2_pieces=DRAW_AFTER_P2_PIECES, fixed_pps=FIXED_PPS
):
    bag_rng1 = random.Random(seed)
    bag_rng2 = random.Random(seed)
    hole_rng = random.Random(seed + 10000)

    board1 = Board(rows=BOARD_ROWS, cols=BOARD_COLS)
    board2 = Board(rows=BOARD_ROWS, cols=BOARD_COLS)

    game1 = Game(board1, bag_rng=bag_rng1)
    game2 = Game(board2, bag_rng=bag_rng2)

    game1.spawn_random()
    game2.spawn_random()

    ai_p1 = create_ai_p1()
    ai_p2 = create_ai_p2()

    # pps control
    ai_p1_pps = PPSController(18.4)
    ai_p2_pps = PPSController(8)

    sim_time = 0.0
    p1_pieces_placed = 0
    p2_pieces_placed = 0

    last_time = time.perf_counter()

    # main match loop
    while True:
        frame_start = time.perf_counter()
        dt = frame_start - last_time
        last_time = frame_start

        if game1.game_over or game2.game_over:
            break
        if p2_pieces_placed >= draw_after_p2_pieces:
            break

        sim_time += dt

        game1.process_incoming_attacks(sim_time)
        game2.process_incoming_attacks(sim_time)

        ai_p1_pps.update(dt)
        ai_p2_pps.update(dt)

        resolve_attacks(game1, board1, game2, board2, sim_time, hole_rng)

        spawn_if_needed(game1, board1)
        spawn_if_needed(game2, board2)

        if not game1.game_over:
            while ai_p1_pps.can_place():
                before = p1_pieces_placed
                p1_pieces_placed = apply_ai_action_instant(
                    game1, game2, ai_p1, ai_p1_pps, p1_pieces_placed
                )
                if p1_pieces_placed == before:
                    break

        if not game2.game_over:
            while ai_p2_pps.can_place():
                before = p2_pieces_placed
                p2_pieces_placed = apply_ai_action_instant(
                    game2, game1, ai_p2, ai_p2_pps, p2_pieces_placed
                )
                if p2_pieces_placed == before:
                    break

        if p2_pieces_placed >= draw_after_p2_pieces:
            break

        elapsed = time.perf_counter() - frame_start
        if elapsed < DT:
            time.sleep(DT - elapsed)

    winner, win_reason = choose_winner(game1, game2)

    return {
        "seed": seed,
        "winner": winner,
        "win_reason": win_reason,
        "sim_time": sim_time,
        "p1_attack": game1.total_attack_sent,
        "p2_attack": game2.total_attack_sent,
        "p1_pieces": p1_pieces_placed,
        "p2_pieces": p2_pieces_placed,
    }


# run multiple matches
def run_series(
    num_games=NUM_GAMES,
    start_seed=START_SEED,
    draw_after_p2_pieces=DRAW_AFTER_P2_PIECES,
    fixed_pps=FIXED_PPS,
):
    results = []
    run_start = time.perf_counter()

    for i, seed in enumerate(range(start_seed, start_seed + num_games)):
        result = simulate_match(
            seed,
            draw_after_p2_pieces=draw_after_p2_pieces,
            fixed_pps=fixed_pps,
        )
        results.append(result)

        if (i + 1) % 5 == 0 or (i + 1) == num_games:
            pct = (i + 1) / num_games * 100
            p1_wins = sum(r["winner"] == "p1" for r in results)
            p2_wins = sum(r["winner"] == "p2" for r in results)
            draws = sum(r["winner"] == "draw" for r in results)

            gameover_wins = sum(r["win_reason"] == "gameover" for r in results)
            tiebreak_attack_wins = sum(
                r["win_reason"] == "tiebreak_attack" for r in results
            )
            tiebreak_height_wins = sum(
                r["win_reason"] == "tiebreak_height" for r in results
            )
            tiebreak_draws = sum(r["win_reason"] == "tiebreak_draw" for r in results)

            print(
                f"[{i+1}/{num_games}] "
                f"{pct:.1f}% | "
                f"P1: {p1_wins/(i+1):.2%} "
                f"P2: {p2_wins/(i+1):.2%} "
                f"D: {draws/(i+1):.2%} | "
                f"gameover={gameover_wins} "
                f"tb_atk={tiebreak_attack_wins} "
                f"tb_h={tiebreak_height_wins} "
                f"tb_draw={tiebreak_draws}"
            )

    run_time = time.perf_counter() - run_start
    run_time_minutes = run_time / 60.0 if run_time > 0 else 0.0

    p1_wins = sum(r["winner"] == "p1" for r in results)
    p2_wins = sum(r["winner"] == "p2" for r in results)
    draws = sum(r["winner"] == "draw" for r in results)

    total_time = sum(r["sim_time"] for r in results)
    total_p1_attack = sum(r["p1_attack"] for r in results)
    total_p2_attack = sum(r["p2_attack"] for r in results)
    total_p1_pieces = sum(r["p1_pieces"] for r in results)
    total_p2_pieces = sum(r["p2_pieces"] for r in results)

    return {
        "games": num_games,
        "fixed_pps": fixed_pps,
        "draw_after_p2_pieces": draw_after_p2_pieces,
        "p1_win_rate": p1_wins / num_games,
        "p2_win_rate": p2_wins / num_games,
        "draw_rate": draws / num_games,
        "total_sim_time": total_time,
        "run_time_seconds": run_time,
        "run_time_minutes": run_time_minutes,
        "p1_pps": total_p1_pieces / run_time if run_time > 0 else 0.0,
        "p2_pps": total_p2_pieces / run_time if run_time > 0 else 0.0,
        "p1_apm": total_p1_attack / run_time_minutes if run_time_minutes > 0 else 0.0,
        "p2_apm": total_p2_attack / run_time_minutes if run_time_minutes > 0 else 0.0,
        "p1_total_attack": total_p1_attack,
        "p2_total_attack": total_p2_attack,
        "p1_total_pieces": total_p1_pieces,
        "p2_total_pieces": total_p2_pieces,
    }


# entry point
if __name__ == "__main__":
    summary = run_series()

    print("AI1OPTIMISTIC vs AI2")
    print("Games:", summary["games"])
    print("P1 win rate:", summary["p1_win_rate"])
    print("P2 win rate:", summary["p2_win_rate"])
    print("Draw rate:", summary["draw_rate"])
    print("Global runtime (seconds):", summary["run_time_seconds"])
    print("P1 total attack:", summary["p1_total_attack"])
    print("P2 total attack:", summary["p2_total_attack"])
    print("P1 total pieces:", summary["p1_total_pieces"])
    print("P2 total pieces:", summary["p2_total_pieces"])
    print("P1 APM:", summary["p1_apm"])
    print("P2 APM:", summary["p2_apm"])
    print("P1 PPS:", summary["p1_pps"])
    print("P2 PPS:", summary["p2_pps"])
