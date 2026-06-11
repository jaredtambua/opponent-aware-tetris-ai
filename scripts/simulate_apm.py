import time
import random

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

from src.game.bitboard import BitBoard
from src.search.search import column_heights

# single-board configuration
BOARD_ROWS = 20
BOARD_COLS = 10

# simulation config
FPS = 1000
DT = 1.0 / FPS
TARGET_PIECES = 1000
START_SEED = 0
FIXED_PPS = 1000
NUM_GAMES = 10


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


# read current stack height
def get_max_height(board):
    bb = BitBoard.from_board(board)
    heights = column_heights(bb)
    return max(heights) if heights else 0


# choose which ai to test
def create_ai():
    # ai1 controls
    # return AI1UP(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # return AI1Down(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)
    # return AI1OPTIMISTIC(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai1 standard
    # return TetrisAI(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai1 opponent-aware
    # return TetrisAI_Opp(ply=2, depth_beam_width=10, gamma=0.95, well_col=9)

    # ai2 controls
    # return AI2UP(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    # return AI2Down(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)
    return AI2OPTIMISTIC(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # ai2 standard
    # return TetrisAI2(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)

    # ai2 opponent-aware
    # return TetrisAI2_Opp(ply=6, depth_beam_width=14, gamma=0.95, well_col=9)


# support both ai call signatures
def choose_ai_action(game, ai):
    try:
        return ai.choose_action(game, opponent_game=None)
    except TypeError:
        return ai.choose_action(game)


# run ai inputs until it places one piece
def run_ai_action_instant(
    game, ai, pps_controller, pieces_placed_ref, max_inputs_per_piece=64
):
    if game.game_over or not pps_controller.can_place():
        return False

    for _ in range(max_inputs_per_piece):
        if game.game_over or game.current_piece is None:
            return False

        action = choose_ai_action(game, ai)

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

        elif action is None:
            return False

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


# run one timed apm simulation
def simulate_apm(seed=START_SEED, target_pieces=TARGET_PIECES, fixed_pps=FIXED_PPS):
    # use a seeded bag rng for repeatable runs
    bag_rng = random.Random(seed)

    board = Board(rows=BOARD_ROWS, cols=BOARD_COLS)
    game = Game(board, bag_rng=bag_rng)
    game.spawn_random()

    ai = create_ai()
    ai_pps = PPSController(fixed_pps)

    sim_time = 0.0
    pieces_placed = [0]

    # track average stack height after placements
    max_height_sum = 0
    max_height_samples = 0

    run_start = time.perf_counter()
    last_time = run_start

    # main simulation loop
    while not game.game_over and pieces_placed[0] < target_pieces:
        frame_start = time.perf_counter()
        dt = frame_start - last_time
        last_time = frame_start
        sim_time += dt

        game.process_incoming_attacks(sim_time)
        ai_pps.update(dt)

        spawn_if_needed(game, board)

        # place as many pieces as timing allows this frame
        while (
            ai_pps.can_place()
            and not game.game_over
            and pieces_placed[0] < target_pieces
        ):
            placed = run_ai_action_instant(game, ai, ai_pps, pieces_placed)
            if not placed:
                break

            current_max_height = get_max_height(board)
            max_height_sum += current_max_height
            max_height_samples += 1

            spawn_if_needed(game, board)

        elapsed = time.perf_counter() - frame_start
        if elapsed < DT:
            time.sleep(DT - elapsed)

    run_time_seconds = time.perf_counter() - run_start
    run_time_minutes = run_time_seconds / 60.0 if run_time_seconds > 0 else 0.0

    apm = game.total_attack_sent / run_time_minutes if run_time_minutes > 0 else 0.0
    pps = pieces_placed[0] / run_time_seconds if run_time_seconds > 0 else 0.0

    return {
        "seed": seed,
        "target_pieces": target_pieces,
        "pieces_placed": pieces_placed[0],
        "total_attack": game.total_attack_sent,
        "run_time_seconds": run_time_seconds,
        "sim_time": sim_time,
        "apm": apm,
        "pps": pps,
        "game_over": game.game_over,
        "max_height_sum": max_height_sum,
        "max_height_samples": max_height_samples,
    }


# summarize all runs
if __name__ == "__main__":
    summaries = [simulate_apm(seed=START_SEED + i) for i in range(NUM_GAMES)]

    total_attack = sum(s["total_attack"] for s in summaries)
    total_pieces = sum(s["pieces_placed"] for s in summaries)
    total_runtime = sum(s["run_time_seconds"] for s in summaries)
    total_sim_time = sum(s["sim_time"] for s in summaries)
    total_max_height_sum = sum(s["max_height_sum"] for s in summaries)
    total_max_height_samples = sum(s["max_height_samples"] for s in summaries)

    avg_apm = sum(s["apm"] for s in summaries) / NUM_GAMES
    avg_pps = sum(s["pps"] for s in summaries) / NUM_GAMES
    avg_attack = total_attack / NUM_GAMES
    avg_pieces = total_pieces / NUM_GAMES
    avg_runtime = total_runtime / NUM_GAMES
    avg_sim_time = total_sim_time / NUM_GAMES

    global_avg_max_height = (
        total_max_height_sum / total_max_height_samples
        if total_max_height_samples > 0
        else 0.0
    )

    num_game_overs = sum(1 for s in summaries if s["game_over"])

    print(f"Single AI APM Simulation over {NUM_GAMES} games")
    print("Start seed:", START_SEED)
    print("Target pieces per game:", TARGET_PIECES)
    print("Average pieces placed:", avg_pieces)
    print("Games ended by top out:", num_game_overs)
    print("Average total attack:", avg_attack)
    print("Average runtime (seconds):", avg_runtime)
    print("Average sim time:", avg_sim_time)
    print("Average APM:", avg_apm)
    print("Average PPS:", avg_pps)
    print("Average Max Height:", global_avg_max_height)
