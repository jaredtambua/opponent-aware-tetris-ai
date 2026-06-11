import argparse
import copy
import math
import random

import numpy as np

from src.game.board import Board
from src.game.game import Game
from src.ai.efficient_ai import TetrisAI2
from scripts.survival_simulation import _enqueue_garbage_pressure
from src.game.bitboard import BitBoard
from src.search.search import column_heights

# basic simulation config
BOARD_ROWS = 22
BOARD_COLS = 10
MAX_PIECES = 1000
EVAL_SEEDS = [1212, 2323, 3434, 4545, 5656]

# garbage pressure settings
PRESSURE_LINES = 0
PRESSURE_MESSINESS = 0
PRESSURE_EVERY_PIECES = 0

# encourage cancellation
HOLD_I_KEEP_BONUS = 0.1
READY_WELL_MIN_DEPTH = 4
READY_WELL_BONUS = 0.4
LONGEST_B2B_BONUS = 20

GENE_KEYS = [
    "HOLES",
    "FILL_WELL",
    "WELL_HEIGHT",
    "MAX_HEIGHT",
    "TOTAL_HEIGHT",
    "ROW_TRANSITIONS",
    "BUMPINESS",
    "NON_DEEPEST_WELL",
    "HOLD_ACTION",
    "TETRIS",
    "ANY_CLEAR_PENALTY",
    "HOLD_I",
]

BASE_GENOME = {
    "HOLES": 543.2205,
    "FILL_WELL": 0.0,
    "WELL_HEIGHT": 0.0,
    "MAX_HEIGHT": 40,
    "TOTAL_HEIGHT": 3.6011,
    "ROW_TRANSITIONS": 5.0,
    "BUMPINESS": 100,
    "NON_DEEPEST_WELL": 112.3126,
    "HOLD_ACTION": 4.9953,
    "TETRIS": 0.0627,
    "ANY_CLEAR_PENALTY": 0.0,
    "HOLD_I": 0.0,
}


# ai that always uses "safe" weights
class SafeOnlyAI2(TetrisAI2):
    def __init__(self, genome, ply=6, well_col=9, gamma=0.95, depth_beam_width=14):
        super().__init__(
            ply=ply,
            well_col=well_col,
            gamma=gamma,
            depth_beam_width=depth_beam_width,
        )

        # full_genome = dict(genome)
        # full_genome["FILL_WELL"] = 0.0
        # full_genome["WELL_HEIGHT"] = 0.0

        self.SAFE = copy.deepcopy(genome)
        self.W_DANGER = copy.deepcopy(genome)
        self.W = self.SAFE
        self.in_danger = False

    def choose_action(self, game, opponent_game=None):
        self.in_danger = False
        self.W = self.SAFE
        return super().choose_action(game, opponent_game)


# clamp negative values to zero
def clamp_nonnegative(x):
    if x < 0.0:
        return 0.0
    return float(x)


# set all random seeds for reproducibility
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)


# apply a single action to the game
def apply_action(game, action):
    if action == "hold":
        game.hold()
    elif action == "hard_drop":
        game.hard_drop()
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


def play_one_piece(game, ai):
    while game.current_piece and not game.game_over:
        action = ai.choose_action(game)
        if action is None:
            break
        apply_action(game, action)
        if action == "hard_drop":
            return True
    return False


# run one full simulation for a genome
def simulate_genome(genome, seed, ply=6):
    set_all_seeds(seed)

    board = Board(rows=BOARD_ROWS, cols=BOARD_COLS)
    game = Game(board)
    ai = SafeOnlyAI2(genome=genome, ply=ply)

    pieces_placed = 0
    sim_time = 0.0
    hold_i_steps = 0
    ready_well_steps = 0
    longest_b2b_chain_len = 0

    game.spawn_random()

    while not game.game_over and pieces_placed < MAX_PIECES:
        # process incoming garbage
        game.process_incoming_attacks(sim_time)

        had_piece = bool(game.current_piece)
        placed_now = play_one_piece(game, ai)

        if game.game_over:
            break

        # count placed pieces
        if had_piece and placed_now and not game.current_piece:
            pieces_placed += 1
            sim_time += 1.0

            longest_b2b_chain_len = max(longest_b2b_chain_len, game.b2b_chain_len)

            if game.hold_label == "I":
                hold_i_steps += 1

            bb = BitBoard.from_board(board)
            if deepest_well_depth(bb) >= READY_WELL_MIN_DEPTH:
                ready_well_steps += 1

            # periodically inject garbage pressure
            if PRESSURE_EVERY_PIECES > 0 and (
                pieces_placed % PRESSURE_EVERY_PIECES == 0
            ):
                _enqueue_garbage_pressure(
                    game=game,
                    cols=board.cols,
                    lines=PRESSURE_LINES,
                    messiness=PRESSURE_MESSINESS,
                    sim_time=sim_time,
                )

        # handle garbage application + spawning
        if not game.game_over and not game.current_piece:
            if game.skip_garbage_apply_once:
                game.skip_garbage_apply_once = False
            else:
                packets = game.pop_garbage_to_apply(game.MAX_GARBAGE_PER_PIECE)
                for lines, hole in packets:
                    board.add_garbage_rows(lines, hole_col=hole)

            game.spawn_random()

        if not placed_now and game.current_piece:
            break

    fitness = (
        LONGEST_B2B_BONUS * longest_b2b_chain_len
        + HOLD_I_KEEP_BONUS * hold_i_steps
        + READY_WELL_BONUS * ready_well_steps
    )

    return fitness


def simulate_genome_with_stats(genome, seed, ply=6):
    set_all_seeds(seed)

    board = Board(rows=BOARD_ROWS, cols=BOARD_COLS)
    game = Game(board)
    ai = SafeOnlyAI2(genome=genome, ply=ply)

    pieces_placed = 0
    sim_time = 0.0
    hold_i_steps = 0
    ready_well_steps = 0
    longest_b2b_chain_len = 0

    game.spawn_random()

    while not game.game_over and pieces_placed < MAX_PIECES:
        game.process_incoming_attacks(sim_time)

        had_piece = bool(game.current_piece)
        placed_now = play_one_piece(game, ai)

        if game.game_over:
            break

        if had_piece and placed_now and not game.current_piece:
            pieces_placed += 1
            sim_time += 1.0

            longest_b2b_chain_len = max(longest_b2b_chain_len, game.b2b_chain_len)

            if game.hold_label == "I":
                hold_i_steps += 1

            bb = BitBoard.from_board(board)
            if deepest_well_depth(bb) >= READY_WELL_MIN_DEPTH:
                ready_well_steps += 1

            if PRESSURE_EVERY_PIECES > 0 and (
                pieces_placed % PRESSURE_EVERY_PIECES == 0
            ):
                _enqueue_garbage_pressure(
                    game=game,
                    cols=board.cols,
                    lines=PRESSURE_LINES,
                    messiness=PRESSURE_MESSINESS,
                    sim_time=sim_time,
                )

        if not game.game_over and not game.current_piece:
            if game.skip_garbage_apply_once:
                game.skip_garbage_apply_once = False
            else:
                packets = game.pop_garbage_to_apply(game.MAX_GARBAGE_PER_PIECE)
                for lines, hole in packets:
                    board.add_garbage_rows(lines, hole_col=hole)

            game.spawn_random()

        if not placed_now and game.current_piece:
            break

    fitness = (
        LONGEST_B2B_BONUS * longest_b2b_chain_len
        + HOLD_I_KEEP_BONUS * hold_i_steps
        + READY_WELL_BONUS * ready_well_steps
    )

    return fitness, longest_b2b_chain_len, hold_i_steps, ready_well_steps


# evaluate genome across multiple seeds
def evaluate_genome(genome, seeds, ply=6):
    runs = []
    for seed in seeds:
        runs.append(simulate_genome(genome, seed, ply=ply))
    avg = sum(runs) / len(runs)
    return avg, runs


def random_gene_value():
    if random.random() < 0.20:
        return 0.0
    return math.exp(random.uniform(math.log(1e-3), math.log(1e3)))


def make_random_genome():
    genome = {}
    for key in GENE_KEYS:
        genome[key] = random_gene_value()
    return genome


def mutate_gene(value):
    if value <= 0.0:
        if random.random() < 0.80:
            return 0.0
        return random.uniform(0.0, 10.0)

    factor = math.exp(random.gauss(0.0, 0.35))
    return clamp_nonnegative(value * factor)


# mutate an entire genome
def mutate_genome(genome, per_gene_rate=0.22, reset_rate=0.03):
    child = dict(genome)
    for key in GENE_KEYS:
        if random.random() < reset_rate:
            child[key] = random_gene_value()
        elif random.random() < per_gene_rate:
            child[key] = mutate_gene(child[key])
    return child


# crossover two parents
def crossover(parent_a, parent_b):
    child = {}
    for key in GENE_KEYS:
        a = parent_a[key]
        b = parent_b[key]
        r = random.random()
        if r < 0.45:
            child[key] = a
        elif r < 0.90:
            child[key] = b
        else:
            alpha = random.random()
            child[key] = clamp_nonnegative(alpha * a + (1.0 - alpha) * b)
    return child


# key for caching genomes
def genome_cache_key(genome):
    return tuple((key, round(genome[key], 6)) for key in GENE_KEYS)


# tournament selection
def tournament_select(scored_population, tournament_size=3):
    pool = random.sample(scored_population, tournament_size)
    pool.sort(key=lambda item: item[0], reverse=True)
    return pool[0][1]


# initialise population with one base genome + randoms
def initialise_population(population_size):
    population = []
    while len(population) < population_size:
        population.append(make_random_genome())
    return population


def deepest_well_depth(bb):
    heights = column_heights(bb)

    best = 0
    cols = len(heights)

    for c in range(cols):
        left = heights[c - 1] if c > 0 else bb.rows
        right = heights[c + 1] if c < cols - 1 else bb.rows
        depth = min(left, right) - heights[c]
        if depth > best:
            best = depth

    return max(0, best)


# main genetic algorithm loop
def run_ga(
    generations=30,
    population_size=24,
    elite_count=4,
    tournament_size=3,
    mutation_rate=0.22,
    reset_rate=0.03,
    ply=6,
    seeds=None,
):
    if seeds is None:
        seeds = list(EVAL_SEEDS)

    population = initialise_population(population_size)
    fitness_cache = {}

    best_overall_score = -1.0
    best_overall_genome = None
    best_overall_runs = None

    for generation in range(1, generations + 1):
        scored = []

        for genome in population:
            key = genome_cache_key(genome)
            if key in fitness_cache:
                avg_score, runs = fitness_cache[key]
            else:
                avg_score, runs = evaluate_genome(genome, seeds=seeds, ply=ply)
                fitness_cache[key] = (avg_score, runs)
            scored.append((avg_score, genome, runs))

        scored.sort(key=lambda item: item[0], reverse=True)
        gen_best_score, gen_best_genome, gen_best_runs = scored[0]

        b2b_runs = []
        hold_runs = []
        well_runs = []

        for seed in seeds:
            _, b2b, hold_i, well = simulate_genome_with_stats(
                gen_best_genome, seed, ply=ply
            )
            b2b_runs.append(b2b)
            hold_runs.append(hold_i)
            well_runs.append(well)

        if gen_best_score > best_overall_score:
            best_overall_score = gen_best_score
            best_overall_genome = dict(gen_best_genome)
            best_overall_runs = list(gen_best_runs)

        print(
            f"generation {generation}: fitness={gen_best_score:.2f} "
            f"b2b_runs={b2b_runs} "
            f"hold_i_steps={hold_runs} "
            f"ready_well_steps={well_runs}"
        )

        next_population = [dict(item[1]) for item in scored[:elite_count]]

        while len(next_population) < population_size:
            parent_a = tournament_select(scored, tournament_size=tournament_size)
            parent_b = tournament_select(scored, tournament_size=tournament_size)
            child = crossover(parent_a, parent_b)
            child = mutate_genome(
                child,
                per_gene_rate=mutation_rate,
                reset_rate=reset_rate,
            )
            next_population.append(child)

        population = next_population

    b2b_runs = []
    hold_runs = []
    well_runs = []

    for seed in seeds:
        _, b2b, hold_i, well = simulate_genome_with_stats(
            best_overall_genome, seed, ply=ply
        )
        b2b_runs.append(b2b)
        hold_runs.append(hold_i)
        well_runs.append(well)

    print("\nbest overall average fitness:", f"{best_overall_score:.2f}")
    print("best overall fitness runs:", best_overall_runs)
    print("best overall b2b_runs:", b2b_runs)
    print("best overall hold_i_steps:", hold_runs)
    print("best overall ready_well_steps:", well_runs)
    print("best overall genome:")
    print("{")
    for key in GENE_KEYS:
        print(f'    "{key}": {best_overall_genome[key]},')
    print("}")

    return best_overall_genome, best_overall_score, best_overall_runs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--tournament", type=int, default=3)
    parser.add_argument("--mutation", type=float, default=0.22)
    parser.add_argument("--reset", type=float, default=0.03)
    parser.add_argument("--ply", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ga(
        generations=args.generations,
        population_size=args.population,
        elite_count=args.elite,
        tournament_size=args.tournament,
        mutation_rate=args.mutation,
        reset_rate=args.reset,
        ply=args.ply,
        seeds=EVAL_SEEDS,
    )
