import heapq
from src.game.bitboard import BitBoard
from src.search.search import (
    count_holes,
    column_heights,
    enumerate_hard_drop_landings,
    non_deepest_well_penalty,
    row_transitions,
    bumpiness,
    column_holes_via_transitions,
    placement_exceeds_top,
)
from src.game.tetromino import PIECE_ROW_MASKS


class TetrisAI:
    def __init__(
        self,
        ply=2,
        well_col=9,
        gamma=0.95,
        depth_beam_width=14,
    ):
        self.ply = max(1, int(ply))
        self.gamma = float(gamma)
        self.well_col = int(well_col)

        # plan cache / pending hold-followup
        self._pending_after_hold = None
        self._pending_after_hold_label = None

        # candidate cache
        self._cand_cache = {}
        self._cand_cache_max = 200_000

        # transposition table
        self._tt = {}
        self._tt_max = 200_000

        # feature cache
        self._feat_cache = {}
        self._feat_cache_max = 250_000

        # search / pruning knobs
        self.beam_width = 20
        self.use_beam = False
        self.use_upper_bound_prune = False
        self.STEP_UPPER = 80

        # danger
        self.in_danger = False
        self.has_incoming_garbage = False
        self.DANGER_THRESHOLD = 14

        # beam search width for AI2 planner
        self.depth_beam_width = int(depth_beam_width)

        self.TOP_OUT_PENALTY = 1_000_000

        # self.W_SAFE = {
        #     "HOLES": 372,
        #     "FILL_WELL": 145,
        #     # "FILL_WELL": 0,
        #     "WELL_HEIGHT": 46,
        #     # "WELL_HEIGHT": 0,
        #     "MAX_HEIGHT": 32,
        #     "TOTAL_HEIGHT": 0,
        #     "ROW_TRANSITIONS": 4,
        #     "BUMPINESS": 1,
        #     "NON_DEEPEST_WELL": 200,
        #     "HOLD_ACTION": 1,
        #     "TETRIS": 5,
        #     "ANY_CLEAR_PENALTY": 60,
        #     "HOLD_I": 120,
        # }

        self.W_SAFE = {
            "HOLES": 43.20399769649276,
            "FILL_WELL": 4.417937571688473,
            "WELL_HEIGHT": 0.49931370180438184,
            # "WELL_HEIGHT": 20,
            "MAX_HEIGHT": 0.6071598600711604,
            "TOTAL_HEIGHT": 0.001370819100639725,
            "ROW_TRANSITIONS": 0.09960404784534359,
            "BUMPINESS": 0.02712821913179512,
            "NON_DEEPEST_WELL": 0.5506895630342138,
            "HOLD_ACTION": 0.007468269214495767,
            "TETRIS": 0.0,
            "ANY_CLEAR_PENALTY": 1.80086484733867,
            # "HOLD_I": 0.14695665157609938,
            "HOLD_I": 3,
        }

        self.W_OPTIMISTIC = {
            "HOLES": 43.20399769649276,
            # "FILL_WELL": 4.417937571688473,
            "FILL_WELL": 0,
            # "WELL_HEIGHT": 0.49931370180438184,
            "WELL_HEIGHT": 0,
            # "WELL_HEIGHT": 20,
            "MAX_HEIGHT": 0.6071598600711604,
            "TOTAL_HEIGHT": 0.001370819100639725,
            "ROW_TRANSITIONS": 0.09960404784534359,
            "BUMPINESS": 0.02712821913179512,
            "NON_DEEPEST_WELL": 0.5506895630342138,
            "HOLD_ACTION": 0.007468269214495767,
            "TETRIS": 0.0,
            "ANY_CLEAR_PENALTY": 6.80086484733867,
            # "HOLD_I": 0.14695665157609938,
            "HOLD_I": 3,
        }

        # self.W_OPTIMISTIC = {
        #     "HOLES": 372,
        #     # "FILL_WELL": 145,
        #     "FILL_WELL": 0,
        #     "WELL_HEIGHT": 46,
        #     # "WELL_HEIGHT": 0,
        #     "MAX_HEIGHT": 32,
        #     "TOTAL_HEIGHT": 0,
        #     "ROW_TRANSITIONS": 4,
        #     "BUMPINESS": 1,
        #     "NON_DEEPEST_WELL": 200,
        #     "HOLD_ACTION": 1,
        #     "TETRIS": 5,
        #     "ANY_CLEAR_PENALTY": 60,
        #     "HOLD_I": 120,
        # }

        self.W_DANGER = {
            "HOLES": 527.1930015347469,
            "FILL_WELL": 0,
            "WELL_HEIGHT": 0.0,
            "MAX_HEIGHT": 43.21985401138501,
            "TOTAL_HEIGHT": 0.0010804263149219301,
            "ROW_TRANSITIONS": 29.808917976597172,
            "BUMPINESS": 18.493971703712425,
            "NON_DEEPEST_WELL": 3.5775764823940723,
            "HOLD_ACTION": 3.3295509839237623,
            "TETRIS": 0.0009644943756212026,
            # "ANY_CLEAR_PENALTY": -0.0012338866701557963,
            "ANY_CLEAR_PENALTY": -20,
            "HOLD_I": 5.574812799951162,
        }

        # {
        #     "HOLES": 158.1865,
        #     "FILL_WELL": 14.3271,
        #     # "FILL_WELL": 0,

        #     "WELL_HEIGHT": 0.0,
        #     "MAX_HEIGHT": 20.8346,
        #     "TOTAL_HEIGHT": 4.1587,
        #     "ROW_TRANSITIONS": 8.0,
        #     "BUMPINESS": 6.0,
        #     "NON_DEEPEST_WELL": 69.6025,
        #     "HOLD_ACTION": 0.0,
        #     "TETRIS": 0.0,
        #     "ANY_CLEAR_PENALTY": 65.9463,
        #     "HOLD_I": 0.0,
        # }

        # default startup table
        self.W = self.W_SAFE

        # diagnostics
        self.tt_hits = 0
        self.tt_misses = 0
        self.cand_hits = 0
        self.cand_misses = 0
        self.feat_hits = 0
        self.feat_misses = 0
        self.ub_checks = 0
        self.ub_prunes = 0

    def _update_incoming_signal(self, game):
        self.has_incoming_garbage = game.incoming_garbage_lines() > 0

    def _current_hold_i_bonus(self):
        if self.has_incoming_garbage:
            return 0
        return self.W["HOLD_I"]

    def _hold_i_bonus(self, hold_label):
        return self._current_hold_i_bonus() if hold_label == "I" else 0.0

    def choose_action(self, game, opponent_game=None):
        if getattr(game, "game_over", False):
            return None
        if not getattr(game, "current_piece", None):
            return None

        self._update_incoming_signal(game)

        cur_label, _cur_rot, _cur_x, _cur_y, _shape = game.current_piece

        # execute stored landing right after a hold
        if game._hold_used and self._pending_after_hold is not None:
            if cur_label == self._pending_after_hold_label:
                x, y, rot = self._pending_after_hold
                self._pending_after_hold = None
                self._pending_after_hold_label = None
                if game.force_to_landing(x, y, rot):
                    return "hard_drop"

        next_queue = self._extract_next_labels(game)
        # print("PLY:", self.ply, " | next_queue len:", len(next_queue), next_queue)
        bb0 = BitBoard.from_board(game.board)

        # danger assessment
        _heights, max_h, _total_h, holes, _bump, _row_t, _well_h, _nd_well = (
            self._features(bb0)
        )

        my_grouped_holes = sum(column_holes_via_transitions(bb0))

        if max_h > 16 or (max_h > 12 and my_grouped_holes >= 4):
            self.in_danger = True
            self.W = self.W_DANGER
        elif 8 <= max_h <= 16:
            self.in_danger = False
            self.W = self.W_OPTIMISTIC
        else:
            self.in_danger = False
            self.W = self.W_SAFE

        plan = self._plan(
            bb0=bb0,
            cur_label=cur_label,
            hold_label=game.hold_label,
            hold_used=game._hold_used,
            next_queue=next_queue,
        )

        if plan is None:
            return None

        used_hold, landing, swapped_in = plan

        if used_hold:
            self._pending_after_hold = landing
            self._pending_after_hold_label = swapped_in
            return "hold"

        x, y, rot = landing
        if game.force_to_landing(x, y, rot):
            return "hard_drop"

        return None

    def _plan(
        self,
        *,
        bb0,
        cur_label,
        hold_label,
        hold_used,
        next_queue,
    ):
        if self.depth_beam_width <= 0:
            return None

        def push_node(
            acc_score,
            bb,
            h_label,
            q,
            current_label,
            first_used_hold,
            first_landing,
            first_swapped_in,
        ):
            return (
                acc_score,
                bb,
                h_label,
                q,
                current_label,
                first_used_hold,
                first_landing,
                first_swapped_in,
            )

        frontier = []

        # depth 0: no hold
        for imm0, landing, bb1, _cleared in self._build_candidates(bb0, cur_label):
            imm0_adj = imm0 + self._hold_i_bonus(hold_label)

            if self.ply <= 1:
                frontier.append(
                    push_node(imm0_adj, bb1, hold_label, [], None, False, landing, None)
                )
                continue

            if not next_queue:
                continue

            frontier.append(
                push_node(
                    imm0_adj,
                    bb1,
                    hold_label,
                    next_queue[1:],
                    next_queue[0],
                    False,
                    landing,
                    None,
                )
            )

        # depth 0: hold
        if not hold_used:
            swapped_in = self._label_after_hold(hold_label, next_queue)
            if swapped_in is not None:
                remaining = (
                    next_queue[1:]
                    if (hold_label is None and next_queue)
                    else next_queue
                )

                for imm0, landing, bb1, _cleared in self._build_candidates(
                    bb0, swapped_in
                ):
                    imm0_adj = (
                        imm0 - self.W["HOLD_ACTION"] + self._hold_i_bonus(cur_label)
                    )

                    if self.ply <= 1:
                        frontier.append(
                            push_node(
                                imm0_adj,
                                bb1,
                                cur_label,
                                [],
                                None,
                                True,
                                landing,
                                swapped_in,
                            )
                        )
                        continue

                    if not remaining:
                        continue

                    frontier.append(
                        push_node(
                            imm0_adj,
                            bb1,
                            cur_label,
                            remaining[1:],
                            remaining[0],
                            True,
                            landing,
                            swapped_in,
                        )
                    )

        if not frontier:
            return None

        frontier = heapq.nlargest(self.depth_beam_width, frontier, key=lambda n: n[0])

        # deeper expansions
        depth_done = 1
        discount = self.gamma

        while depth_done < self.ply:
            new_frontier = []

            for (
                acc,
                bb,
                h_label,
                q,
                current_label,
                first_used_hold,
                first_landing,
                first_swapped_in,
            ) in frontier:
                if current_label is None:
                    new_frontier.append(
                        (
                            acc,
                            bb,
                            h_label,
                            q,
                            None,
                            first_used_hold,
                            first_landing,
                            first_swapped_in,
                        )
                    )
                    continue

                # no hold
                for imm, _landing2, bb2, _cleared in self._build_candidates(
                    bb, current_label
                ):
                    imm_adj = imm + self._hold_i_bonus(h_label)

                    if depth_done + 1 >= self.ply:
                        nxt, rest = None, []
                    else:
                        if not q:
                            continue
                        nxt, rest = q[0], q[1:]

                    new_frontier.append(
                        push_node(
                            acc + discount * imm_adj,
                            bb2,
                            h_label,
                            rest,
                            nxt,
                            first_used_hold,
                            first_landing,
                            first_swapped_in,
                        )
                    )

                # hold
                swapped_in = self._label_after_hold(h_label, q)
                if swapped_in is not None:
                    remaining = q[1:] if (h_label is None and q) else q

                    for imm, _landing2, bb2, _cleared in self._build_candidates(
                        bb, swapped_in
                    ):
                        imm_adj = (
                            imm
                            - self.W["HOLD_ACTION"]
                            + self._hold_i_bonus(current_label)
                        )

                        if depth_done + 1 >= self.ply:
                            nxt, rest = None, []
                        else:
                            if not remaining:
                                continue
                            nxt, rest = remaining[0], remaining[1:]

                        new_frontier.append(
                            push_node(
                                acc + discount * imm_adj,
                                bb2,
                                current_label,
                                rest,
                                nxt,
                                first_used_hold,
                                first_landing,
                                first_swapped_in,
                            )
                        )

            if not new_frontier:
                break

            frontier = heapq.nlargest(
                self.depth_beam_width, new_frontier, key=lambda n: n[0]
            )
            depth_done += 1
            discount *= self.gamma

        best = max(frontier, key=lambda n: n[0])
        _acc, _bb, _h, _q, _cur, first_used_hold, first_landing, first_swapped_in = best
        return (first_used_hold, first_landing, first_swapped_in)

    def _tt_key(self, bb, current_label, hold_label, hold_used, next_queue, depth):
        return (
            bb.rows_bits,
            current_label,
            hold_label,
            hold_used,
            tuple(next_queue[:depth]),
            depth,
            self.in_danger,
            self.has_incoming_garbage,
        )

    def _search(self, *, bb, current_label, hold_label, hold_used, next_queue, depth):
        if depth <= 0:
            return 0.0

        key = self._tt_key(bb, current_label, hold_label, hold_used, next_queue, depth)
        cached = self._tt.get(key)
        if cached is not None:
            return cached

        best = -1e18
        future_ub = (
            self._future_upper_bound(depth - 1) if self.use_upper_bound_prune else None
        )

        # place current
        for imm0, landing, bb2, cleared in self._build_candidates(bb, current_label):
            immediate = imm0

            if self.use_upper_bound_prune and depth > 1:
                if immediate + self.gamma * future_ub < best:
                    continue

            future = 0.0
            if depth > 1 and next_queue:
                future = self._search(
                    bb=bb2,
                    current_label=next_queue[0],
                    hold_label=hold_label,
                    hold_used=False,
                    next_queue=next_queue[1:],
                    depth=depth - 1,
                )

            best = max(best, immediate + self.gamma * future)

        # consider hold
        if not hold_used:
            swapped_in = self._label_after_hold(hold_label, next_queue)
            if swapped_in is not None:
                for imm0, landing, bb2, cleared in self._build_candidates(
                    bb, swapped_in
                ):
                    immediate = imm0 - self.W["HOLD_ACTION"]

                    if self.use_upper_bound_prune and depth > 1:
                        if immediate + self.gamma * future_ub < best:
                            continue

                    remaining = (
                        next_queue[1:]
                        if (hold_label is None and next_queue)
                        else next_queue
                    )

                    future = 0.0
                    if depth > 1 and remaining:
                        future = self._search(
                            bb=bb2,
                            current_label=remaining[0],
                            hold_label=current_label,
                            hold_used=False,
                            next_queue=remaining[1:],
                            depth=depth - 1,
                        )

                    best = max(best, immediate + self.gamma * future)

        result = best if best > -1e17 else 0.0

        if len(self._tt) >= self._tt_max:
            self._tt.clear()
        self._tt[key] = result
        return result

    @staticmethod
    def _full_row_mask(cols):
        return (1 << cols) - 1

    def _apply_and_clear_bit(self, bb, label, landing):
        x, y, rot = landing
        row_masks, h, _w = PIECE_ROW_MASKS[label][int(rot) % 4]
        board_mask = self._full_row_mask(bb.cols)

        new_rows = list(bb.rows_bits)

        for r in range(h):
            row_mask = row_masks[r]
            if row_mask == 0:
                continue

            board_row = y + r
            shifted = (row_mask << x) if x >= 0 else (row_mask >> (-x))
            shifted &= board_mask

            if board_row < 0:
                continue
            if 0 <= board_row < bb.rows:
                new_rows[board_row] |= shifted

        placed = BitBoard(bb.rows, bb.cols, tuple(new_rows))
        cleared_board, cleared = placed.clear_full_rows()
        return cleared_board, cleared

    def _build_candidates(self, bb, label):
        cache_key = (
            bb.rows_bits,
            label,
            self.in_danger,
            self.has_incoming_garbage,
        )
        cached = self._cand_cache.get(cache_key)
        if cached is not None:
            return cached

        best_for_state = {}

        for rot, x, y in enumerate_hard_drop_landings(bb, label):
            landing = (x, y, rot)
            bb2, cleared = self._apply_and_clear_bit(bb, label, landing)
            state_key = bb2.rows_bits

            # immediate = self._evaluate_transition(prev=bb, new=bb2, cleared=cleared)

            immediate = self._evaluate_transition(
                prev=bb,
                new=bb2,
                cleared=cleared,
                label=label,
                landing=landing,
            )

            old = best_for_state.get(state_key)
            if old is None or immediate > old[0]:
                best_for_state[state_key] = (immediate, landing, bb2, cleared)

        cands = list(best_for_state.values())
        if not cands:
            self._cand_cache[cache_key] = []
            return []

        if self.use_beam and self.beam_width and self.beam_width > 0:
            cands = heapq.nlargest(self.beam_width, cands, key=lambda t: t[0])
        else:
            cands.sort(key=lambda t: t[0], reverse=True)

        if len(self._cand_cache) >= self._cand_cache_max:
            self._cand_cache.clear()
        self._cand_cache[cache_key] = cands
        return cands

    def _features(self, bb):
        key = bb.rows_bits
        cached = self._feat_cache.get(key)
        if cached is not None:
            return cached

        heights = column_heights(bb)
        heights_t = tuple(heights)

        max_h = max(heights) if heights else 0
        total_h = sum(heights)
        holes = int(count_holes(bb))
        bump = int(bumpiness(bb, heights))
        row_t = int(row_transitions(bb))
        well_h = int(heights[self.well_col]) if 0 <= self.well_col < bb.cols else 0
        nd_well = int(
            non_deepest_well_penalty(bb, heights=list(heights_t), min_depth=2)
        )

        out = (heights_t, max_h, total_h, holes, bump, row_t, well_h, nd_well)

        if len(self._feat_cache) >= self._feat_cache_max:
            self._feat_cache.clear()
        self._feat_cache[key] = out
        return out

    def _evaluate_grid(self, bb):
        heights, max_h, total_h, holes, bump, row_t, well_h, nd_well = self._features(
            bb
        )
        W = self.W

        well_pen = 0.0
        if well_h > 0:
            well_pen += W["FILL_WELL"]
            well_pen += W["WELL_HEIGHT"] * well_h

        score = 0.0
        score -= W["HOLES"] * holes
        score -= well_pen
        score -= W["MAX_HEIGHT"] * max_h
        score -= W["TOTAL_HEIGHT"] * total_h
        score -= W["ROW_TRANSITIONS"] * row_t
        score -= W["BUMPINESS"] * bump
        score -= W["NON_DEEPEST_WELL"] * nd_well

        # print("nd_well= ", nd_well)
        return score

    # def _evaluate_transition(self, *, prev, new, cleared):
    #     base = self._evaluate_grid(new)
    #     W = self.W

    #     if cleared == 4:
    #         return base + W["TETRIS"]

    #     if 0 < cleared < 4:
    #         base -= W["ANY_CLEAR_PENALTY"]

    #     return base

    def _evaluate_transition(self, *, prev, new, cleared, label, landing):
        base = self._evaluate_grid(new)
        W = self.W

        if placement_exceeds_top(label, landing):
            base -= self.TOP_OUT_PENALTY

        if cleared == 4:
            return base + W["TETRIS"]

        if 0 < cleared < 4:
            base -= W["ANY_CLEAR_PENALTY"]

        return base

    def _extract_next_labels(self, game):
        if hasattr(game, "peek_next"):
            raw = game.peek_next(max(1, self.ply + 2))
            out = []
            for it in raw:
                if isinstance(it, str):
                    out.append(it)
                elif isinstance(it, (list, tuple)) and it and isinstance(it[0], str):
                    out.append(it[0])
            return out
        return []

    @staticmethod
    def _label_after_hold(hold_label, next_queue):
        return (
            hold_label
            if hold_label is not None
            else (next_queue[0] if next_queue else None)
        )

    def _future_upper_bound(self, depth):
        ub = 0.0
        g = 1.0
        for _ in range(depth):
            ub += g * self.STEP_UPPER
            g *= self.gamma
        return ub
