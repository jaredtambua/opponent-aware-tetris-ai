_BASE_TABLE = {
    "single":            [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    "double":            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6],
    "triple":            [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9,10,10,11,11,12,12],
    "quad":              [4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],

    "tspin_mini_single": [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
    "tspin_single":      [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9,10,10,11,11,12],
    "tspin_mini_double": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6],
    "tspin_double":      [4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
    "tspin_triple":      [6, 7, 9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36],
}


def b2b_level(chain_len):
    if chain_len <= 0:
        return 0
    if 1 <= chain_len <= 2:
        return 1
    if 3 <= chain_len <= 7:
        return 2
    if 8 <= chain_len <= 23:
        return 3
    return 4


def clamp_combo_index(combo_index):
    if combo_index < 0:
        return 0
    if combo_index > 20:
        return 20
    return combo_index


def is_b2b_eligible(atk_type):
    return (atk_type == "quad") or ("spin" in atk_type)


def compute_attack(*, atk_type, combo_count, b2b_chain_len, all_clear):
    combo_index = clamp_combo_index(combo_count)
    base = _BASE_TABLE[atk_type][combo_index]

    if is_b2b_eligible(atk_type):
        base += b2b_level(b2b_chain_len)

    if all_clear:
        base += 5

    return max(0, int(base))