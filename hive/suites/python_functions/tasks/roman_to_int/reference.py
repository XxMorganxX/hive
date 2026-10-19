VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def roman_to_int(s: str) -> int:
    total = 0
    for index, symbol in enumerate(s):
        value = VALUES[symbol]
        next_value = VALUES[s[index + 1]] if index + 1 < len(s) else 0
        total += -value if value < next_value else value
    return total
