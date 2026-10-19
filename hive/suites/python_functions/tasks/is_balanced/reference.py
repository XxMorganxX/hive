PAIRS = {")": "(", "]": "[", "}": "{"}


def is_balanced(text: str) -> bool:
    open_brackets: list[str] = []
    for char in text:
        if char in "([{":
            open_brackets.append(char)
        elif char in PAIRS:
            if not open_brackets or open_brackets.pop() != PAIRS[char]:
                return False
    return not open_brackets
