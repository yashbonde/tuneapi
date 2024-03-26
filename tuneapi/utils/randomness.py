# Copyright © 2024- Frello Technology Private Limited

import random
import string


def get_random_string(length: int, numbers: bool = True, special: bool = False) -> str:
    choice_items = string.ascii_letters
    if numbers:
        choice_items += string.digits
    if special:
        choice_items += string.punctuation
    return "".join((random.choice(choice_items) for _ in range(length)))
