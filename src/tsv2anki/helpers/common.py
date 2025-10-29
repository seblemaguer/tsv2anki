CATEGORIES = [
    "Word",
    "Grammar",
    "Cultural",
    "Phrase"
]

def validate_criterion(filter_criterion: str, cur_value: int) -> bool:

    validate = False

    direction = filter_criterion[0]
    value_ref = int(filter_criterion[1:])

    if (direction == "<"):
        return cur_value < value_ref
    elif direction == ">":
        return cur_value > value_ref
    elif direction == "=":
        return cur_value == value_ref
    else:
        raise Exception(f"The comparison direction {direction} is unknown.")
