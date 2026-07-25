"""Fixed-size UTF-8 encoding for text observations."""

import numpy as np


MAX_GOAL_DESCRIPTION_BYTES = 256


def encode_goal_description(description):
    encoded = str(description).encode("utf-8")
    if len(encoded) > MAX_GOAL_DESCRIPTION_BYTES:
        raise ValueError(f"Goal description exceeds {MAX_GOAL_DESCRIPTION_BYTES} bytes")
    return np.frombuffer(
        encoded.ljust(MAX_GOAL_DESCRIPTION_BYTES, b"\0"), dtype=np.uint8
    )


def decode_goal_description(encoded):
    return np.asarray(encoded, dtype=np.uint8).tobytes().split(b"\0", 1)[0].decode()
