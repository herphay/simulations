import pandas as pd
import numpy as np
import math

from collections.abc import Iterable

def main():
    ...


def rps_winner(
        party1: Iterable[str] = 'r',
        party2: Iterable[str] = 'r',
    ) -> bool | Iterable[bool]:
    """
    Determine the winner of a rock paper scissor game:
        r > s | s > p | p > r
    """
    if len(party1) == len(party2):
        ...
    else:
        raise ValueError('The length of both inputs must match')


if __name__ == '__main__':
    main()