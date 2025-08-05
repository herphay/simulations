import pandas as pd
import numpy as np
import math

from collections.abc import Iterable

def main():
    ...


def rps_winner(
        party1: str | Iterable[str] = 'r',
        party2: str | Iterable[str] = 'r',
    ) -> bool | Iterable[bool]:
    """
    Determine the winner of a rock paper scissor game
    """
    if isinstance(party1, str) and isinstance(party2, str):
        print('str')
    elif isinstance(party1, Iterable) and isinstance(party2, Iterable):
        print('iter')
    else:
        raise ValueError('The 2 input must match')


if __name__ == '__main__':
    main()