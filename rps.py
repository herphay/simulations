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
        r > s  | s > p  | p > r
    
        r: 114 | p: 112 | s: 115

        rr: 0  | pp: 0  | ss: 0    # Tie
        rs: -1 | sp: 3  | pr: -2   # 1st position win
        sr: 1  | ps: -3 | rp: 2    # 2nd position win
    """
    if len(party1) != len(party2):
        raise ValueError('The length of both inputs must match')
    
    for arr in [party1, party2]:
        # Convert input into np array of single byte chars
        arr = np.array(arr, dtype='S1')
        # Convert the single byte chars into ASCII ints
        arr = arr.view(np.uint8)
        
        # Check whether input is a subset of proper inputs
        if not set([112, 114, 115]) >= set(arr):
            raise ValueError('Unacceptable input: only rps accepted')
    
    return party1, party2
    
        


if __name__ == '__main__':
    main()