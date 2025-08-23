import pandas as pd
import numpy as np
import math

from collections.abc import Iterable

import rps_strats as strategy

def main():
    ...


def simulate_1_gen(
        ecosystem: list[list] = [[strategy.always_paper, 500], 
                                 [strategy.always_rock, 700], 
                                 [strategy.always_scissor, 800]],
        rng: np.random.Generator = None,
    ) -> list[list]:
    """
    Simulate 1 generation of competition among different RPS strategies

    ecosystem: collection of strategy classes and their starting population
        list of list, each with:
            key of the class, value of population size
    
    Returns the same list of list with updated population
    """
    # if method == 1:
    #     strats = np.array([i for i, species in enumerate(ecosystem) for _ in range(species[1])])
    # elif method == 2:
    #     strats = []
    #     for i, species in enumerate(ecosystem):
    #         strats += [i] * species[1]
    #     strats = np.array(strats)
    # else:
    #     strats = np.concat([np.full(species[1], i) for i, species in enumerate(ecosystem)])
    # PERFORMANCE RESULTS
    # timeit('simulate_1_gen(e, 1)', 
    # "from rps import simulate_1_gen;e = [['a', 500], ['b', 700], ['c', 800]]", 
    # number=50_000)
    # Method 1: 3.102, 3.182, 3.137
    # Method 2: 2.499, 2.527, 2.539
    # Method 3: 0.156, 0.150, 0.157

    # Create strat array -> [0, 0, 1, 2, 2, 2] means first 2 element are strat 0, next 1 strat 1 etc
    strats = np.concat([np.full(species[1], i) for i, species in enumerate(ecosystem)])
    # Get the plays for each individual, array is in order of species
    plays = np.concat([list(species[0].get_plays(species[1])) for species in ecosystem])
    
    if not rng:
        rng = np.random.default_rng()

    # Create random matching by shuffling individuals around, unshuffle 
    pop_count = len(plays)
    shuffle = rng.permutation(pop_count)
    unshuffle = np.zeros(pop_count)
    unshuffle[shuffle] = np.arange(pop_count)

    return strats, plays


def rps_winner(
        party1: Iterable[str] = 'r',
        party2: Iterable[str] = 'r',
    ) -> bool | Iterable[bool]:
    """
    Determine the winner of a rock paper scissor game:
        r > s  | s > p  | p > r
    
        r: 114 | p: 112 | s: 115

        rr: 0  | pp: 0  | ss: 0    # Tie
        rs: -1 | ps: -3 | pr: -2   # 1st position win
        sr: 1  | sp: 3  | rp: 2    # 2nd position win
    """
    # from rps import rps_winner;a=list('rpsrppssr');b=list('rpsssrrpp')
    if len(party1) != len(party2):
        raise ValueError('The length of both inputs must match')
    
    def integerize_rps(arr):
        # Convert input into np array of single byte chars
        arr = np.array(arr, dtype='S1')
        # Convert the single byte chars into ASCII ints
        arr = arr.view(np.int8)
        
        # Check whether input is a subset of proper inputs
        if not set([112, 114, 115]) >= set(arr):
            raise ValueError('Unacceptable input: only rps accepted')
        return arr
    
    party1 = integerize_rps(party1)
    party2 = integerize_rps(party2)

    return party1 - party2
    
        


if __name__ == '__main__':
    main()