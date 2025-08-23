import pandas as pd
import numpy as np
import math

from collections.abc import Iterable

import rps_strats as strategy

def main():
    ...


def simulate_1_gen(
        ecosystem: list[list],
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
    unshuffle = np.zeros(pop_count, dtype=int)
    unshuffle[shuffle] = np.arange(pop_count)

    # Shuffle the plays
    plays = plays[shuffle]

    # Split into 2 groups for playoffs
    h1_start = pop_count % 2
    h1_stop = pop_count // 2 + h1_start

    results = np.zeros(pop_count, dtype=np.int8)
    # Generate results of playoff -> +ve means member won, -ve means lost
    half_result = rps_winner(plays[h1_start:h1_stop], plays[h1_stop:])
    results[h1_start:h1_stop] = half_result
    results[h1_stop:] = -half_result
    
    # Get the outcome array, which represents how much each individual propagate to the next gen
    # 0 mean the individual have 0 offspring, 1 means 1 offspring etc.
    results = play_outcome_to_pop(results, method='const')
    
    # Unshuffle the results to match the initial individual sequence
    results = results[unshuffle]

    pop_cumsum = 0
    for species in ecosystem:
        next_pop_cumsum = pop_cumsum + species[1]
        species[1] = results[pop_cumsum:next_pop_cumsum].sum()
        pop_cumsum = next_pop_cumsum

    return ecosystem


def rps_winner(
        party1: Iterable[str] = 'r',
        party2: Iterable[str] = 'r',
    ) -> bool | Iterable[bool]:
    """
    Determine the winner of a rock paper scissor game:
        r > s  | s > p  | p > r
    
        r: 114 | p: 112 | s: 115

        rr: 0  | pp: 0  | ss: 0    # Tie
        rs: -1 | sp: 3 | pr: -2    # 1st position win
        sr: 1  | ps: -3  | rp: 2   # 2nd position win
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
    result = party1 - party2
    result[((-3 < result) & (result < 0)) | (result == 3)] = 99  # 1st position win
    result[((3 > result) & (result > 0)) | (result == -3)] = -99 # 2nd position win

    return result
    
        
def play_outcome_to_pop(
        results: np.ndarray,
        method: str = 'const'
    ) -> np.ndarray:
    """
    Based on each individual's play outcome, determine how population change
    """
    match method:
        case 'const':
            results[results > 0] = 2
            results[results == 0] = 1
            results[results < 0] = 0
            return results
        case _:
            raise ValueError('Invalid method')


if __name__ == '__main__':
    main()