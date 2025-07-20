import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Iterable
from collections import Counter

from time import perf_counter

def main() -> None:
    """
    This file is for MIT OpenCourseWare - Introduction to Probability and Statistics
    18.05 | Spring 2022 | Undergraduate

    Course is conducted using R -> I will translate it into python
    """

#%%
# Week 1 - general R samples
def sample(
        x: np.ndarray = np.arange(10),
        k: int = 5,
        replace: bool = False
    ) -> np.ndarray:
    """
    sample(x,k) generates a random permutation of k objects from the vector x. 
    That is, all k choices are different
    """
    # timeit('rng.integers(1, 7)', 'import numpy as np; rng = np.random.default_rng(); die = np.arange(1,7)', number=100000)
    # performance: 0.0852
    # timeit('rng.choice(die)', 'import numpy as np; rng = np.random.default_rng(); die = np.arange(1,7)', number=100000)
    # performance: 0.3648
    # 
    # Better to use random integers for interger dice rolling, unless absolute permutation is required
    rng = np.random.default_rng()
    return rng.choice(x, k, replace=replace)


def dice_roller(
        sides: int = 6,
        times: int = 4,
        repeat: int = 1000,
        check_for: int = 6,
        reset_rng_for_repeats: bool = False
    ) -> float:
    """
    Roll a dice n times, check if a specific number turn up
    """
    if not reset_rng_for_repeats:
        rng = np.random.default_rng()
        rolls = rng.integers(1, sides + 1, (times, repeat))

        prob_get = ((rolls == check_for).sum(axis=0) > 0).sum() / repeat
        prob_theo = 1 - (5 / 6) ** times
    
    print(f'Experiment result of getting at least 1 {check_for} out of {times} dice rolls is: ' + 
          f'{prob_get}, \nthe theoretical probability is {prob_theo:.3f}.')


def dice_sum_check(
        sides: int = 6,
        per_trial_rolls: int = 2,
        repeat: int = 1000,
        check_for_sum: int = 7,
    ) -> None:
    """
    Roll n dice, check if the sum is X
    """
    rng = np.random.default_rng()
    rolls = rng.integers(1, sides + 1, (per_trial_rolls, repeat))
    trial_sum = rolls.sum(axis=0)

    prob_exp = (trial_sum == check_for_sum).sum() / repeat
    prob_theo = 1/6 # to be implemented

    print(f'Experiment result of getting sum of {check_for_sum} with {per_trial_rolls} dice rolls' +
          f' {prob_exp:.3f}.\nThe theoretical result probability is: {prob_theo:.3f}.')


def get_prob_dice_sum(
        sides: int = 6,
        rolls: int = 2,
    ) -> np.ndarray:
    """
    Calculates the probability of getting each possible sum with rolls roll of sides sided dice
    """
    # References:
    # On mathematical formulation: https://mathworld.wolfram.com/Dice.html
    # supplemental math formulation: https://blogs.sas.com/content/iml/2024/08/26/formula-sum-of-dice.html
    # On dynamic programming formulation: 
    # https://www.geeksforgeeks.org/dsa/probability-of-getting-all-possible-values-on-throwing-n-dices/


# Week 1 - R Studio
def birthday_collider(
        ndays_in_year: int = 365,
        npeople: int = 50,
        ntrials: int = 1000
    ) -> float:
    ...


def have_dup_counter(
        arr: Iterable[int]
    ) -> bool:
    """
    Check if an array of ints have any duplicates.
    Uses built in Counter method
    """
    # Counter(arr) returns a dict with unique values of the arr as keys and count as values
    # .values() method on a dict returns a view of the dict's values (dict_value object)
    # Iterate through the counts the check if any > 1 (have duplicate)
    # Perf test 1 (repeat 100k, per trial 100): 0.0003537
    # Perf test 2 (dup_perf below): 0.4388, 0.4535, 0.4491
    return any(count > 1 for count in Counter(arr).values())


def have_dup_manual(
        arr: Iterable[int],
        range: int,
    ) -> bool:
    """
    Check if an array of ints have any duplicates.
    Uses manual count method
    """
    # Performance testing:
    # repeat('have_dup_manual(b, 365)', setup='from mit_stats_intro import have_dup_counter, 
    # have_dup_manual; from numpy.random import default_rng; 
    # b = default_rng().integers(365, size=50)', repeat=1_000, number=100)
    # Perf test 1 (repeat 100k, per trial 100): 0.0004728
    # Perf test 2 (dup_perf below): 0.5129, 9,5259, 0.5089
    counts = np.zeros(range)

    for item in arr:
        if counts[item] == 1:
            return True
        else:
            counts[item] += 1
    
    return False


def dup_perf(
        trials: int = 100_000,
    ) -> dict[str, float]:
    total_times = np.zeros(2)

    for i in range(trials):
        rng = np.random.default_rng()
        b = rng.integers(365, size=50)

        s = perf_counter()
        have_dup_counter(b)
        e = perf_counter()
        total_times[0] += e - s

        s = perf_counter()
        have_dup_manual(b,  365)
        e = perf_counter()
        total_times[1] += e - s
    
    return total_times


#%%
if __name__ == '__main__':
    main()