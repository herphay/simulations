import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    """
    This file is for MIT OpenCourseWare - Introduction to Probability and Statistics
    18.05 | Spring 2022 | Undergraduate

    Course is conducted using R -> I will translate it into python
    """

#%%
# Week 1
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


#%%
if __name__ == '__main__':
    main()