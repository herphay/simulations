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



#%%
if __name__ == '__main__':
    main()