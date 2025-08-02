import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes

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
        ndice: int = 2,
        ver: str = 'theoretical'
    ) -> np.ndarray:
    """
    Calculates the probability of getting each possible sum with 'ndice' roll of 'sides' sided dice
    """
    # References:
    # On mathematical formulation: https://mathworld.wolfram.com/Dice.html
    # supplemental math formulation: https://blogs.sas.com/content/iml/2024/08/26/formula-sum-of-dice.html
    # On dynamic programming formulation: 
    # https://www.geeksforgeeks.org/dsa/probability-of-getting-all-possible-values-on-throwing-n-dices/

    # Implementation for theoretical probability calc
    if ver == 'theoretical':
        possible_sums = list(range(ndice, ndice * sides + 1))
        ways_to_achieve_sum = np.zeros(len(possible_sums), dtype=int)

        for i, sum in enumerate(possible_sums):
            ways = 0
            for k in range((sum - ndice) // sides + 1):
                ways += (-1) ** k * math.comb(ndice, k) * math.comb(sum - sides * k - 1, ndice - 1)
            
            ways_to_achieve_sum[i] = ways
        
        prob_to_achieve_sum = ways_to_achieve_sum / sides ** ndice

        return dict(zip(possible_sums, prob_to_achieve_sum)), \
               dict(zip(possible_sums, ways_to_achieve_sum))


# Week 1 - R Studio
def birthday_collider(
        ndays_in_year: int = 365,
        npeople: int = 50,
        ntrials: int = 100_000,
        print_results: bool = True,
        check_n: int = 2,
    ) -> float:
    """
    1. Randomly gen bdays based on ndays in year for npeople
    2. Check if at least n people share the same bday
    3. Repeat experiment for ntrials times
    4. Report the average
    """
    shared_bdays = 0

    for _ in range(ntrials):
        rng = np.random.default_rng() # Reset RNG for each trial rather than use the same for all
        # Generate random integers, each representing 1 day in the year for npeople all at once
        bdays = rng.integers(ndays_in_year, size=npeople)

        if have_n_counter(bdays, check_n):
            shared_bdays += 1
    
    exp_prob = shared_bdays / ntrials

    if check_n == 2:
        theo_prob = 1 - math.perm(ndays_in_year, npeople) / ndays_in_year ** npeople
    elif check_n == 3:
        theo_prob = 1 - (math.perm(ndays_in_year, npeople) + 
                         bday_share_sum(ndays_in_year, npeople, nshare=2)) / ndays_in_year ** npeople
    else:
        theo_prob = "--Not supported--"

    if print_results:
        theo_print = round(theo_prob, 4) if not isinstance(theo_prob, str) else theo_prob
        print(f'Experimental probability of shared birthday with {ndays_in_year} days in a year ' +
            f'and group size of {npeople} is: {exp_prob:.4f}. ' + 
            f'Theoretical probability is: {theo_print}.')
    
    # Problem 2b: what is the min number of people for >50% prob of colliding bday in 365 day year
    # ANS: 23 -> 0.5073 (22 at 0.4757)

    return exp_prob, theo_prob


def bday_prob_variance(
        ndays_in_year: int = 365,
        npeople: int = 15,
        test_trials: Iterable[int] = [50, 100, 500, 1000, 2000],
        num_trials: int = 100
    ) -> dict[int, float]:
    """
    Returns standard deviation of the probability based on a number of trials
    """
    exp_sd = {}
    for ntrials in test_trials:
        exp_results = np.zeros(num_trials)
        for i in range(num_trials):
            exp_results[i], _ = birthday_collider(ndays_in_year, npeople, 
                                                  ntrials=ntrials, print_results=False)

        exp_sd[ntrials] = np.std(exp_results)
    
    return exp_sd


def bday_sum_checker(
        ndays_in_year: int = 365,
        npeople: int = 50,
    ) -> int:
    """
    WRONG MATH -> missed out mixed repeats like 2 ppl share + 3 ppl share and all the permu thereof
    Checks if my math is correct.

    This does not give all the ways to sequence 50 bdays, because it only sum up no repeat, 1 repeat
    2 repeat etc. but not 2 bdays each repeat once etc. So it under-counts
    """
    ways = math.perm(ndays_in_year, npeople)
    for k in range(2, npeople + 1):
        ways += math.comb(npeople, k) * ndays_in_year * math.perm(ndays_in_year - 1, npeople - k)
    
    print(ways)
    total_ways = ndays_in_year ** npeople
    print(total_ways)
    return ways - total_ways


def bday_share_sum(
        ndays_in_year: int = 365,
        npeople: int = 50,
        nshare: int = 2,
    ) -> int:
    """
    CORRECT FOR nshare = 2
    Gives the number of ways where only exactly nshare people are sharing bdays (there can be
    multiple groups of nshare people each sharing different bdays)
    """
    # ways = math.perm(ndays_in_year, npeople) # Number ways to sequence n ppl with unique bdays
    ways = 0
    for k in range(1, npeople // nshare + 1):
        ways += math.comb(ndays_in_year, k) * math.comb(ndays_in_year - k, npeople - nshare * k) * \
                math.factorial(npeople) / (math.factorial(nshare) ** k)
    
    return ways


def bday_total_sum(
        ndays_in_year: int = 365,
        npeople: int = 50,
    ) -> int:
    """
    WRONG MATH
    Also wrong for total bday sum (365^50), because there can be mixed bday sharing. I.e. pairs 
    sharing + triplets sharing etc.
    """
    ways = math.perm(ndays_in_year, npeople)
    for n in range(2, npeople + 1):
        ways += bday_share_sum(ndays_in_year, npeople, n)
    
    return ways


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


def have_n_counter(
        arr: Iterable[int],
        n
    ) -> bool:
    """
    Check if there is at least 1 instance of n or more repeats
    IN-USE
    """
    return any(count >= n for count in Counter(arr).values())


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

    for _ in range(trials):
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
# Week 2: Class 4 R reading questions
def w2_R_rq_dice_avg(
        ntrials: int = 100_000,
        nsides: int = 6
    ) -> float:
    rng = np.random.default_rng()
    return np.average(rng.integers(1, nsides + 1, size=ntrials))


def w2_R_rq_longest_run(
        seq_len: int = 20,
        upper: int = 2
    ) -> int:
    rng = np.random.default_rng()
    seq = rng.integers(upper, size=seq_len)
    print(seq)

    max_len = 0
    current_len = 0
    for i in range(1, len(seq)):
        if seq[i - 1] == seq[i]:
            current_len += 1
        else:
            max_len = max(max_len, current_len)
            current_len = 0
    
    return max_len + 1


def plt_binom(
        n: int = 10,
        p: float = 0.5,
        method: str = 'notvect',
        output: bool = False,
        plot: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
    # Vectorize math.comb function for numpy to compute the combis efficiently
    # TO TEST OUT EFFICIENCY vs LOOP
    # 
    # Performance testing
    # timeit("plt_binom(method='vect', plot=False)", 'from mit_stats_intro import plt_binom', number=100_000)
    # 1.1244, 1.1090, 1.1289
    # VS
    # timeit("plt_binom(method='va', plot=False)", 'from mit_stats_intro import plt_binom', number=100_000)
    # 0.2861, 0.2694, 0.2693
    # seems like vectorization for just 10 computations don't work
    # At n = 60 vectorized:
    # 1.5494, 1.5286, 1.5399
    # loop:
    # 1.0572, 1.0603, 1.0498
    # 
    # VERDICT: vectorization will catch up eventually BUT np.vect can't handle large python ints
    # -> throws error "Python int too large to convert to C long" at n >= 67, so loop always wins

    if method == 'vect':
        vect_comb = np.vectorize(math.comb)
        k = np.arange(n + 1)
        pmf = vect_comb(n, k) * p ** k * (1 - p) ** (n - k)
    
    else:
        pmf = np.zeros(n + 1)
        for k in range(n + 1):
            pmf[k] = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
    
    cdf = np.cumsum(pmf)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1: matplotlib.axes.Axes
        ax2: matplotlib.axes.Axes

        x = np.arange(n + 1)
        ax1.plot(x, pmf, 'o')
        ax2.plot(x, cdf, 'o')
    
    if output:
        return pmf, cdf


def plt_geom(
        p: float = 0.5,
        n_limit: int = 10,
        output: bool = True,
        plot: bool = True,
    ) -> np.ndarray:

    k = np.arange(n_limit)

    pmf = (1 - p) ** k * p # No difference in performance vs calc & assign q first
    cdf = np.cumsum(pmf)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1: matplotlib.axes.Axes
        ax2: matplotlib.axes.Axes
        ax1.plot(k, pmf, 'o')
        ax2.plot(k, cdf, 'o')

    if output:
        return pmf, cdf


def plt_uni(
        n: int = 1000,
        output: bool = True
    ) -> np.ndarray:
    rng = np.random.default_rng()
    samples = rng.random(n)

    fig, ax = plt.subplots()
    ax.plot(samples, np.ones(n), '.')


def dice_func_EV(
        nsides: int = 6,
        ndice: int = 2,
    ) -> float:
    prob, _ = get_prob_dice_sum(nsides, ndice)
    prob: dict

    sums = np.fromiter(prob.keys(), dtype=int)
    probs = np.fromiter(prob.values(), dtype=float)

    return sum((sums ** 2 - 6 * sums + 1) * probs)

def sim_binom(
        n: int = 10,
        p: float = 0.5,
        check_k: int = 5,
        trials: int = 250_000
    ) -> tuple[float, float]:
    """
    Y~binom(n, p)
    Simulates P(Y = k) & P(Y <= k)
    """
    theo_pob, theo_cdf = plt_binom(n, p, output=True, plot=True)

    rng = np.random.default_rng()
    
    p_k = 0
    p_se_k = 0

    # sim_result = rng.choice(2, size=n, p=[1 - p, p])
    for _ in range(trials):
        if (k := (rng.random(n) < p).sum()) < check_k:
            p_se_k += 1
        elif k == check_k:
            p_se_k += 1
            p_k += 1
    
    p_k /= trials
    p_se_k /= trials

    print(f'We are testing Y~binim({n}, {p})')
    print(f'Theoretical vs simulated P(Y = {check_k}): {theo_pob[check_k]:.4f} vs {p_k:.4f}')
    print(f'Theoretical vs simulated P(Y <= {check_k}): {theo_cdf[check_k]:.4f} vs {p_se_k:.4f}')

    return p_k, p_se_k
    # An aside from sim_binom performance
    # np choice vs np random
    # timeit('r.choice(2, size=10, p=[0.9, 1-0.9]).sum()', 'import numpy as np;r=np.random.default_rng()', number=100_000)
    # 0.717, 0.738, 0.737
    # timeit('(r.random(10) < 0.1).sum()', 'import numpy as np;r=np.random.default_rng()', number=100_000)
    # 0.164, 0.164, 0.178


######## Coin Toss Payoff R-Studio Q2 ###########
def w2_RS_Q2a_plt_payoff(
        payoff_func: str = 'k ** 2 - 7 * k',
        ntosses: int = 10,
        plot: bool = True,
    ) -> np.ndarray:
    k = np.arange(ntosses + 1)
    payoff = eval(payoff_func, locals={'k': k})

    if plot:
        fig, ax = plt.subplots()
        ax.plot(k, [0] * (ntosses + 1))
        ax.plot(k, payoff, '.')

    return payoff


def w2_RS_Q2b_decide_game_value(
        ntosses: int = 10,
        p: float = 0.6,
        print_decision: bool = True
    ) -> float:
    """
    Compute EV for the game with certain payoff
    """
    payoff = w2_RS_Q2a_plt_payoff(ntosses=ntosses, plot=False)
    pmf, _ = plt_binom(n=ntosses, p=p, output=True, plot=False)

    ev = (payoff * pmf).sum()
    
    if print_decision:
        decision = 'not a' if ev < 0 else 'a'
        print(f'Game is {decision} good bet, because EV is: {ev:.2f}')
    
    return payoff, pmf, ev


def w2_RS_Q2c_sim_game(
        ntosses: int = 10,
        p: float = 0.6,
        ntrials: int = 250_000,
        method: str = 'not_loop',
        print_results: bool = True
    ) -> tuple[float, float]:
    payoff, _, theo_ev = w2_RS_Q2b_decide_game_value(ntosses=ntosses, p=p, print_decision=False)

    rng = np.random.default_rng()

    # timeit("w2_RS_Q2c_sim_game(method='loop', print_results=False)", setup='from mit_stats_intro import w2_RS_Q2c_sim_game', number=20)
    # 7.959, 8.148, 7.998
    # vs
    # timeit("w2_RS_Q2c_sim_game(method='not', print_results=False)", setup='from mit_stats_intro import w2_RS_Q2c_sim_game', number=20)
    # 0.288, 0.268, 0.266
    if method == 'loop':
        payout = 0
        for _ in range(ntrials):
            payout += payoff[(rng.random(ntosses) < p).sum()]
    else:
        payout = payoff[(rng.random(ntrials * ntosses).\
                         reshape((ntrials, ntosses)) < p).sum(axis=1)].sum()
    
    payout /= ntrials

    if print_results:
        print(f'Theoretical EV is {theo_ev:.2f}, vs simulated avg. payout of {payout:.2f}')

    return theo_ev, payout

#%%
if __name__ == '__main__':
    main()