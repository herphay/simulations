import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.axes

import scipy.stats as scistat

from collections.abc import Iterable
from collections import Counter

from typing import Literal

from time import perf_counter

def main() -> None:
    """
    This file is for MIT OpenCourseWare - Introduction to Probability and Statistics
    18.05 | Spring 2022 | Undergraduate

    Course is conducted using R -> I will translate it into python
    """

#%%
# Week 1 - general R samples
def w1_sample(
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


######## Derangement R-Studio Q3 ###########
def w2_RS_Q3_derangement(
        nobjects: int = 10,
        ntrials: int = 100_000,
    ) -> float:
    rng = np.random.default_rng()

    base = np.arange(nobjects)

    derangement_count = 0
    for _ in range(ntrials):
        if not any(base == rng.permutation(nobjects)):
            derangement_count += 1
    
    # Theoretical derangement count
    nfact = math.factorial(nobjects)
    theo_derange_prob = int(nfact / math.e + 0.5) / nfact

    exp_derange_prob = derangement_count / ntrials

    print(f'Theoretical derangement probability is {theo_derange_prob:.3f} ' + 
          f'vs simulation prob of {exp_derange_prob:.3f}')


######## W2 Pset Q6 ###########
def w2_Pset_Q6_longest_run(
        ntosses: int = 50,
        p: float = 0.5,
        ntrials: int = 10_000,
        repeats: int = 3,
        check_be_len: int = 8,
    ) -> np.ndarray:
    """
    Find the longest run in a sequence of coin tosses
    """
    def find_longest_run(arr):
        max_len = 0
        run_len = 1
        for i in range(len(arr) -  1):
            if arr[i] == arr[i + 1]:
                run_len += 1
            else:
                max_len = max(max_len, run_len)
                run_len = 1

        return max(max_len, run_len)
    
    rng = np.random.default_rng()

    avg_run = np.zeros(3)
    prob_run = np.zeros(3)

    for i in range(repeats):
        sim_tosses = rng.integers(2, size=ntrials * ntosses).reshape((ntrials, ntosses))
        run_lens = np.fromiter((find_longest_run(arr) for arr in sim_tosses), dtype=int)

        avg_run[i] = run_lens.sum() / ntrials
        prob_run[i] = (run_lens >= check_be_len).sum() / ntrials
    
    return avg_run, prob_run


#%%
# Week 3:
def w3_Selfcheck_plt_hist(
        data_points: int = 100_000,
        bins: int = 20,
        bar_width: float = 0.85
    ) -> None:
    rng = np.random.default_rng()
    scatter = rng.random(data_points)
    plt.hist(scatter, bins=bins, rwidth=bar_width)


def w3_C5_mu_var_calculator(
        values: np.ndarray,
        pmf: np.ndarray,
        print_results: bool = True
    ) -> tuple[float, float]:
    mu = (values * pmf).sum()
    variance = (pmf * (values - mu) ** 2).sum()

    if print_results:
        print(f'Mean: {mu} | Var: {variance}')
    
    return mu, variance

def w3_C5_E2():
    fig, axs = plt.subplots(2, 2)
    
    v1 = np.arange(1,6)
    pmf1 = np.array([1/5] * 5)
    w3_C5_mu_var_calculator(v1, pmf1)
    axs[0, 0].bar(v1, pmf1)

    pmf2 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    w3_C5_mu_var_calculator(v1, pmf2)
    axs[0, 1].bar(v1, pmf2)

    pmf3 = np.array([0.5, 0, 0, 0, 0.5])
    w3_C5_mu_var_calculator(v1, pmf3)
    axs[1, 0].bar(v1, pmf3)

    pmf4 = np.array([0, 0, 1, 0, 0])
    w3_C5_mu_var_calculator(v1, pmf4)
    axs[1, 1].bar(v1, pmf4)

    pmf5 = np.array([0.2, 0.3, 0.5])
    v5 = np.array([1, 2,4])
    w3_C5_mu_var_calculator(v5, pmf5)


def w3_s3_q1a(
        rate: float = 1,
        nsamples: int = 1000,
        bin_width: float | None = None
    ) -> None:
    """
    Draw frequency histogram of an exponential distribution
    Rate example (lm): 5 cars / minute, 1 defect / meter
    pdf = -lm * e^(-lm * x)
    cdf = 1 - e^(-lm * x)
    range: [0, inf)
    """
    rng = np.random.default_rng()
    bin_width = 1 / rate / 3

    ### np exponential dist generator use SCALE instead of RATE ###
    # Scale is the inverse of rate -> scale = 1 / rate (rate is lambda)
    # Rate is the arrival rate, scale is then the average time between arrivals
    sample = rng.exponential(1 / rate, nsamples) # generate exponential samples
    bins = np.arange(0, sample.max() + bin_width, bin_width) # get the binds

    # bins parameter allows us to define all the bins [a, b, c ... z] is [a,b), [b,c) until [y,z] 
    plt.hist(sample, bins=bins)


def w3_s3_q1b(
        rate: float = 1,
        nsamples: int = 1000,
        bin_width: float = 0.4,
        exp_point_count: int = 101
    ) -> None:
    rng = np.random.default_rng()
    sample = rng.exponential(1 / rate, nsamples)
    max_sim = sample.max()
    bins = np.arange(0, max_sim + bin_width, bin_width)

    x_points = np.linspace(0, max_sim, exp_point_count)
    exp_points = rate * math.e ** (-rate * x_points)

    fig, ax = plt.subplots()
    ax.hist(sample, bins=bins, density=True)
    ax.plot(x_points, exp_points)


def w3_s3_q2a(
        rate: float = 1,
        nsamples: int = 1000,
        n_to_avg: int = 2,
        bin_width: float = 0.4
    ) -> None:
    """
    Simulate the average of n_to_avg exponential distributed samples: Y = (X1 + X2 + ... Xn) / n. 
    """
    rng = np.random.default_rng()
    exp_data = rng.exponential(1 / rate, size=nsamples * n_to_avg)
    exp_data = np.average(exp_data.reshape((nsamples, n_to_avg)), axis=1)

    bins = np.arange(0, exp_data.max() + bin_width, bin_width)

    plt.hist(exp_data, bins=bins, density=True)


def w3_s3_q2b(
        rate: float = 1,
        nsamples: int = 1000,
        n_to_avg: int = 2,
        bin_width: float | None = None,
        theo_point_count: int = 101
    ) -> None:
    rng = np.random.default_rng()

    exp_data = rng.exponential(1 / rate, size=nsamples * n_to_avg)
    exp_data = np.average(exp_data.reshape((nsamples, n_to_avg)), axis=1)

    # rate is interpreted as the amount of incidents per unit
    # E.g. 10 cars passing per min, or 2 defect per meter
    # Mean of the exp. distribution is the 1/rate, and the s.d. is 1/rate^2
    # Mean of the average of N identical dist. is the mean of each dist.
    # s.d. of the average of N identical dist. is then found using CLT
    # CLT states that s.d. of the avg. is the s.d. of each dist / root of N
    mean_of_avg = 1 / rate
    std_of_avg = mean_of_avg / n_to_avg ** 0.5

    if not bin_width:
        bin_width = std_of_avg / 4
    bins = np.arange(max(0, mean_of_avg - 4 * std_of_avg), 
                     exp_data.max() + bin_width, bin_width)
    
    theo_x_points = np.linspace(max(0, mean_of_avg - 4 * std_of_avg), 
                                max(exp_data.max(), mean_of_avg + 4 * std_of_avg), 
                                theo_point_count)
    
    # Under CLT, the theoretical dist. of the avg. becomes a normal dist.
    theo_density = math.e ** -(((theo_x_points - mean_of_avg) / std_of_avg) ** 2 / 2) / \
                   (std_of_avg * (2 * math.pi) ** 0.5)
    
    plt.hist(exp_data, bins=bins, density=True)
    plt.plot(theo_x_points, theo_density)


def w3_pset3_data():
    # Pre-prosessing (original file given as 1 row of data)
    # with open('data/pset3_data.csv') as f:
    #     a = f.readline()
    # with open('data/pset3_data.csv', 'w') as f:
    #     writer = csv.writer(f,delimiter=',')
    #     writer.writerows([[v] for v in a.split(',')])

    source = r'data/pset3_data1.csv'
    # data = pd.read_csv(source, header=None).T[0].to_numpy()
    data = np.genfromtxt(source, delimiter=',')

    plt.hist(data, bins=50)
    # ~20% survive >= 5 years
    # ~50% still dies within 15 months
    # So treatment is helping for 50% of the time with only 20% complete cure rate
    return data


#%%
def w4_lec_bq1():
    data = [1, 1.2, 1.3, 1.6, 1.6, 2.1, 2.2, 2.6, 
            2.7, 3.1, 3.2, 3.4, 3.8, 3.9, 3.9]
    
    # plt.hist offers only left close bins
    # use pd.cut to get bin count for right close bins
    # rightclose_bin_count = pd.cut(data, np.arange(0, 4, 0.5), right=True).value_counts()
    # rightclose_bin_count.plot(kind='bar')
    bins_0 = np.arange(0, 4.1, 0.5)
    bins_1 = [0, 1, 3, 4]

    fig, axes = plt.subplots(3, 1)

    axes[0].hist(data, bins_0, edgecolor='k')
    axes[1].hist(data, bins_1, edgecolor='k')
    axes[2].hist(data, bins_1, edgecolor='k', density=True)

    # return rightclose_bin_count


def w4_lec_extra1():
    # (a)
    x = scistat.norm.ppf([0.25, 0.5, 0.75])
    print('Quantiles of Z at 0.25/0.5/0.75 are:', x)
    # (b)
    pts = np.arange(-4, 4, 0.05)
    density = np.e ** (-pts ** 2 / 2) / (2 * np.pi) ** 0.5
    plt.plot(pts, density)
    # (c)
    print('Cumulative prob at quantiles are:', scistat.norm.cdf(x))


def w4_pset4_2():
    # (a) 50% will vote for Alexandra, poll of 400, prob of >52.5% vote A
    print('Prob of Alexandra getting >52.5% =', 1 - scistat.norm.cdf(0.525, 0.5, 1 / 40))
    # (b) 
    print('Prob of others getting <31% =', scistat.norm.cdf(0.31, 0.3, (0.3 * 0.7 / 400) ** 0.5))
    

def w4_pset4_3():
    # 1000 orders, each rounded to nearest 5th cent (.57 to .55, .58 to .6)
    # Prob of daily rounding error being >100 or <-100 cents
    # mean = 0, var = 2 for single order, discrete uniform dist
    # mean = 0, var = 2000 for 1000 order, normal dist
    print('Probability of >100 or <-100 =', 2 * scistat.norm.cdf(-100, 0, 2000 ** 0.5))

    # bonus part to simulate
    def rounding_sim(orders: int = 1000, ntrials: int = 10_000):
        rng = np.random.default_rng()
        roundings = rng.integers(-2, 3, (orders, ntrials))
        roundings = abs(roundings.sum(axis=0))
        exceed_pct = (roundings > 100).sum() / ntrials
        return exceed_pct
    
    for _ in range(3):
        print(rounding_sim())


def w4_pset4_6():
    # IQ has mean=100, s.d.=15. Find P(IQ>160)
    print('IQ >160 has probability =', scistat.norm.sf(160, 100, 15))
    # mod IQ with norm dist, mean=0, s.d.=3^0.5
    print('norm mod_IQ P(>4 s.d.) =', scistat.norm.sf(4 * 3 ** 0.5, 0, 3 ** 0.5))
    # mod IQ with t dist, mean=0, s.d.=3^0.5
    # scipy t dist can takes: x value to check survival func (1 - cdf(x)),
    # degree of freedom (df=3 implies s.d.=3**0.5)
    # loc=0 default -> mean
    # scale=1 default -> not s.d. unlike norm, here it scales the s.d.
    # scipy calc s.d. by: s.d. = scale * (df / (df - 2)) ** 0.5
    print('t mod_IQ P(>4 s.d.) =', scistat.t.sf(4 * 3 ** 0.5, df=3))

    # (d)(i) norm vs t-3 tail probabilities
    print('Normal(0, 3**0.5) @ 20, 40, 200:', 
          '\n', scistat.norm.sf(20, loc=0, scale=3 ** 0.5),
          '\n', scistat.norm.sf(40, loc=0, scale=3 ** 0.5),
          '\n', scistat.norm.sf(200, loc=0, scale=3 ** 0.5))
    
    print('T dist, 3df @ 20, 40, 200:', 
          '\n', scistat.t.sf(20, df=3),
          '\n', scistat.t.sf(40, df=3),
          '\n', scistat.t.sf(200, df=3))



def w4_pset4_6_Tdist(
        edge: float = 3 ** 0.5 * 4
    ):
    """Plot t distribution"""
    # t dist. have s.d. = 3 ** 0.5
    x = np.linspace(-edge, edge, 1000)
    t_fx = 2 / 3 / np.pi * (1 + x ** 2 / 3) ** - 2
    normal_fx = 1 / 3 ** 0.5 / (2 * np.pi) ** 0.5 * np.e ** (- x ** 2 / 6) 
    plt.plot(x, t_fx)
    plt.plot(x, normal_fx)


#%%
def w5_lec_binom(
        n: int = 100,
        p: float = 0.5,
        x: int = 50
    ) -> float:
    return math.comb(n, x) * p ** x * (1 - p) ** (n - x)


def w5_lec_binom_cdf(
        n: int = 100,
        p: float = 0.5,
        x: int = 50
    ) -> float:
    cum_prob = 0
    for i in range(x + 1):
        cum_prob += w5_lec_binom(n, p, i)
    # print(cum_prob - scistat.binom.cdf(x, n, p))
    return cum_prob

def w5_lec_qn_sample():
    """Useful chatgpt solving method"""
    # import sympy as sp

    # x, y = sp.symbols('x y')
    # f = 2*x**3 + 2*y**3

    # # Verify integration over [0, 1] x [0, 1]
    # total_prob = sp.integrate(f, (x, 0, 1), (y, 0, 1))
    # print(f"{total_prob=}")

    # # Means
    # ex = sp.integrate(x * f, (x, 0, 1), (y, 0, 1))
    # ey = sp.integrate(y * f, (x, 0, 1), (y, 0, 1))

    # # E[XY]
    # exy = sp.integrate(x * y * f, (x, 0, 1), (y, 0, 1))

    # # Covariance
    # cov = exy - ex * ey

    # print(f"{ex=}")
    # print(f"{ey=}")
    # print(f"{exy=}")
    # print(f"{cov=}")


def w5_s4_q1a(
        ntgt: int = 10,
        nalone: int = 10,
        ntrials: int = 1_000_000,
        print_results: bool = True,
        return_results: bool = False
    ):
    """
    A and B play roulette together ntgt times
    B then plays alone nalone times
    Roulette have 18 red, 18 black, 2 green.
    A & B always bet 1 on red. Win you double, loose you lose the bet
    """
    rng = np.random.default_rng()
    # Simulate [0,1) and < 18/38 wins
    payout = rng.random((ntrials, ntgt + nalone))
    payout = np.where(payout < 18 / 38, 1, -1)
    A = payout[:, :ntgt].sum(axis=1)
    B = payout.sum(axis=1)

    mean_A = np.mean(A)
    mean_B = np.mean(B)
    cov = np.cov(A, B)
    var_A = cov[0, 0]
    var_B = cov[1, 1]
    cov_AB = cov[1, 0]
    cor_AB = np.corrcoef(A, B)[1, 0]

    if print_results:
        print(f'A plays {ntgt} times, B plays {ntgt + nalone} times.')
        print(f'Over {ntrials} sims:')
        print(f'Mean winning of A: {mean_A}  ||  Variance: {var_A}')
        print(f'Mean winning of B: {mean_B}  ||  Variance: {var_B}')
        print(f'Covariance(A, B): {cov_AB}  || Correlation(A, B): {cor_AB}')
        # theoretical
        # Single trial
        mean_X = 18/38 - 20/38
        var_X = 1 - mean_X ** 2
        cov_theo = ntgt * var_X
        cor_theo = (ntgt / (ntgt + nalone)) ** 0.5
        print(f'Theoretical single game mean: {mean_X}  ||  Variance: {var_X}')
        print(f'Theoretical Covariance(A, B): {cov_theo}  || Correlation(A, B): {cor_theo}')
    
    if return_results:
        return cov_AB, cor_AB


def w5_s4_q1b(
        nalone_limit: int = 41
    ):
    """
    How cov and cor varies as nalone increase
    """
    x = np.arange(nalone_limit)
    covs = np.array([])
    cors = np.array([])
    for nalone in x:
        cov, cor = w5_s4_q1a(nalone=nalone, ntrials=1_000_000, print_results=False, return_results=True)
        covs = np.append(covs, cov)
        cors = np.append(cors, cor)
    
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(x, covs)
    axs[0].set_title('Covaraince as nalone increase')
    axs[0].set_xlabel('nalone')
    axs[0].set_ylabel('Covariance')

    axs[1].plot(x, cors)
    axs[1].set_title('Correlation as nalone increase')
    axs[1].set_xlabel('nalone')
    axs[1].set_ylabel('Correlation')

    plt.tight_layout()


def w5_s4_q2(
        n_bets: int = 100,
        ntrials: int = 100_000
    ):
    # theoretical single trial
    mean = 18 / 38 - 20 / 38
    var = 1 - mean ** 2
    sd = var ** 0.5

    # CLT as n_bets increase sum approach a normal dist X~N(n*mean, n*var)
    N_mean = n_bets * mean
    N_sd = sd * n_bets ** 0.5
    axis = np.linspace(N_mean - 4 * N_sd, N_mean + 4 * N_sd, 1000)

    y = 1 / N_sd / (2 * math.pi) ** 0.5 * math.e ** (-((axis - N_mean) / N_sd) ** 2 / 2)

    # simulate the bets
    rng = np.random.default_rng()
    payout = rng.random((ntrials, n_bets))
    payout = np.where(payout < 18 / 38, 1, -1)
    
    payout = payout.sum(axis=1)

    nbin = np.arange(payout.min() - 1, payout.max() + 3, 2)
    
    plt.hist(payout, bins=nbin, density=True)
    plt.plot(axis, y)


def w7_c11_bayesian_updating(
        prior: np.ndarray = np.array([0.4, 0.4, 0.2]),
        likelihood_h: np.ndarray = np.array([0.5, 0.6, 0.9]),
        trials: int = 300
    ) -> None:
    """
    prior: the base probability set for each of the possible hypothesis
    likelihood_h: the probability of getting heads for each hypothesis
    tials: the number of trials to run the experiment for

    Simulates if each of the 3 types of coins are chosen, each coin have likelihood_h probability of
    flipping heads. 
    """
    rng = np.random.default_rng()
    # Get rolls for each of the dice trials time
    rolls = rng.random((3, 1, trials))

    # Get the roll results the attach the equivalent likelihood -> 3x3xn
    likelihood_h = likelihood_h.reshape((3,1))
    likelihoods = np.where(rolls < likelihood_h.reshape(3, 1, 1), likelihood_h, 1 - likelihood_h)

    # Attach prior to get numerator chain
    prior = np.tile(prior.reshape((3, 1)), (3, 1, 1))
    numerators = np.concatenate((prior, likelihoods), axis=2)
    # posterior is then the cumulative product of all the prior & likelihood and normalized
    numerators = numerators.cumprod(axis=2)
    posterior = numerators / numerators.sum(axis=1).reshape((3, 1, trials + 1))

    # Plot results
    labels = [f'{likelihood_h[0][0]}', f'{likelihood_h[1][0]}', f'{likelihood_h[2][0]}']
    fig, ax = plt.subplots(3, 1)
    for i in range(3):
        ax[i].plot(posterior[i].T, label=labels)
        ax[i].legend()
    plt.tight_layout()
    # print(posterior[:, :, -1])


class dice_data:
    def __init__(
            self, 
            dice: list[int] = [4, 6, 8, 12, 20], 
            prior: list[float] = [0.2] * 5,
            censored: bool = False
        ):
        self.hypothesis = dice, censored
        self.prior = prior
        self.censored = censored

    def __repr__(self):
        if self.censored:
            cols = [0, 1]
        else:
            cols = np.arange(1, self.likelihood.shape[1] + 1)
        identity = f'{len(self.hypothesis)} dice in set with sides: ' +\
                   f'{", ".join(self.hypothesis.astype(str))}\n' +\
                   f'Corresponding prior probabilities of {", ".join(self.prior.astype(str))}\n' +\
                   f'Likelihood table for the dice:\n' +\
                   pd.DataFrame(self._likelihood, 
                                index=self._hypothesis, 
                                columns=cols).\
                                    to_string(line_width=80)

        return identity
    
    @property
    def censored(self):
        return self._censored
    
    @censored.setter
    def censored(self, censor):
        self._censored = censor

    @property
    def hypothesis(self):
        return self._hypothesis

    @hypothesis.setter
    def hypothesis(self, composite):
        self._hypothesis = np.array(composite[0])
        self._likelihood_setter(composite[1])
    
    @property
    def prior(self):
        return self._prior

    @prior.setter
    def prior(self, prior):
        if sum(prior) != 1:
            raise ValueError('Priors must sum up to 1')
        self._prior = np.array(prior)

    @property
    def likelihood(self):
        return self._likelihood
    
    def _likelihood_setter(self, censored):
        # create nx1 array for the prob. of rolling any number for each x-sided die
        p_anynumber = 1 / self._hypothesis[:, np.newaxis] 

        # create n x max_side array (True for within a die's range, False for outside of it)
        # np.newaxis works like None -> create a new axis in specified location.
        # So for 3, array indexed [:, np.newaxis], it becomes 3x1
        if not censored:
            mask = np.arange(self._hypothesis.max()) < self._hypothesis[:, np.newaxis]
            
            # Create the final likelihood table
            self._likelihood = np.where(mask, p_anynumber, 0)
        else:
            self._likelihood = np.concat([1 - p_anynumber, p_anynumber], axis=1)



def w7_s5_q1b(
        dice_sides: list[int] = [4, 6, 8, 12, 20], 
        prior: list[float] = [0.2] * 5,
        chosen: int = 8,
        nrolls: int = 8,
        plot_posteriors: bool = True,
        plot_ind_posterior: bool = False,
        return_data: bool = False
    ):
    """
    Select 1 specific die, simulate it's rolls and plot out the posterior probabilities
    """
    dice = dice_data(dice=dice_sides, prior=prior)
    rng = np.random.default_rng()

    # roll the dice
    rolls = rng.integers(chosen, size=nrolls)

    # get the corresponding likelihood for each roll
    likelihoods = dice.likelihood[:, rolls] # likelihood is indexed by column of each roll
    numerators = np.concat([dice.prior[:, None], likelihoods], axis=1) 
    numerators = numerators.cumprod(axis=1)
    posteriors = numerators / numerators.sum(axis=0)

    if plot_posteriors:
        x_labels = ['prior'] + [str(i) for i in range(1, nrolls + 1)]
        stacking = np.zeros(nrolls + 1)
        for i in range(len(dice_sides)):
            plt.bar(x_labels, posteriors[i], bottom=stacking, label=f'D{dice_sides[i]}')
            stacking += posteriors[i]
        
        plt.legend()
        plt.xlabel('rolls')
        plt.ylabel('Probability')
        plt.tight_layout()
        plt.show()
    
    if plot_ind_posterior:
        dice_names = [f'D{i}' for i in dice_sides]
        for i in range(nrolls + 1):
            plt.bar(dice_names, posteriors[:, i])
            plt.show()
    
    if return_data:
        return posteriors


def w7_s5_q1c():
    w7_s5_q1b(prior=[0.2] * 5, nrolls=20)
    w7_s5_q1b(prior=[0.001, 0.001, 0.001, 0.001, 0.996], nrolls=20)


def w7_s5_q1d():
    w7_s5_q1b(prior=[0.25, 0.25, 0, 0.25, 0.25], nrolls=20)


def w7_s5_q2a(
        dice_sides: list[int] = [4, 6, 8, 12, 20], 
        prior: list[float] = [0.2] * 5,
        nrolls: int = 30,
        return_data: bool = False
    ):
    """
    Data is censored, data is 1 if roll is 1, 0 otherwise
    """
    dice = dice_data(dice=dice_sides, prior=prior, censored=True)
    rng = np.random.default_rng()

    # Choose a die, then get its probability of getting 1
    chosen = rng.integers(dice.hypothesis.size)
    p1 = dice.likelihood[chosen][1]

    # generates which rolls get 1 which dont
    rolls = (rng.random(size=nrolls) <= p1).astype(int)
    likelihoods = dice.likelihood[:, rolls]
    numerator = np.concat([dice.prior[:, np.newaxis], likelihoods], axis=1)
    numerator = numerator.cumprod(axis=1)
    posteriors = numerator / numerator.sum(axis=0)

    x_labels = ['prior'] + [str(i) for i in range(1, nrolls + 1)]
    stacking = np.zeros(nrolls + 1)
    for i in range(dice.prior.size):
        plt.bar(x_labels, posteriors[i], bottom=stacking, label=f'D{dice_sides[i]}')
        stacking += posteriors[i]
    
    plt.legend()
    plt.xlabel('rolls')
    plt.ylabel('Probability')
    plt.title(f'Posterior probabilities with censored data. Chosen dice: D{dice_sides[chosen]}')
    plt.tight_layout()
    plt.show()

    if return_data:
        return posteriors

    
def w8_s6_q0(
        ntrials: int = 10_000
    ):
    """
    Histogram of 1 data point from N(10,6^2) and avg of 9 data points
    """
    rng = np.random.default_rng()

    data_1 = rng.normal(10, 6, ntrials)
    data_2 = rng.normal(10, 6, (ntrials, 9)).sum(axis=1) / 9
    range = (min(data_1.min(), data_2.min()), max(data_1.max(), data_2.max()))

    fig, axs = plt.subplots(1, 2)

    axs[0].hist(data_1, range=range, bins=36, density=True)
    axs[1].hist(data_2, range=range, bins=36, density=True)
    fig.tight_layout()

    print("Standard deviations of data1, data2 are:\n", data_1.std(), data_2.std())


def w8_s6_q1a():
    print('Cauchy distribution formula:')
    print('f(x; theta, gamma) = 1/pi * (gamma / ((x - theta)^2 + gamma^2))')
    print('f(x; theta, gamma) = 1/pi * (1 / ((x - theta)^2 + 1))')


def w8_s6_q1b(
        mu: float = 0,
        sigma: float = 1
    ):
    """Compares Normal pdf to Cauchy pdf"""
    x = np.linspace(-4.5, 4.5, 1001)
    normal = 1 / (sigma * (2 * np.pi) ** 0.5) * np.e ** (-(x - mu) ** 2 / (2 * sigma ** 2))
    cauchy = 1 / np.pi * (sigma / ((x - mu) ** 2 + sigma ** 2))

    plt.plot(x, normal, color='tab:orange')
    plt.plot(x, cauchy, color='blue')
    plt.title('PDF of Normal (orange) and Cauchy (blue) distributions')


def w8_s6_q1c():
    print("We say Cauchy dist have fat tails because Cauchy's pdf on the far left/right are higher")


def w8_s6_q1d(
        x0: float = 0,
        scale: float = 1,
        ntrials: int = 10_000,
        minmax: float = 10
    ):
    x = np.linspace(-10, 10, 1001)
    cauchy1 = 1 / np.pi * (scale / ((x - x0) ** 2 + scale ** 2))

    min_x = x0 - minmax * scale
    max_x = x0 + minmax * scale
    
    rng = np.random.default_rng()
    data1 = rng.standard_cauchy(ntrials) * scale + x0
    data1 = data1[(data1 > min_x) & (data1 < max_x)]
    data2 = (rng.standard_cauchy((ntrials, 9)) * scale + x0).sum(axis=1) / 9
    data2 = data2[(data2 > min_x) & (data2 < max_x)]

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(x, cauchy1)
    ax[0].hist(data1, density=True, bins=36)
    ax[1].plot(x, cauchy1)
    ax[1].hist(data2, density=True, bins=36)
    fig.tight_layout()

    print('Averaging the data does NOT change the spread of the histogram!!')


def w8_s6_q2data() -> list:
    return [-0.491220417425751, -7.29807825846222,  0.445026391301098, 
            -2.01399156118547,  -3.31926706437694,  -2.09199513293618, 
            -0.66458098096206,  -34.5102687569877,  -0.0679571835900508, 
            8.08881636741333,   15.5319265619357,   25.3777623364045, 
            0.720533188131477,  -1.31825660397482,  -1.40917663604347]


def w8_s6_q2a():
    data = w8_s6_q2data()
    plt.scatter(np.arange(1, 16), data)


def w8_s6_q2b(
        theta_min: float = -10,
        theta_max: float = 10,
        dtheta: float = 0.02,
        scale: float = 1
    ):
    data = w8_s6_q2data()

    # Discretize priors by getting midpoint of each range
    theta = np.arange(theta_min + dtheta / 2, theta_max, dtheta)
    probabilities = np.zeros(shape=(len(data) + 1, theta.size))
    probabilities[0] = 1 / theta.size

    for i, dpoint in enumerate(data):
        # Calculate likelihood -> in this case just need density, no need actual probability
        # This is because calculating prob like area of trapezium is just scaling
        # Scaling will cancel out -> this is the likelihood principal
        # Basically, likelihood that are proportional will result in the same posterior
        likelihoods = 1 / np.pi * (scale / ((dpoint - theta) ** 2 + scale ** 2))
        bayesian_numerators = probabilities[i] * likelihoods
        probabilities[i + 1] = bayesian_numerators / bayesian_numerators.sum()

    from matplotlib.colors import LinearSegmentedColormap
    color_start = 'lightblue'
    color_end = 'navy'
    cmap = LinearSegmentedColormap.from_list("my_gradient", [color_start, color_end])
    fig, ax = plt.subplots(3, 1, figsize=(6, 6))
    for i, prob in enumerate(probabilities):
        ax[0].plot(theta, prob, color=cmap(i / (probabilities.shape[0] - 1)))
    
    MAP_estimate = theta[probabilities.argmax(axis=1)]
    ax[1].plot(np.arange(probabilities.shape[0]), MAP_estimate)

    ax[2].plot(theta, probabilities[-1])
    ax[2].axvline(x=MAP_estimate[-1], color='red', linestyle='--')
    
    fig.tight_layout()

    print(f'Look for the obscure path at location: {MAP_estimate[-1]:.3f}')


def w9_lec16_bq1b(
        prior_type: Literal['flat', 'informed'] = 'informed'
    ):
    if prior_type == 'informed':
        prior = [
            [7, 7, 7, 49],
            [1, 1, 1,  7],
            [1, 1, 1,  7],
            [1, 1, 1,  7],
        ]
    elif prior_type == 'flat':
        prior = [
            [1, 1, 1,  1],
            [1, 1, 1,  1],
            [1, 1, 1,  1],
            [1, 1, 1,  1],
        ]

    prior = np.array(prior)
    prior = prior / prior.sum()

    return prior

def w9_lec16_bq1c():
    def likelihood(e, c):
        return 29 * e ** 28 * (1 - e) * 210 * c ** 6 * (1 - c) ** 4
    p = np.arange(1,9,2) / 8
    likelihoods = np.array([likelihood(e, c) for c in p 
                                             for e in p]).reshape((4,4))
    # print(likelihoods.sum())
    # likelihoods /= likelihoods.sum()
    # print(likelihoods)
    likelihoods_df = pd.DataFrame(likelihoods, columns=p, index=p)
    # print(likelihoods_df)
    return likelihoods


def w9_lec16_bq1d(
        prior_type: Literal['flat', 'informed']
        ):
    prior = w9_lec16_bq1b(prior_type)
    likelihoods = w9_lec16_bq1c()

    posterior = prior * likelihoods
    posterior /= posterior.sum()

    p = np.arange(1,9,2) / 8
    p = p.astype(str)
    posterior = pd.DataFrame(posterior, columns='E_' + p, index='C_' + p)

    s = posterior.stack().sort_values(ascending=False)
    top1_idx = s.index[0]
    top2_idx = s.index[1]

    print(f'Most likely parameter is {top1_idx} with {s[top1_idx]:.1%}')
    print(f'2nd most likely parameter is {top2_idx} with {s[top2_idx]:.1%}')
    print(f'Total probability where E is more effective than C is ' +
          f'{np.triu(posterior, k=1).sum():.1%}')
    print(f'Probability that E - C >= 0.6 is {np.triu(posterior, k=3).sum():.3%}')

    return posterior
    

def w9_s7_q1(
        theta_HA: float = 0.7, 
        alpha: float = 0.05, 
        n_tosses: int = 18,
        print_results: bool = True,
        return_reject: bool = False
        ):
    """
    H0 is that the coin is fair
    theta_HA: P(head) for an unfair coin, to keep it simple, theta_HA must be >0.5
    alpha: significance level
    n_tosses: the number of coin tossed in 1 trial

    test stat is # of heads
    """
    if theta_HA <= 0.5:
        raise ValueError('theta_HA must be > 0.5')
    
    reject_region = []
    actual_alpha = 0
    power = 0

    for i in range(n_tosses, -1, -1):
        h0_p = scistat.binom.pmf(i, n_tosses, 0.5)
        ha_p = scistat.binom.pmf(i, n_tosses, theta_HA)

        if actual_alpha + h0_p < alpha:
            actual_alpha += h0_p
            power += ha_p
            reject_region.append(i)
        else:
            break
    
    if print_results:
        print(f"Reject region: {sorted(reject_region)}")
        print(f"Actual significance: {actual_alpha:.7f}")
        print(f"Power: {power:.7f}")

    if return_reject:
        return min(reject_region), actual_alpha, power


def w9_s7_q2(
        theta_HA: float = 0.7, 
        alpha: float = 0.05, 
        n_tosses: int = 18,
        n_trials: int = 1000,
        secret_prior: float = 0.3,
    ):
    """
    H0 is that the coin is fair
    theta_HA: P(head) for an unfair coin, to keep it simple, theta_HA must be >0.5
    alpha: significance level
    n_tosses: the number of coin tossed in 1 trial
    n_trials: # of trials to run in the simulation
    secret_prior: P(H0) used to choose whether H0 or HAis used for each trial

    test stat is # of heads
    """
    print('#' * 45)
    min_reject, significance, power = w9_s7_q1(theta_HA, alpha, n_tosses, return_reject=True)
    print('#' * 45)

    rng = np.random.default_rng()
    coin_choice = rng.random(size=n_trials)[:, np.newaxis] < secret_prior # select H0 or HA
    num_H0 = (coin_choice * 1).sum()
    num_HA = n_trials - num_H0
    coin_choice = np.where(coin_choice, 0.5, theta_HA) # each trial get the P(heads)
    
    tosses = rng.random(size=(n_trials, n_tosses))

    heads = tosses < coin_choice
    head_count = heads.sum(axis=1)[:, np.newaxis]

    rejected = head_count >= min_reject
    outcome_vector = coin_choice + rejected

    num_rejected = rejected.sum()
    num_nonRejected = n_trials - num_rejected
    num_t1_error = (outcome_vector == 1.5).sum()
    num_t2_error = (outcome_vector == theta_HA).sum()

    if num_H0 != 0:
        p_rej_given_H0 = num_t1_error / num_H0
    else:
        p_rej_given_H0 = '0, no H0 for prob'

    p_H0_given_rej = num_t1_error / num_rejected

    if num_HA != 0:
        p_rej_given_HA = (num_HA - num_t2_error) / num_HA # this is the actual Power
    else:
        p_rej_given_HA = '0, no HA for prob'

    p_HA_given_rej = 1 - p_H0_given_rej
    p_rej = num_rejected / n_trials

    print(f'Alt coin P(heads)={theta_HA}, alpha={alpha}')
    print(f'n_tosses={n_tosses}, n_trials={n_trials}')
    print(f'Secret prior: P(H0)={secret_prior}, P(HA)={1 - secret_prior}')
    print(f'Number of rejections:  {num_rejected}')
    print(f'Number of type 1:  {num_t1_error}')
    print(f'Number of type 2:  {num_t2_error}')
    print(f'P(rejection | H0):  {p_rej_given_H0}')
    print(f'P(H0 | rejection):  {p_H0_given_rej:.7f}')
    print(f'P(rejection | HA):  {p_rej_given_HA}')
    print(f'P(HA | rejection):  {p_HA_given_rej:.7f}')
    print(f'P(rejection):  {p_rej:.7f}')

    return significance, power


def w9_s7_q3a():
    w9_s7_q2(n_trials=10_000, secret_prior=1)
    print('#' * 45)
    print('P(rejection | H0) will always be around the significance level (this is the definition)')
    print('P(H0 | rejection) is 1 here, since there are no HA ever, it has no meaning')
    print('P(HA | rejection) is 0, compliment of the above, sinec HA is never true')
    print('P(rejection | HA) is undefined, as no HA to contingent on')


def w9_s7_q3b():
    w9_s7_q2(n_trials=10_000, secret_prior=0)
    print('#' * 45)
    print('P(rejection | H0) is undefined, as no H0 to contingent on')
    print('P(H0 | rejection) is 0 here, since there are no H0 ever, it has no meaning')
    print('P(HA | rejection) is 1, compliment of the above, sinec HA is always true')
    print('P(rejection | HA) is the power, and will always be around the theoretical power')


def w9_s7_q3c():
    print('P(H0 | rejection) is basically the posterior probability, while it can depend on ' +
          'the likelihood P(rejection | H0) which is the significance, it is equally influenced ' +
          'by the prior probability and can vary from 0 to 1. For frequentists, ' +
          'this is meaningless.')
    

def w9_s7_q3d():
    print('THE SIGNIFICANCE IS NOT THE PROBABILITY OF AN ERROR GIVEN REJECTION')
    print("It is the probability of rejection given H0.")
    print("Frequentists don't compute P(Error|rejection)")


def w9_s7_q4(
        theta_HA: float = 0.7, 
        alpha: float = 0.05, 
        n_tosses: int = 18,
        n_trials: int = 1000,
        secret_prior: float = 0.3,
    ):
    significance, power = w9_s7_q2(theta_HA, alpha, n_tosses, n_trials, secret_prior)
    print('#' * 45)

    p_reject = significance * secret_prior + power * (1 - secret_prior)
    theo_p_H0_given_rej = significance * secret_prior / p_reject
    theo_p_HA_given_rej = power * (1 - secret_prior) / p_reject

    print(f'Theoretical P(H0 | rejection), the posterior of H0, is: {theo_p_H0_given_rej:.7f}')
    print(f'Theoretical P(HA | rejection), the posterior of H0, is: {theo_p_HA_given_rej:.7f}')


def w9_pset9_q1e(
        H0: float = 0.5,
        HA: float = 0.55,
        alpha: float = 0.05,
        power: float = 0.9,
    ) -> int:
    """
    Check the exact number of tosses required to get a certain power for a coin with P(H) of either 
    H0 or HA, and a significance level of alpha. 2 sided reject region.
    """
    # Find the normal estimtate first
    n_est = (
        (scistat.norm.ppf(1 - power) * (HA * (1 - HA)) ** 0.5 - 
         scistat.norm.ppf(1 - alpha / 2) * (H0 * (1 - H0)) ** 0.5) / (H0 - HA)
    ) ** 2

    n_start = int(0.9 * n_est)
    n_end = int(1.1 * n_est)

    for n in range(n_start, n_end):
        # # Find the left hand side critical value starting point, 0.95 from norm estimate
        # n_crit_left = int(0.95 * scistat.norm.ppf(alpha / 2) * (n * H0 * (1  -H0)) ** 0.5)
        # # Keep increasing crit value until we exceed the alpha (become unable to reject)
        # while 2 * scistat.binom.cdf(n_crit_left, n, H0) < alpha:
        #     n_crit_left += 1
        
        # # Get the critical values based on the alpha
        # n_crit_left -= 1
        # n_crit_right = n - n_crit_left
        n_crit_left = int(scistat.binom.ppf(alpha / 2, n, H0)) - 1
        n_crit_right = n - n_crit_left

        # If the actual power based on the crit values exceed the requisite power, return the n
        if (scistat.binom.cdf(n_crit_left, n, HA) + 
            1 - scistat.binom.cdf(n_crit_right - 1, n, HA)) > power:
            return n
    # Return error if somehow not able to solve
    return -1
        
#%%
if __name__ == '__main__':
    main()