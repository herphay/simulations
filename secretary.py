import numpy as np
# import pandas as pd

def secretary(
        ncandidates: int = 100,
        ntrials: int = 10_000,
        look_phase_cutoff: float = 1 / np.e
    ) -> None:
    rng = np.random.default_rng()

    candidates = np.empty((ntrials, ncandidates), dtype=int)
    candidates[:] = np.arange(ncandidates)

    rng.permuted(candidates, axis=1, out=candidates)

    # Look phase, note down the best candidate
    look_phase_cutoff = round(look_phase_cutoff * ncandidates)
    
    best_sofar = candidates[:, :look_phase_cutoff].max(axis=1)[:, None]
    idx_best = np.argmax(candidates > best_sofar, axis=1)
    idx_best = np.where(idx_best == 0, ncandidates - 1, idx_best)

    return idx_best, candidates, best_sofar



#%%
def main() -> None:
    ...


main()