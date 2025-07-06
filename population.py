import pandas as pd
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TFR:
    # Singapore 2024 ASFR data
    asfr_grp_2024 = np.array([2.3, 9.8, 42.6, 79.3, 50, 10.2, 0.7])
    start_age_2024 = np.arange(15, 50, 5)

def main() -> None:
    ...


def create_asfr_df(
        asfr_grp: np.ndarray,
        start_age: np.ndarray,
        plot: bool = True
    ) -> pd.DataFrame:
    asfr_df = pd.DataFrame({'asfr': [np.nan] * 35}, index=np.arange(15, 50))
    asfr_df['asfr'] = pd.DataFrame({'asfr': asfr_grp}, index=start_age)['asfr']
    asfr_df = asfr_df.ffill()

    if plot:
        fig, ax = plt.subplots()
        ax.plot(asfr_df.index, asfr_df.asfr)
        plt.show()
    
    return asfr_df

main()