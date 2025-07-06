import pandas as pd
import numpy as np
import scipy.interpolate as inter
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TFR:
    std_ages = np.array([15, 22, 27, 32, 37, 42, 49])
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


def interpolate_tfr(
        asfr_df: pd.DataFrame,
        plot: bool = True
    ) -> pd.DataFrame:
    # Get curve shape of the asfr distribution
    curve_shape = inter.pchip_interpolate(TFR.std_ages, # x axis -> standard ages with end and mid pts
                                          asfr_df.loc[TFR.std_ages, 'asfr'], # corresponding asfr
                                          asfr_df.index) # interpolate to all ages
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(asfr_df.index, curve_shape)

        plt.show()

    return curve_shape

main()