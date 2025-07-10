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


def create_pop_df(
        asfr_grp: np.ndarray | None = None,
        life_table_year: int = 2019,
    ) -> pd.DataFrame:
    ...


def create_asfr_df(
        asfr_grp: np.ndarray,
        start_age: np.ndarray,
        plot: bool = True
    ) -> pd.DataFrame:
    """
    Create a dataframe of Age Specific Fertility Rate.

    Dataframe is only for females aged 15-49. Assumption is that outside of this range, fertility is 
    0.
    """
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
    """
    Usually asfr are given in bin size of 5 years each, thus there is a need to interpolate to get 
    the singel year asfr.

    This func interpolates using the pchip func, and anchors the start/end point using the min/max 
    ages, while non-extreme bins are anchored using the midpoint of said bin. This is to prevent -ve 
    asfr values 
    """
    # Get curve shape of the asfr distribution
    curve_shape = inter.pchip_interpolate(TFR.std_ages, # x axis -> standard ages with end and mid pts
                                          asfr_df.loc[TFR.std_ages, 'asfr'], # corresponding asfr
                                          asfr_df.index) # interpolate to all ages
    
    # Create initial df with the interpolated curve, age 15-49
    final_asfr = pd.DataFrame(curve_shape, index=asfr_df.index, columns=asfr_df.columns)
    # Group the interpolated values into their respective 5-yr age bins (15-19 etc)
    section_sum = final_asfr.groupby(final_asfr.index // 5).sum()
    # Update index to the first age of the bin (15, 20 etc.)
    section_sum.index = np.arange(15, 50, 5)
    # reindex, which is to expand the index to include every age rather than just the 1st age of bin
    # then forward fill so each bin is the sum of the interpolated values of the bin.
    # This will help us get the weights of each age in the bin relative to other bin ages
    section_sum = section_sum.reindex(np.arange(15, 50)).ffill()

    # Get each age's weight and the final asfr for said age
    final_asfr = final_asfr / section_sum * asfr_df * 5
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(asfr_df.index, asfr_df['asfr'])
        ax.plot(asfr_df.index, final_asfr['asfr'])
        ax.plot(asfr_df.index, curve_shape)

        plt.show()

    return final_asfr


def process_lifetable(
        path: str = r'data/lifetables.csv',
        year: int = 2019,
    ) -> pd.DataFrame:
    """
    Reach the csv containing SG's lifetables and process it into a usable df.
    """
    lifetable = pd.read_csv(path)
    lifetable = lifetable.loc[(lifetable['year'] == year) & (lifetable['sex'] != 'Total'), 
                              ['sex', 'age_x', 'lx', 'dx']]
    lifetable['age_x'] = lifetable['age_x'].str.replace('100_and_over', '100').astype(np.int64)

    lifetable['qx'] = lifetable['dx'] / lifetable['lx']

    lifetable.columns = lifetable.columns.str.replace('age_x', 'age')

    return lifetable.set_index(['sex', 'age'])


def integrate_birth_death(
        asfr_grp: np.ndarray | None = None,
        life_table_year: int = 2019,
    ) -> pd.DataFrame:
    """
    Combine asfr df (determines birth) and lifetable df (determines death) into 1 overall df that 
    will allow for birth and death calculation.
    """
    if not asfr_grp:
        asfr_grp = TFR.asfr_grp_2024
    
    birth_df = create_asfr_df(asfr_grp, TFR.start_age_2024)
    birth_df = interpolate_tfr(birth_df)
    birth_df.index.name = 'age'
    birth_df = pd.concat({'Female': birth_df}, names=['sex'])

    death_df = process_lifetable()

    death_df.sort_index(inplace=True)

    death_df.loc[('Female', slice(15, 49)), 'asfr'] = birth_df
    # Slicing check: d.loc[('Female',slice(14, 50)), ]

    return birth_df, death_df


main()