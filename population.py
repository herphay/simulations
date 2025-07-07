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
    
    final_asfr = pd.DataFrame(curve_shape, index=asfr_df.index, columns=asfr_df.columns)
    section_sum = final_asfr.groupby(final_asfr.index // 5).sum()
    section_sum.index = np.arange(15, 50, 5)
    section_sum = section_sum.reindex(np.arange(15, 50)).ffill()

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
    if not asfr_grp:
        asfr_grp = TFR.asfr_grp_2024
    
    birth_df = create_asfr_df(asfr_grp, TFR.start_age_2024)
    birth_df = interpolate_tfr(birth_df)
    birth_df.index.name = 'age'
    birth_df = pd.concat({'Female': birth_df}, names=['sex'])

    death_df = process_lifetable()

    death_df.loc[('Female', slice(15, 49)), 'asfr'] = birth_df
    # Slicing check: d.loc[('Female',slice(14, 50)), ]

    return birth_df, death_df


main()