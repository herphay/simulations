import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class TFR:
    group_asfr: np.ndarray = np.array([2.3, 9.8, 42.6, 79.3, 50, 10.2, 0.7])
    group_mean_age: np.ndarray = np.arange(17, 50, 5)

def main() -> None:
    ...



main()