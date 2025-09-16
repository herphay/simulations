import numpy as np


class always_rock:
    def __init__(self, pop_count):
        ...
    
    @staticmethod
    def get_plays(pop_count):
        return 'r' * pop_count
    
    @staticmethod
    def get_name():
        return 'always_rock'

class always_paper:
    @staticmethod
    def get_plays(pop_count):
        return 'p' * pop_count
    
    @staticmethod
    def get_name():
        return 'always_paper'


class always_scissor:
    @staticmethod
    def get_plays(pop_count):
        return 's' * pop_count
    
    @staticmethod
    def get_name():
        return 'always_scissor'
    

class equal:
    @staticmethod
    def get_plays(pop_count):
        rng = np.random.default_rng()
        choice = rng.random(pop_count)
        plays = np.full(pop_count, 'r')
        plays[choice < 1/3] = 'p'
        plays[choice > 2/3] = 's'
        return ''.join(plays)
    
    @staticmethod
    def get_name():
        return 'equal_chance'


class rp_12:
    @staticmethod
    def get_plays(pop_count):
        rng = np.random.default_rng()
        choice = rng.random(pop_count)
        plays = np.full(pop_count, 'p')
        plays[choice < 1/3] = 'r'
        # plays[choice > 1/3] = 'p'
        return ''.join(plays)
    
    @staticmethod
    def get_name():
        return '1/3r,2/3p'


class rs_12:
    @staticmethod
    def get_plays(pop_count):
        rng = np.random.default_rng()
        choice = rng.random(pop_count)
        plays = np.full(pop_count, 's')
        plays[choice < 1/3] = 'r'
        # plays[choice > 1/3] = 's'
        return ''.join(plays)
    
    @staticmethod
    def get_name():
        return '1/3r,2/3s'
    

class gen_species:
    @staticmethod
    def get_plays(
            affinity: np.ndarray
        ):
        """
        affinity: np.ndarray
            a 3xm array where m is the number of individuals in the population
            1st row is the individual's probability to play rock, 2nd is paper, 3rd is scissors
        """
        rng = np.random.default_rng()
        choice = rng.random(affinity.shape[1])
        plays = np.full(affinity.shape[1], 's')
        plays[choice < affinity[0]] = 'r'
        plays[choice > affinity[1]] = 'p'
        # plays[choice > 1/3] = 's'
        return ''.join(plays)
    
    @staticmethod
    def validate_affinity(
            affinity: np.ndarray
        ):
        """
        Returns True if affinity is valid, otherwise False
        """
        total_prob = affinity.sum(axis=0)
        return sum(abs(total_prob - 1) > 0.000001) == 0