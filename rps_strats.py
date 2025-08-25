import numpy as np


class always_rock:
    def __init__(self, pop_count):
        ...
    
    @staticmethod
    def get_plays(pop_count):
        return 'r' * pop_count

class always_paper:
    @staticmethod
    def get_plays(pop_count):
        return 'p' * pop_count


class always_scissor:
    @staticmethod
    def get_plays(pop_count):
        return 's' * pop_count
    

class equal:
    @staticmethod
    def get_plays(pop_count):
        rng = np.random.default_rng()
        choice = rng.random(pop_count)
        plays = np.full(pop_count, 'r')
        plays[choice < 1/3] = 'p'
        plays[choice > 1/3] = 's'
        return ''.join(plays)