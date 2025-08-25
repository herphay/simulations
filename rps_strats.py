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