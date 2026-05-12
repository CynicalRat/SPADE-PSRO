import numpy as np

class PayoffTable:
    def __init__(self, table:np.array=None):
        self.table = table if table is not None \
            else np.zeros((0, 0))

    def expand(self, new_size):
        old_size = self.table.shape[0]
        if new_size > old_size:
            new_table = np.zeros((new_size, new_size))
            new_table[:old_size, :old_size] = self.table
            self.table = new_table

    def update(self, i, j, payoff:dict):
        self.table[i, j] = payoff[1]
        self.table[j, i] = payoff[0]

    def sample_strategy(self, meta_nash):
        return np.random.choice(len(meta_nash), p=meta_nash)
    
    def refresh_table(self, table:np.array):
        self.table = table
