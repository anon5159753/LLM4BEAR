import numpy as np
import pandas as pd

# Not used during the EGPO mechanism, but you could edit and use this if you wanted to.

class Metric():
    def __init__(self, score_list, true_list, conf) -> None:
        self.score_list = score_list
        self.true_list = true_list
        self.conf = conf

    def rmse(self): # No need to pass lists if you're using self
        valid_indices = self.score_list != None

        valid_scores = self.score_list[valid_indices].astype(float)
        valid_trues = self.true_list[valid_indices].astype(float)

        # Check if there are any valid scores to prevent division by zero
        if len(valid_scores) == 0:
            return 0.0

        return np.sqrt(np.mean((valid_scores - valid_trues)**2))

