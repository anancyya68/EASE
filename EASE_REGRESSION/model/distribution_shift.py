# @File : distribution_shift.py
# @Time : 2024/5/20 17:24
# @Author :

import numpy as np
import pandas as pd
from scipy.special import kl_div
from scipy.spatial.distance import jensenshannon


class DistributionShiftEvaluator:
    def __init__(self, P, Q):
        """
        :param P: DataFrame, first distribution
        :param Q: DataFrame, second distribution
        """
        self.P = P
        self.Q = Q

    def compute_js_divergence(self):
        """
        compute Jensen-Shannon between two distribution
        :param P: first distribution
        :param Q: second distribution
        :return: Jensen-Shannon
        """
        P = np.array(self.P)
        Q = np.array(self.Q)
        max_len = max(P.size, Q.size)
        P_flat = np.pad(P.flatten(), (0, max_len - P.size), 'constant')
        Q_flat = np.pad(Q.flatten(), (0, max_len - Q.size), 'constant')
        js_divergence = jensenshannon(P_flat, Q_flat)
        return js_divergence




