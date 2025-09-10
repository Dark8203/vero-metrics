from vero.metrics import MetricBase
import numpy as np

class OverlapScore(MetricBase):
    name = 'overlap_score'

    def __init__(self,answer:list, key_terms:list):
        self.answer = answer
        self.key_terms = key_terms

#TODO: make it case agnostic?
    def evaluate(self) -> float:
        if len(self.key_terms) == 0:
            return np.nan
        score = 0
        for i in self.key_terms:
            if i in self.answer:
                score += 1
        return score / len(self.key_terms)