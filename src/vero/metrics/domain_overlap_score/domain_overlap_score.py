from vero.metrics import MetricBase
import numpy as np

class OverlapScore(MetricBase):
    '''
        Calculates the domain specific overlap score.

        :param answer: Pass the answer.
        :param key_terms: Pass the key terms for domain.

        Methods
        ---------
        1. __init__(answer, key_terms)
            Initializes the metric.
        2. evaluate() -> list
            Returns the overlap score.

        :returns: float
        '''
    name = 'overlap_score'

    def __init__(self,answer:str, key_terms:list):
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