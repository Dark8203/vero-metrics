from vero.metrics import MetricBase
import numpy as np


#TODO: not right implementation, to be improved or changed
class CitationScore(MetricBase):
    name = 'citation_score'

    def __init__(self,chunks_cited:list, chunks_true:list, k=20):
        self.chunks_cited = chunks_cited
        self.chunks_true = chunks_true
        self.k = k

    def evaluate(self):
        if len(self.chunks_cited) == 0:
            return np.nan

        score = 0
        for i in self.chunks_cited:
            if i in self.chunks_true:
                score += 1
        return score / len(self.chunks_cited)