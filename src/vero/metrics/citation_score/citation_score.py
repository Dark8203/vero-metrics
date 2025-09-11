from vero.metrics import MetricBase
import numpy as np


#TODO: not right implementation, to be improved or changed
class CitationScore(MetricBase):
    '''
        Calculates Citation Score.

        :param chunks_cited: Pass the chunks cited.
        :param chunks_true: Pass the true chunks for reference.

        Methods
        ---------
        1. __init__(chunks_cited, chunks_true)
            Initializes the metric.
        2. evaluate() -> float
            Returns the citation score.

        :returns: float
        '''
    name = 'citation_score'

    def __init__(self,chunks_cited:list, chunks_true:list, k=20):
        self.chunks_cited = chunks_cited
        self.chunks_true = chunks_true
        self.k = k

    def evaluate(self) -> float:
        if len(self.chunks_cited) == 0:
            return np.nan

        score = 0
        for i in self.chunks_cited:
            if i in self.chunks_true:
                score += 1
        return score / len(self.chunks_cited)