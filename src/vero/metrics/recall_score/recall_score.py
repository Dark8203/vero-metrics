from vero.metrics import MetricBase
import numpy as np

class RecallScore(MetricBase):
    '''
        Calculates Recall Score.

        :param chunks_retrieved: Pass the chunks retrieved.
        :param chunks_true: Pass the true chunks for reference.

        Methods
        ---------
        1. __init__(chunks_retrieved, chunks_true)
            Initializes the metric.
        2. evaluate() -> float
            Returns the recall score.

        :returns: float
        '''
    name = 'recall_score'

    def __init__(self,chunks_retrieved:list, chunks_true:list, k=20):
        self.chunks_retrieved = chunks_retrieved
        self.chunks_true = chunks_true
        self.k = k

    def evaluate(self) -> float:
        ch_r = self.chunks_retrieved[:self.k]
        if len(ch_r) == 0:
            return np.nan
        sc = 0
        for i in self.chunks_true:
            if i in ch_r:
                sc += 1
        return sc / len(self.chunks_true)

