from vero.metrics import MetricBase
import numpy as np
import re

class NumericalHallucinationScore(MetricBase):
    '''
        Calculates Numerical Hallucination Score.

        :param answer: Pass the answer to be checked.
        :param chunks_retrieved: Pass the chunks retrieved.
        :param chunks: Optional: Pass the list of ids for the retrieved chunks.

        Methods
        ---------
        1. __init__(chunks_retrieved, chunks_true)
            Initializes the metric.
        2. evaluate() -> float
            Returns the numerical hallucination score.

        :returns: float
        '''
    name:str = 'numerical_hallucination_score'

    def __init__(self, answer:str, chunks_retrieved:list|str, chunks:list =[], k=20):
        self.answer = answer
        self.chunks_retrieved = chunks_retrieved
        self.chunks = chunks
        self.k = k

    def evaluate(self) -> float:
        ch_ret = self.chunks_retrieved[:self.k]
        s_true = ''  ## true numbers in retrieved chunks
        if len(self.chunks) > 0:
            for i in ch_ret:
                s_true += ' <> ' + self.chunks[i]
        else:
            for i in ch_ret:
                s_true += ' <> ' + i
        # num_ret = []
        pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:[a-zA-Z%]+)?\b|\b\d+\.\d+(?:[a-zA-Z%]+)?\b|\b\d+(?:[a-zA-Z%]+)?\b'
        # for i in ch_ret:
        num_ret = re.findall(pattern, self.answer)
        if len(num_ret) == 0:
            return np.nan
        score = 0
        for i in num_ret:
            if i in s_true:
                score += 1
        return score / len(num_ret)