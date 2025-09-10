from vero.metrics import MetricBase
import numpy as np
import re

class NumericalHallucinationScore(MetricBase):
    name:str = 'numerical_hallucination_score'

    def __init__(self, answer:str, chunks_retrieved:list|str, chunks=[], k=20):
        self.answer = answer
        self.chunks_retrieved = chunks_retrieved
        self.chunks = chunks
        self.k = k

    def evaluate(self):
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