from vero.metrics import MetricBase
import numpy as np
import math
# from rag_code.tracing_components import logger


class MeanAP(MetricBase):
    name = 'mean_average_precision'

    def __init__(self,reranked_docs: list, ground_truth: list):
        self.reranked_docs = reranked_docs
        self.ground_truth = ground_truth

    def evaluate(self) -> float | None:
        # logger.info('Starting MAP calculation...')
        avg_precision = []
        try:
            for docs, truth in zip(self.reranked_docs, self.ground_truth):
                precesion = 0
                doc_count = 0
                truth_length = len(truth)
                for i in range(len(docs)):
                    if docs[i] in truth:
                        doc_count += 1
                        precesion += doc_count / (i + 1)
                avg_precision.append(precesion / truth_length)

            map = round(sum(avg_precision) / len(avg_precision), 2)
            return map
        except Exception as e:
            # logger.info('Exception occured during MAP calculation\nError:', e)
            return None