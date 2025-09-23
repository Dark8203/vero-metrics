from metrics import *
from tqdm import tqdm
import pandas as pd

METRICS_REGISTRY = {
    cls.name: cls for cls in [
        SufficiencyScore,
        SemScore,
        BleurtScore,
        BartScore,
        BertScore,
        AlignScore,
        CitationScore,
        CumulativeNDCG,
        OverlapScore,
        GEvalScore,
        MeanRR,
        MeanAP,
        RerankerNDCG,
        NumericalHallucinationScore,
        PrecisionScore,
        RecallScore,
        RougeScore,
    ]
}

MODEL_METRICS = {
    'align_score',
    'bart_score',
    'bert_score',
    'bleurt_score',
    'rouge_score',
    'sem_score',
}

MATH_METRICS = {
    'citation_score',
    'cumulative_ndcg',
    'overlap_score',
    'precision_score',
    'recall_score',
    'reranker_ndcg',
    'numerical_hallucination_score',
    'mean_average_precision',
    'mean_reciprocal_rank',

}

LLM_METRICS = {
    'g_eval_score',
    'sufficiency_score',
}

# eval = Evaluator()
# eval.evaluate(data)
# with BertScore() as bs:
#     bert_results = [bs.evaluate(chunk, ans) for chunk, ans in tqdm(zip(chunks_list, answers_list), total=len(df_new))]
# bert_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in bert_results]
# print(bert_dicts)

class Evaluator:
    name = 'evaluator'

    def __init__(self, metrics: list[str] | None = None):
        if metrics is None:
            metrics_to_run = list(METRICS_REGISTRY.keys())
        else:
            metrics_to_run = metrics

        self.evaluators = []
        for metric_name in metrics_to_run:
            if metric_name not in METRICS_REGISTRY:
                raise ValueError(f"Metric '{metric_name}' not supported currently.")

            evaluator_class = METRICS_REGISTRY[metric_name]
            self.evaluators.append(evaluator_class())

#TODO: data parsing first then continue with this
    def evaluate_math(self, reference_list, answers_list, metrics: list[str] | None = None):
        pass


    def evaluate(self, reference_list, answers_list, metrics: list[str] | None = None):
        pass
        # for metric in MODEL_METRICS:
        #
        #
        # print("Processing SemScore...")
        # with SemScore() as sem_score:
        #     sem_results = [sem_score.evaluate(chunk, ans) for chunk, ans in tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # print(sem_results)
        #
        # print("\nProcessing BERT Score...")
        # bs = BertScore()
        # with BertScore() as bs:
        #     bert_results = [bs.evaluate(chunk, ans) for chunk, ans in tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # bert_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in bert_results]
        # print(bert_dicts)
        #
        # print("\nProcessing RougeL Score...")
        # with RougeScore() as rouge:
        #     rouge_results = [rouge.evaluate(chunk, ans) for chunk, ans in tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # rouge_dicts = [{'Precision': p, 'Recall': r, 'F1score': f} for p, r, f in rouge_results]
        # print(rouge_dicts)
        #
        #
        #
        # print('BartScore')
        # with BartScore() as bart_score:
        #     score = [bart_score.evaluate(chunk, ans) for chunk, ans in tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # print(score)
        #
        # print("Processing BLUERTScore...")
        # with BleurtScore() as bleurt:
        #     bl_results = [bleurt.evaluate(chunk, ans) for chunk, ans in tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # print(bl_results)
        #
        # # TODO: Figure this fuck out
        #
        # print("Processing AlignScore...")
        # with AlignScore() as align:
        #     al_results = [align.evaluate(chunk, ans) for chunk, ans in tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # print(al_results)
        #
        #
        # print("\nProcessing G-Eval...")
        # with GEvalScore() as g_eval:
        #     g_eval_results = [g_eval.evaluate(chunk, ans, metric='Faithfulness') for chunk, ans in
        #                       tqdm(zip(reference_list, answers_list), total=len(df_new))]
        # print(g_eval_results)