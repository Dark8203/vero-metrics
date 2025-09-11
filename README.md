# RAG Evaluation Framework
 A framework for evaluating Retrieval-Augmented Generation (RAG) pipelines with built-in tracing, logging and evaluation metrics.

## Project Details
 The project provides a comprehensive framework for **evaluating** Retrieval-Augmented Generation (RAG) pipelines. It includes simple RAG implementations and a robust tracing system that logs every step of the pipeline's execution to an *SQLite* database for detailed analysis and debugging.
 It has built in **metrics** for deep and end to end evaluation of all the different blocks in RAG pipelines i.e. query, retriever, reranker and generator.
 The core idea is to trace key information for each RAG run—such as the user query, the retrieved context, and the final LLM-generated answer, store it systematically and then **evaluate them based on *ground truth***.

## Project Structure
```
.
├── src/
|   ├── vero/
|   │   ├── metrics/          # Main package for metrics
|   └── └──  all the metrics  # All the metrics are in here
└── tests/
    └── test_main.py/         # file for all the testing

```


## Deep Dive into metrics
### Generation Metrics
#### BERTScore
 * Uses BERTScorer.
 * Pass retrieved context and generated output.
 * Returns precision, recall and f1-score.

#### ROUGE-L
* Uses ROUGE-L which focuses on the Longest Common Subsequence (LCS) between a generated summary and a reference summary.
* Pass retrieved context and generated output.
* Returns ROUGE score - precesion, recall and f1-score.

#### SEMScore
* Uses embeddings of retrieved context and generated output and calculated cosine similarity between them.
* Pass retrieved context and generated output.
* Returns a single SEMScore.
  
| SEMScore       | Inference     |
| -------------- | ------------- |
| closer to 1    | more semantically similar  |
| closer to 0    | unrelated  |
| negative score | semantically opposite |

#### BleurtScore (Weighted Semantic Similarity)
* A unique implementation of BluertScore where not only we calculate BluertScore but also perform weighted sum to give out the more nuanced score.
* With this implementation it can be pretty dynamic as it can be used as both generation and retriever metric.
   * As *generation metric* - it gives insights on which chunks play major part in output generation and they will recieve higher weights than others.
   * As *retriever metric* - it gives insights if their retriever is good at capturing conceptual and semantic relationships, even if it misses the exact answer.
     * It can be very useful for debugging, e.g.:
       * If Context Recall is low, but Weighted Semantic Similarity score is high, it tells the developer: "Your retriever is finding documents that are about the right topic, but it's failing to find the specific sentence or fact needed for the answer"
       * If both scores are low, the retriever is failing at a more fundamental level.
* Pass retrieved context and generated output or user query.
* Returns a single weight BluertScore.

| BluertScore       | Inference     |
| -------------- | ------------- |
| closer to 1    | high semantic similarity  |
| closer to 0    | low semantic similarity  |

#### AlignScore
* Measures the faithfulness of generated answer to the retrieved context.
* Pass retrieved context and generated output.
* Returns a single AlignScore.

| AlignScore       | Inference     |
| -------------- | ------------- |
| closer to 1    | high factual consistency  |
| closer to 0    | low factual consistency  |

#### BartScore
* Uses BartScorer and is a type of comparision score.
* Pass retrieved context and generated output.
* Returns a BartScore.
* This score does not hold any meaning in itself, it can be used to compare two models or versions of RAG pipelines and comparision can done as - higher the score better the generation capabilites of that pipeline compared to another.

#### G-Eval
* A unique implementation of g-eval where we calculate the weighted sum of all the possible scores with their linear probabilities and get the average of it as the final score.
* We provide the prompting capability where if you want you can provide your own custom prompt for evaluation or you can pass the metric name, metric description(optional) and we will generate the prompt for you.
* We also provide the polling capability which basically runs the g-eval any given number of times(default is 5) and get an average score as final score.
* Pass the references and candidate (optional : custom prompt, metric name, metric description, polling flag and polling number).
* Returns a final G-Eval score for the passed metric or prompt.

#### Domain Overlap Score
 * Calculates the domain specific overlap score.
 * Pass key terms and generated output.
 * Returns overlap score.

#### Numerical Hallucination Score
 * Calculates Numerical Hallucination Score.
 * Pass retrieved context and generated output.
 * Returns hallucination score.

### Ranking Metrics
#### Mean Reciprocal Rank (MRR)
* Direct implementation of MRR.
* Pass the reranked docs along with ground truth.
* Returns MRR.

#### Mean Average Precision (MAP)
* Direct implementation of MAP.
* Pass the reranked docs along with ground truth.
* Returns MAP.

#### Reranker NDCG@k
* Direct implementation of NDCG@k.
* Pass the reranked docs along with ground truth and k value.
* Returns the NDCG@k.

#### Cumulative NDCG
* Unique implementation of NDCG@k that can be used to evaluate the cumulative performance of retriever and reranker.
* Pass the reranked docs along with ground truth.
* Returns the NDCG.

### Retrieval Metrics
#### Precision Score
* Calculates Precision Score.
* Pass the retrieved context and the ground truth context.
* Returns the context precision score.

#### Recall Score
* Calculates Recall Score.
* Pass the retrieved context and the ground truth context.
* Returns the context recall score.

#### Context Sufficiency Score
* Calculates the sufficiency score of retrieved context for the user query.
* Uses LLM to score the metric.
* Returns the context sufficiency score.

#### Citation Score
* Calculates Citation Score of the retrieved context.
* Pass the cited context and ground truth citations.
* Returns the citation score.