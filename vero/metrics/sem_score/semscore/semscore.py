from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator
from accelerate.utils import gather_object
from tqdm import tqdm
import torch, gc
import torch.nn as nn


class EmbeddingModelWrapper():
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"

    def __init__(self, model_path=None, bs=8):
        if model_path is None: model_path = self.DEFAULT_MODEL
        self.model, self.tokenizer = self.load_model(model_path)
        self.bs = bs
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def load_model(self, model_path):
        model = AutoModel.from_pretrained(
            model_path,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
        )
        return model, tokenizer

    def emb_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_embeddings(self, sentences):
        embeddings = torch.tensor([])

        if self.bs is None:
            batches = [sentences]
        else:
            batches = [sentences[i:i + self.bs] for i in range(0, len(sentences), self.bs)]

        for sentences in batches:
            encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            batch_embeddings = self.emb_mean_pooling(model_output, encoded_input['attention_mask'])

            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)

        return embeddings

    def get_similarities(self, x, y=None):
        if y is None:
            num_samples = x.shape[0]
            similarities = [[0 for i in range(num_samples)] for f in range(num_samples)]
            for row in tqdm(range(num_samples)):
                similarities[row][0:row + 1] = self.cos(x[row].repeat(row + 1, 1), x[0:row + 1]).tolist()
            return similarities
        else:
            return self.cos(x, y).tolist()
