import math
from typing import Optional, Tuple
from transformers import get_linear_schedule_with_warmup, AutoConfig
from transformers import BertForPreTraining, BertModel, RobertaModel, AlbertModel, AlbertForMaskedLM, RobertaForMaskedLM
import torch
from torch.optim import AdamW
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from dataclasses import dataclass


class BERTAlignModel(pl.LightningModule):
    def __init__(self, model='roberta_large', using_pretrained=True, *args, **kwargs) -> None:
        super().__init__()
        # Already defined in lightning: self.device
        self.save_hyperparameters()
        self.model = model

        if using_pretrained:
            self.base_model = RobertaModel.from_pretrained(model)
            self.mlm_head = RobertaForMaskedLM.from_pretrained(model).lm_head
        else:
            self.base_model = RobertaModel(AutoConfig.from_pretrained(model))
            self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head
    
        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.reg_layer = nn.Linear(self.base_model.config.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)
        
        self.need_mlm = True
        self.is_finetune = False
        self.mlm_loss_factor = 0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch):
        if 'electra' in self.model:
            return self.electra_forward(batch)
        base_model_output = self.base_model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch.keys() else None
            )
        
        prediction_scores = self.mlm_head(base_model_output.last_hidden_state) ## sequence_output for mlm
        seq_relationship_score = self.bin_layer(self.dropout(base_model_output.pooler_output)) ## pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        total_loss = None
        if 'mlm_label' in batch.keys(): ### 'mlm_label' and 'align_label' when training
            ce_loss_fct = nn.CrossEntropyLoss(reduction='sum')
            masked_lm_loss = ce_loss_fct(prediction_scores.view(-1, self.base_model.config.vocab_size), batch['mlm_label'].view(-1)) #/ self.con vocabulary
            next_sentence_loss = ce_loss_fct(seq_relationship_score.view(-1, 2), batch['align_label'].view(-1)) / math.log(2)
            tri_label_loss = ce_loss_fct(tri_label_score.view(-1, 3), batch['tri_label'].view(-1)) / math.log(3)
            reg_label_loss = self.mse_loss(reg_label_score.view(-1), batch['reg_label'].view(-1), reduction='sum')

            masked_lm_loss_num = torch.sum(batch['mlm_label'].view(-1) != -100)
            next_sentence_loss_num = torch.sum(batch['align_label'].view(-1) != -100)
            tri_label_loss_num = torch.sum(batch['tri_label'].view(-1) != -100)
            reg_label_loss_num = torch.sum(batch['reg_label'].view(-1) != -100.0)

        return ModelOutput(
            loss=total_loss,
            all_loss=[masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss]  if 'mlm_label' in batch.keys() else None,
            loss_nums=[masked_lm_loss_num, next_sentence_loss_num, tri_label_loss_num, reg_label_loss_num] if 'mlm_label' in batch.keys() else None,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions
        )

@dataclass
class ModelOutput():
    loss: Optional[torch.FloatTensor] = None
    all_loss: Optional[list] = None
    loss_nums: Optional[list] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    tri_label_logits: torch.FloatTensor = None
    reg_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None