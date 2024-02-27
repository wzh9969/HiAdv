from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os


class BertPoolingLayer(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x


class HTCModel(BertPreTrainedModel):
    def __init__(self, config, loss='CE', avg='cls', multi_label=True, use_graph=False, path_list=None, data_path=None,
                 **kwargs):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pooler = BertPoolingLayer(config)
        self.avg = avg
        self.loss = loss
        self.multi_label = multi_label
        self.use_graph = use_graph
        if self.use_graph:
            from .graph import GraphEncoder
            self.graph_encoder = GraphEncoder(config, path_list=path_list, data_path=data_path)
            self.transform_layer = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        if self.use_graph:
            label_dict = torch.load(os.path.join(data_path, 'value_dict.pt'))
            tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
            label_dict = {i: tokenizer.encode(v, add_special_tokens=False) for i, v in label_dict.items()}
            label_dict[len(label_dict)] = tokenizer.encode('Root', add_special_tokens=False)
            label_emb = []
            input_embeds = self.get_input_embeddings()
            for i in range(len(label_dict)):
                label_emb.append(
                    input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
            label_emb = torch.stack(label_emb, dim=0)
            new_embedding = torch.cat(
                [torch.zeros(1, label_emb.size(-1), device=label_emb.device, dtype=label_emb.dtype),
                 label_emb], dim=0)
            self.graph_encoder.graph_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids if inputs_embeds is None else None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        if self.avg == 'avg':
            pooled_output = self.pooler(sequence_output.mean(dim=1))
        else:
            pooled_output = self.pooler(sequence_output[:, 0, :])
        pooled_output = self.dropout(pooled_output)

        if not self.use_graph:
            logits = self.classifier(pooled_output)
        else:
            label_emb = self.graph_encoder()[:-1,:]
            label_emb = self.transform_layer(label_emb)
            logits = torch.mm(pooled_output, label_emb.transpose(1, 0))

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            elif not self.multi_label:
                if self.loss == 'CE':
                    loss_fct = CrossEntropyLoss()
                else:
                    raise NotImplementedError
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss_fct = nn.BCEWithLogitsLoss()
                label_mask = labels != -100
                logits_flat = logits.view(-1, self.num_labels).masked_select(label_mask)
                labels = labels.to(torch.float32).masked_select(label_mask)
                loss = loss_fct(logits_flat, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        if self.avg == 'avg':
            pooled_output = self.pooler(sequence_output.mean(dim=1))
        else:
            pooled_output = self.pooler(sequence_output[:, 0, :])
        pooled_output = self.dropout(pooled_output)

        if not self.use_graph:
            logits = self.classifier(pooled_output)
        else:
            label_emb = self.graph_encoder()[:-1, :]
            label_emb = self.transform_layer(label_emb)
            logits = torch.mm(pooled_output, label_emb.transpose(1, 0))
        predict_labels = []
        for scores in logits:
            predict_labels.append([])
            for i, score in enumerate(scores):
                if score > 0:
                    predict_labels[-1].append(i)
        return predict_labels, logits
