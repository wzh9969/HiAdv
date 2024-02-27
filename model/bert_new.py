from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os
from .graph import GraphEncoder
from .loss import multilabel_categorical_crossentropy


class BertPoolingLayer(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        return x


class HTCModel(BertPreTrainedModel):
    def __init__(self, config, loss='CE', avg='cls', multi_label=True, use_graph=False, path_list=None, data_path=None,
                 label_mask=None, double_inf=False, layer=1, **kwargs):
        super().__init__(config)
        if label_mask is None:
            label_mask = []
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pooler = BertPoolingLayer(config)
        self.avg = avg
        self.loss = loss
        self.multi_label = multi_label
        self.use_graph = use_graph
        path_list = sorted(path_list, key=lambda x: x[1])
        self.path_list = path_list
        if double_inf:
            self.transform_layer_in = nn.Embedding(3, config.hidden_size, 0)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if self.use_graph:
            self.graph_encoder = GraphEncoder(config, path_list=path_list, data_path=data_path, layer=layer)
            if double_inf:
                self.graph_encoder_two = GraphEncoder(config, path_list=path_list, data_path=data_path, layer=layer)
                self.classifier_two = nn.Linear(config.hidden_size, config.num_labels)
                self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
                self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
                self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
                self.w_proj = nn.Linear(config.hidden_size, config.hidden_size)
                self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

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

    def init_embedding(self):
        if hasattr(self, 'transform_layer_in'):
            tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
            label_dict = [tokenizer.encode(v, add_special_tokens=False) for v in ['true', 'false']]
            label_emb = []
            input_embeds = self.get_input_embeddings()
            for i in label_dict:
                label_emb.append(
                    input_embeds.weight.index_select(0, torch.tensor(i, device=self.device)).mean(dim=0))
            new_embedding = torch.stack(label_emb, dim=0)
            new_embedding = torch.cat(
                [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
                 new_embedding], dim=0)
            self.transform_layer_in = nn.Embedding.from_pretrained(new_embedding, False, 0)


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
            predict=None,
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

        if not self.use_graph:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            label_emb = pooled_output
        else:
            pooled_output = self.dropout(pooled_output)
            label_emb = self.get_output_emb(input_ids.size(0), pooled_output, predict=predict)
            if predict is None:
                logits = self.classifier(label_emb)
            else:
                logits = self.classifier_two(label_emb)

        loss = None
        if labels is not None:
            if predict is not None:
                assert predict.shape == labels.shape
                labels = predict - 1
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
                if self.loss == 'BCE':
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels.to(torch.float32))
                else:
                    loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels),
                                                               logits.view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'label_emb': label_emb,
            'pooled_output': pooled_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

    def get_output_emb(self, batch_size, text_embedding, predict=None):
        if predict is None:
            label_emb = self.graph_encoder(text_embedding)[:, -1, :]
        else:
            fake_node = torch.tensor([0] * batch_size, dtype=torch.long, device=self.device)
            label_emb = self.transform_layer_in(torch.cat([predict, fake_node.unsqueeze(-1)], dim=-1))
            label_emb = self.graph_encoder_two(text_embedding, label_emb=label_emb)[:, -1, :]

        return label_emb

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, token_type_ids, predict=None, **kwargs):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        if self.avg == 'avg':
            pooled_output = self.pooler(sequence_output.mean(dim=1))
        else:
            pooled_output = self.pooler(sequence_output[:, 0, :])

        if not self.use_graph:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            label_emb = pooled_output
        else:
            pooled_output = self.dropout(pooled_output)
            label_emb = self.get_output_emb(input_ids.size(0), pooled_output, predict=predict)
            if predict is None:
                logits = self.classifier(label_emb)
            else:
                logits = self.classifier_two(label_emb)

        predict_labels = []
        for scores in logits:
            predict_labels.append([])
            for i, score in enumerate(scores):
                if score > 0:
                    predict_labels[-1].append(i)
        return predict_labels, logits
