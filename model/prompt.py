from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os
from .loss import multilabel_categorical_crossentropy
from .prompt_graph import GraphEncoder
import math
import torch.nn.functional as F


class GraphEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEmbedding, self).__init__()
        self.graph_type = graph_type
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
            self.graph_two = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.padding_idx = padding_idx
        self.original_embedding = embedding
        new_embedding = torch.cat(
            [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
             new_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)
        self.predict = None
        self.transform_layer_in = nn.Embedding(3, config.hidden_size, 0)

    @property
    def weight(self):
        def foo():
            # label prompt MASK
            edge_features = self.new_embedding.weight[1:, :]
            if self.predict is not None:
                predict = self.predict
                fake_nodes = torch.zeros(
                    (edge_features.size(0) - self.predict.size(0)),
                    dtype=torch.long, device=self.predict.device)
                predict = torch.cat([predict, fake_nodes], dim=0)
                label_emb = self.transform_layer_in(predict)
                edge_features = edge_features + label_emb
            if self.graph_type != '':
                # label prompt
                edge_features = edge_features[:-1, :]
                if self.predict is None:
                    edge_features = self.graph(edge_features, self.original_embedding)
                else:
                    edge_features = self.graph_two(edge_features, self.original_embedding)
                edge_features = torch.cat(
                    [edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return foo

    @property
    def raw_weight(self):
        def foo():
            edge_features = self.new_embedding.weight[1:, :]
            if self.predict is not None:
                predict = self.predict
                fake_nodes = torch.zeros(
                    (edge_features.size(0) - self.predict.size(0)),
                    dtype=torch.long, device=self.predict.device)
                predict = torch.cat([predict, fake_nodes], dim=0)
                label_emb = self.transform_layer_in(predict)
                edge_features = edge_features + label_emb
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return foo

    def forward(self, x):
        weight = self.weight()
        if len(weight.shape) == 3:
            out = []
            for a, w in zip(x, weight):
                out.append(F.embedding(a, w, self.padding_idx))
            x = torch.stack(out, dim=0)
        else:
            x = F.embedding(x, self.weight(), self.padding_idx)

        return x


class OutputEmbedding(nn.Module):
    def __init__(self, bias):
        super(OutputEmbedding, self).__init__()
        self.weight = None
        self.bias = bias

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)


class HTCModel(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None,
                 double_inf=False, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        self.output_transform = nn.Linear(config.hidden_size * len(depth2label), config.hidden_size)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'))
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long))
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0)

        self._init_weights(prompt_embedding)
        # label prompt mask
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        embedding = GraphEmbedding(self.config, input_embeds, label_emb, self.graph_type,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)
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
        embedding.transform_layer_in = nn.Embedding.from_pretrained(new_embedding, False, 0)

        self.set_input_embeddings(embedding)
        output_embeddings = OutputEmbedding(self.get_output_embeddings().bias)
        self.set_output_embeddings(output_embeddings)
        output_embeddings.weight = embedding.raw_weight
        self.vocab_size = output_embeddings.bias.size(0)
        output_embeddings.bias.data = nn.functional.pad(
            output_embeddings.bias.data,
            (
                0,
                embedding.size - output_embeddings.bias.shape[0],
            ),
            "constant",
            0,
        )
        self.graph_encoder = embedding.graph
        self.graph_encoder_two = embedding.graph_two

    def get_layer_features(self, layer, prompt_feature=None):
        labels = torch.tensor(self.depth2label[layer], device=self.device) + 1
        label_features = self.get_input_embeddings().new_embedding(labels)
        label_features = self.transform(label_features)
        label_features = torch.dropout(F.relu(label_features), train=self.training, p=self.config.hidden_dropout_prob)
        return label_features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            predict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        if predict is not None:
            outputs = []
            sequence_output = []
            for input_id, att_mask, tok_id, pred in zip(input_ids, attention_mask, token_type_ids, predict):
                self.get_input_embeddings().predict = pred
                output = self.bert(
                    input_id.unsqueeze(0),
                    attention_mask=att_mask.unsqueeze(0),
                    token_type_ids=tok_id.unsqueeze(0),
                    return_dict=return_dict,
                )
                sequence_output.append(output[0])
                outputs.append(self.cls(output[0]))
            prediction_scores = torch.cat(outputs, dim=0)
            sequence_output = torch.cat(sequence_output, dim=0)
        else:
            self.get_input_embeddings().predict = None

            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            sequence_output = outputs[0]
            prediction_scores = self.cls(sequence_output)

        self.get_input_embeddings().predict = None

        masked_lm_loss = None
        multiclass_logits = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                          prediction_scores.size(
                                                                                              -1))
        multiclass_logits = multiclass_logits[:,
                            self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))

            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            masked_lm_loss += multiclass_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        label_emb = sequence_output.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))).view(multiclass_pos.size(0), -1)
        label_emb = self.output_transform(label_emb)

        ret = {
            'loss': masked_lm_loss,
            'label_emb': label_emb,
            'logits': multiclass_logits,
            'hidden_states': None,
            'attentions': None,
        }
        return ret

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, input_ids, depth2label, token_type_ids=None, predict=None, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask, predict=predict, token_type_ids=token_type_ids)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                for l in depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
        return predict_labels, prediction_scores
