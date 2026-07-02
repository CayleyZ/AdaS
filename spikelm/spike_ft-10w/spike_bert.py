# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""


import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# from transformers.modeling_utils 
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bert.configuration_bert import BertConfig


from transformers import AutoConfig
import copy

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01


BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(out_chn), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


import sys
sys.path.append("..")
from spike_bert_core import BertModel, BertEncoder

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    # load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """
    Bert Model with two heads on top as done during the pretraining: a `masked language modeling` head and a `next
    sentence prediction (classification)` head.
    """,
    BERT_START_DOCSTRING,
)
class GBinaryBertForPreTraining(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias", r"cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BertForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.
    """,
    BERT_START_DOCSTRING,
)
class BertForSequenceClassification2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
def get_model(args, num_labels, new_config):
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)

    config.output_hidden_states = True
    config.output_attentions =True
    
    # if not new_config.pretrain_teacher:  ##
    #     new_config.pretrain_teacher = args.model_name_or_path
    # if new_config.online_dist:
    #     teacher =  AutoModelForSequenceClassification.from_pretrained(
    #                             args.model_name_or_path,
    #                             config=config,
    #                             ignore_mismatched_sizes=False,
    #                         )
    # else: 
    #     teacher =  AutoModelForSequenceClassification.from_pretrained(
    #                             '/xxr/glue2/res/' + args.task_name,
    #                             config=config,
    #                             ignore_mismatched_sizes=False,
    #                         )
    ##########################################################

    student_config = copy.deepcopy(config)
    
    student_config.weight_bits = 32
    student_config.input_bits = 2 ##############################################
    student_config.weight_quant_method = 'bwn'
    student_config.input_quant_method = 'elastic'
    student_config.clip_init_val = 2.5
    student_config.learnable_scaling = True
    student_config.sym_quant_qkvo = True
    student_config.sym_quant_ffn_attn = False
    student_config.embed_layerwise = False
    student_config.weight_layerwise = True
    student_config.input_layerwise = True
    student_config.hidden_act = 'relu'
    student_config.not_quantize_attention = False

    student_config.half_binary = False  ## 
    student_config.improve_linear = False  ##
    student_config.improve_kqv = False  ##
    student_config.lora_rank = 1 ##
    student_config.num_hidden_layers = student_config.num_hidden_layers 
    student_config.quantize_act = True
    student_config.clip_val = 1.0
    student_config.T = 4 ######################################
    # student_config.pretrain_student = new_config.pretrain_student  ##

    if new_config.pretrain_student=='new':
        student = BertForSequenceClassification2(config=student_config)
    elif new_config.pretrain_student:
        student = BertForSequenceClassification2.from_pretrained(new_config.pretrain_student, config=student_config)
    # model = GBinaryBertForSequenceClassification(config=student_config)
    return student, None



class DistModel(nn.Module):
    def __init__(self, args, num_labels, new_config):
        super().__init__()

        self.num_labels = num_labels
        self.config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
        self.student, self.teacher =  get_model(args, num_labels, new_config)
        self.temperature = 1.
        self.loss_mse = MSELoss()
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="sum")

        self.A_r = 48
        self.freeze_teacher = 1000
        # self.online_dist = new_config.online_dist   ##
        # self.offline_dist = not new_config.online_dist

        # self.student_loss = new_config.student_loss   ##
        # self.all_rep_loss = new_config.all_rep_loss   ##
        # self.last_rep_loss = new_config.last_rep_loss   ##

        self.no_dist = True
        # if not self.online_dist and not self.offline_dist:
        #     self.no_dist = True
        #     self.teacher = None
        #     self.student_loss = True
        #     self.all_rep_loss = False
        #     self.last_rep_loss = False
        
    def forward(self,batch, epoch_num):
        if self.no_dist or self.offline_dist or (self.online_dist and epoch_num<self.freeze_teacher):
            student_out = self.student(**batch, output_hidden_states=True, output_attentions=True)
            
            # if self.online_dist:
            #     teacher_out = self.teacher(**batch, output_hidden_states=True, output_attentions=True)
            # if self.offline_dist:
            #     with torch.no_grad():
            #         teacher_out = self.teacher(**batch, output_hidden_states=True, output_attentions=True)

            loss = 0.

            # if not self.no_dist:
            # loss += self.soft_cross_entropy(student_out.logits / self.temperature, teacher_out.logits.detach() / self.temperature)

            # if self.online_dist:
                # loss += teacher_out.loss

            ##############################################################################################
            # if self.student_loss:
            loss += student_out.loss

            ##############################################################################################
            # loss += self.get_att_vv_loss(student_out, teacher_out, batch['attention_mask'])

            ##############################################################################################
            # if self.all_rep_loss:
                # loss += self.get_all_rep_loss(student_out, teacher_out)
            # loss += self.get_all_att_mse_loss(student_out, teacher_out)


            ##################################################################
            # if self.last_rep_loss:
            # loss += self.loss_mse(student_out.hidden_states[-1], teacher_out.hidden_states[-1].detach())
            ##################################################################
            # loss += self.get_att_klloss(student_out, teacher_out, batch['attention_mask'])
            # loss += self.get_v_mseloss(student_out, teacher_out)
            ##################################################################
            

            return SequenceClassifierOutput(
                loss=loss,
                logits=student_out.logits,
                hidden_states=student_out.hidden_states,
                attentions=student_out.attentions,
            )
        # else:
        #     student_out = self.student(**batch, output_hidden_states=True, output_attentions=True)
            
        #     if self.online_dist:
        #         teacher_out = self.teacher(**batch, output_hidden_states=True, output_attentions=True)
        #     if self.offline_dist:
        #         with torch.no_grad():
        #             teacher_out = self.teacher(**batch, output_hidden_states=True, output_attentions=True)

        #     loss = 0.

        #     if not self.no_dist:
        #         loss += self.soft_cross_entropy(student_out.logits / self.temperature, teacher_out.logits.detach() / self.temperature)

        #     if self.online_dist:
        #         loss += teacher_out.loss

        #     ##############################################################################################
        #     if self.student_loss:
        #         loss += student_out.loss

        #     ##############################################################################################
        #     # loss += self.get_att_vv_loss(student_out, teacher_out, batch['attention_mask'])

        #     ##############################################################################################
        #     if self.all_rep_loss:
        #         loss += self.get_all_rep_loss(student_out, teacher_out)
        #     # loss += self.get_all_att_mse_loss(student_out, teacher_out)


        #     ##################################################################
        #     if self.last_rep_loss:
        #         loss += self.loss_mse(student_out.hidden_states[-1], teacher_out.hidden_states[-1].detach())
        #     ##################################################################
        #     # loss += self.get_att_klloss(student_out, teacher_out, batch['attention_mask'])
        #     # loss += self.get_v_mseloss(student_out, teacher_out)
        #     ##################################################################
            

        #     return SequenceClassifierOutput(
        #         loss=loss,
        #         logits=student_out.logits,
        #         hidden_states=student_out.hidden_states,
        #         attentions=student_out.attentions,
        #     )
        
    def get_all_rep_loss(self, student_out, teacher_out):
        rep_loss = 0.
        rep_loss_layerwise = []
        for student_rep, teacher_rep in zip(student_out.hidden_states, teacher_out.hidden_states):
            tmp_loss = self.loss_mse(student_rep, teacher_rep.detach())
            rep_loss += tmp_loss
            rep_loss_layerwise.append(tmp_loss.item())
        
        loss = rep_loss / len(rep_loss_layerwise)
        return loss
    
    def get_all_att_mse_loss(self, student_out, teacher_out):
        rep_loss = 0.
        rep_loss_layerwise = []
        for s_att, t_att in zip(student_out.attentions, teacher_out.attentions):
            s_att = torch.where(s_att <= -1e2, torch.zeros_like(s_att).cuda(), s_att)
            t_att = torch.where(t_att <= -1e2, torch.zeros_like(t_att).cuda(), t_att)
            tmp_loss = self.loss_mse(s_att, t_att.detach())
            rep_loss += tmp_loss
            rep_loss_layerwise.append(tmp_loss.item())
        
        loss = rep_loss / len(rep_loss_layerwise)
        return loss

    def soft_cross_entropy(self, predicts, targets):
        student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
        targets_prob = torch.nn.functional.softmax(targets, dim=-1)
        return (- targets_prob * student_likelihood).mean()
    def train(self):
        self.student.train()
        # self.teacher.train()
    def eval(self):
        self.student.eval()
        # self.teacher.eval()
    
    def save_pretrained(self, *args, **kwargs):
        self.student.save_pretrained(*args, **kwargs)

    def get_att_vv_loss(self, student_out, teacher_out, attention_mask):
        s_att = student_out.attentions[-1]
        t_att = teacher_out.attentions[-1]
        s_att = torch.where(s_att <= -1e2, torch.zeros_like(s_att).cuda(), s_att)
        t_att = torch.where(t_att <= -1e2, torch.zeros_like(t_att).cuda(), t_att)

        l_att = self._get_kl_loss(t_att.detach(), s_att, attention_mask)


        s_prev_hidden = student_out.hidden_states[-2]
        t_prev_hidden = teacher_out.hidden_states[-2]

        s_v = self.student.bert.encoder.layer[-1].attention.self.value(s_prev_hidden)
        t_v = self.teacher.bert.encoder.layer[-1].attention.self.value(t_prev_hidden)
        s_r_dim = self.student.config.hidden_size // self.A_r
        t_r_dim = self.teacher.config.hidden_size // self.A_r
        s_v = self._transpose_for_scores_relation(s_v, s_r_dim)
        t_v = self._transpose_for_scores_relation(t_v, t_r_dim)

        s_vv = torch.matmul(s_v, s_v.transpose(-1, -2)) / math.sqrt(s_r_dim)
        t_vv = torch.matmul(t_v, t_v.transpose(-1, -2)) / math.sqrt(t_r_dim)

        l_relation = self._get_kl_loss(t_vv.detach(), s_vv, attention_mask)

        return l_att + l_relation
    
    def get_att_klloss(self, student_out, teacher_out, attention_mask):
        s_att = student_out.attentions[-1]
        t_att = teacher_out.attentions[-1]
        # s_att = torch.where(s_att <= -1e2, torch.zeros_like(s_att).cuda(), s_att)
        # t_att = torch.where(t_att <= -1e2, torch.zeros_like(t_att).cuda(), t_att)

        loss = self._get_kl_loss(t_att.detach(), s_att, attention_mask)

        return loss
    
    def get_att_mseloss(self, student_out, teacher_out):
        s_att = student_out.attentions[-1]
        t_att = teacher_out.attentions[-1]
        s_att = torch.where(s_att <= -1e2, torch.zeros_like(s_att).cuda(), s_att)
        t_att = torch.where(t_att <= -1e2, torch.zeros_like(t_att).cuda(), t_att)

        # l_att = self._get_kl_loss(t_att.detach(), s_att, attention_mask)
        loss = self.loss_mse(s_att, t_att.detach())

        return loss
    
    def get_v_mseloss(self, student_out, teacher_out):

        s_prev_hidden = student_out.hidden_states[-2]
        t_prev_hidden = teacher_out.hidden_states[-2]

        s_v = self.student.bert.encoder.layer[-1].attention.self.value(s_prev_hidden)
        t_v = self.teacher.bert.encoder.layer[-1].attention.self.value(t_prev_hidden)
        
        loss = self.loss_mse(s_v, t_v.detach())

        return loss

    def _transpose_for_scores_relation(self, x: torch.Tensor, relation_head_size: int):
        """Adapted from BertSelfAttention.get_transposed_attns().

        Arguments:
            x (Tensor): a vector (query, key, or value) of shape (batch_size, seq_length, hidden_size)
            relation_head_size (int): relation head size
        Return:
            x_relation (Tensor): a vector (query, key, or value) of shape
                                (batch_size, relation_head_number, seq_length, relation_head_size)
        """
        new_x_shape = [*x.size()[:-1], self.A_r, relation_head_size]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _get_kl_loss(
        self, rel_T: torch.Tensor, rel_S: torch.Tensor, attention_mask: torch.Tensor
    ):
        """Compute KL divergence loss of teacher and student on one relation.

        This function is a vectorized version of formula (6) in the MiniLM paper.
        The paper does not handle batching and attention mask.

        Arguments:
            rel_T: a self attention relation of the teacher (batch_size, A_r, seq_len, seq_len)
            rel_S: a self attention relation of the student (batch_size, A_r, seq_len, seq_len)
            attention_mask: attention mask of a batch of input
        """
        # Note: rel_T is the target and rel_S is the input of KL Div loss for KLDivLoss(), before softmax.
        # KLDivLoss() needs log of inputs (rel_S)
        # Reference:
        # (1) torch source: https://github.com/pytorch/pytorch/blob/7cc029cb75c292e93d168e117e46a681ace02e79/aten/src/ATen/native/Loss.cpp#L71
        # (2) wikipedia: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
        loss = 0.0
        batch_size = attention_mask.shape[0]
        print
        seq_lengths = attention_mask.sum(-1).tolist()
        for b in range(batch_size):
            cur_seq_len = seq_lengths[b]  # current sequence length
            # While we kind of get the same values from output.attentions from BertModel, it seems to do a weird thing by
            # applying dropout post softmax. The paper's calculations do not apply this
            R_L_T = torch.nn.Softmax(dim=-1)(rel_T[b, :, :cur_seq_len, :cur_seq_len])
            R_M_S = torch.nn.functional.log_softmax(
                rel_S[b, :, :cur_seq_len, :cur_seq_len], dim=-1
            )  # KL DIV loss needs log, so do log_softmax
            loss += self.kl_loss_fn(
                R_M_S.reshape(-1, cur_seq_len), R_L_T.reshape(-1, cur_seq_len)
            ) / (
                self.A_r * cur_seq_len
            )  # normalize by relation head num and seq length
        loss /= batch_size  # normalize by batch_size as well
        return loss


