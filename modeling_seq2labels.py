from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModel, BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput

import torch


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    return torch.arange(0, size, dtype=torch.long, device=device)


from dataclasses import dataclass

@dataclass
class Seq2LabelsOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    detect_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    max_error_probability: Optional[torch.FloatTensor] = None


class Seq2LabelsModel(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.num_detect_classes = config.num_detect_classes
        self.label_smoothing = config.label_smoothing

        if config.load_pretrained:
            self.bert = AutoModel.from_pretrained(config.pretrained_name_or_path)
            bert_config = self.bert.config
        else:
            bert_config = AutoConfig.from_pretrained(config.pretrained_name_or_path)
            self.bert = AutoModel.from_config(bert_config)
            
            # PATCH: Fix "Tensor.item() cannot be called on meta tensors"
            # If self.bert is on 'meta' device (due to accelerate/transformers optimization),
            # resize_token_embeddings will fail. We must materialize it.
            try:
                param = next(self.bert.parameters())
                if param.device.type == 'meta':
                    print("Detected meta device. Materializing model to CPU...")
                    self.bert.to_empty(device='cpu')
                    self.bert.to(torch.float32) # Ensure float32 to match new embeddings
                    self.bert.init_weights() # Correct call
                    print("Model materialized and initialized.")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Meta check warning: {e}")

        if config.special_tokens_fix:
            # Ensure embeddings are float32 to prevent "incompatible tensor type" error during resize
            # This is necessary because resize_token_embeddings creates new embeddings that must match existing ones
            if hasattr(self.bert, 'embeddings'):
                self.bert.embeddings.to(torch.float32)
            if hasattr(self.bert, 'word_embedding'):
                self.bert.word_embedding.to(torch.float32)
                
            try:
                vocab_size = self.bert.embeddings.word_embeddings.num_embeddings
            except AttributeError:
                # reserve more space
                vocab_size = self.bert.word_embedding.num_embeddings + 5
            self.bert.resize_token_embeddings(vocab_size + 1)

        predictor_dropout = config.predictor_dropout if config.predictor_dropout is not None else 0.0
        self.dropout = nn.Dropout(predictor_dropout)
        self.classifier = nn.Linear(bert_config.hidden_size, config.vocab_size)
        self.detector = nn.Linear(bert_config.hidden_size, config.num_detect_classes)
        
        if self.classifier.weight.device.type == 'meta':
            # Materializing classifier/detector on CPU to support weight loading
            self.classifier.to_empty(device='cpu')
            self.detector.to_empty(device='cpu')

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_offsets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        d_tags: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2LabelsOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
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

        sequence_output = outputs[0]

        if input_offsets is not None:
            # offsets is (batch_size, d1, ..., dn, orig_sequence_length)
            range_vector = get_range_vector(input_offsets.size(0), device=sequence_output.device).unsqueeze(1)
            # selected embeddings is also (batch_size * d1 * ... * dn, orig_sequence_length)
            sequence_output = sequence_output[range_vector, input_offsets]

        logits = self.classifier(self.dropout(sequence_output))
        logits_d = self.detector(sequence_output)

        loss = None
        if labels is not None and d_tags is not None:
            loss_labels_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing)
            loss_d_fct = CrossEntropyLoss()
            loss_labels = loss_labels_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss_d = loss_d_fct(logits_d.view(-1, self.num_detect_classes), d_tags.view(-1))
            loss = loss_labels + loss_d

        if not return_dict:
            output = (logits, logits_d) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return Seq2LabelsOutput(
            loss=loss,
            logits=logits,
            detect_logits=logits_d,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            max_error_probability=torch.ones(logits.size(0), device=logits.device),
        )
