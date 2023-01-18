from transformers.models.wav2vec2.modeling_wav2vec2 import (
	Wav2Vec2PreTrainedModel,
	Wav2Vec2Model
)
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Dict, List, Optional, Union, Tuple, Any
from Wav2Vec2ClassificationHead import Wav2Vec2ClassificationHead

@dataclass
class SpeechClassifierOutput(ModelOutput):
	loss: Optional[torch.FloatTensor] = None
	logits: torch.FloatTensor = None
	hidden_states: Optional[Tuple[torch.FloatTensor]] = None
	attentions: Optional[Tuple[torch.FloatTensor]] = None
	
class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.num_labels = config.num_labels
		self.pooling_mode = config.pooling_mode
		self.config = config

		self.wav2vec2 = Wav2Vec2Model(config)
		
		# we can pass the same config since the list of labels does not play a role in the heads
		self.classifier_age = Wav2Vec2ClassificationHead(config)
		self.classifier_vocalization = Wav2Vec2ClassificationHead(config)

		self.init_weights()

	def freeze_feature_extractor(self):
		self.wav2vec2.feature_extractor._freeze_parameters()

	def merged_strategy(
			self,
			hidden_states,
			mode="mean"
	):
		if mode == "mean":
			print("Mode mean")
			outputs = torch.mean(hidden_states, dim=1)
		elif mode == "sum":
			print("Mode sum - can be deleted?")
			outputs = torch.sum(hidden_states, dim=1)
		elif mode == "max":
			print("Mode max - can be deleted?")
			outputs = torch.max(hidden_states, dim=1)[0]
		else:
			print("Exception never occurs - ups...")
			raise Exception(
				"The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

		return outputs

	def forward(
			self,
			input_values,
			attention_mask=None,
			output_attentions=None,
			output_hidden_states=None,
			return_dict=None,
			labels=None,
			task=None
	):
		# task 0 = voc, task 1 = age!
		print("1 - return_dict is: ", return_dict)
		print("labels: ", labels)
		print("tasks", task)
		print("output_hidden_states:", output_hidden_states)
		print("output_attentions:", output_attentions)
		print("attention_mask:", attention_mask)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		print("Classifier input values:", input_values.size())
		outputs = self.wav2vec2(
			input_values,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		print("Outputs:", outputs.__dict__.keys(), "object:", outputs)
		print("len last hidden state:", outputs.last_hidden_state.size())
		print("len extract_features:", outputs.extract_features.size())
		if outputs.hidden_states:
			print("len hidden states:", len(outputs.hidden_states))
		if outputs.attentions:
			print("len attentions:", len(outputs.attentions))
			
		print("Outputs:", outputs.__dict__.keys())
		hidden_states = outputs[0]
		print("Hidden states size (first):", hidden_states.size())
		hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
		print(2)
		print("Hidden states size (second):", hidden_states.size())
		
		# split hidden_state by task. what datatype is it? what dimension is expected?
		hidden_states_age = hidden_states.clone().detach()
		hidden_states_vocal = hidden_states.clone().detach()
		labels_age = labels.clone().detach()
		labels_vocal = labels.clone().detach()

		print("Hiddenstates 1 (before):", hidden_states_age.size(), "hiddenstates 2 (before):", hidden_states_vocal.size())

		for idx, t in enumerate(task):
			# todo: check ids
			if t == 1:
				hidden_states_vocal = torch.cat([hidden_states_vocal[0:idx], hidden_states_vocal[idx+1:]])
				labels_vocal = torch.cat([labels_vocal[0:idx], labels_vocal[idx+1:]])
			elif t == 0:
				hidden_states_age = torch.cat([hidden_states_age[0:idx], hidden_states_age[idx+1:]])
				labels_age = torch.cat([labels_age[0:idx], labels_age[idx+1:]])
			
		print("Hiddenstates 1 (after):", hidden_states_age.size(), "hiddenstates 2 (after):", hidden_states_vocal.size())
				
		logits_age = self.classifier_age(torch.tensor(hidden_states_age))
		logits_vocal = self.classifier_vocalization(torch.tensor(hidden_states_vocal))
		print("logits_age: ", logits_age.size(), "logits_vocal: ", logits_vocal.size())
		loss = None
		if labels_age is not None and labels_vocal is not None:
			if self.config.problem_type is None:
				if self.num_labels == 1:
					self.config.problem_type = "regression"
				elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
					self.config.problem_type = "single_label_classification"
				else:
					self.config.problem_type = "multi_label_classification"

			if self.config.problem_type == "regression":
				loss_fct = MSELoss()
				#loss = loss_fct(logits.view(-1, self.num_labels), labels)
				loss_age = loss_fct(logits_age.view(-1, self.num_labels_age), labels_age.view(-1))
				loss_vocalization = loss_fct(logits_vocal.view(-1, self.num_labels_vocalization), labels_vocal.view(-1))
				loss = (loss_age * 0.5)  + (loss_vocalization * 0.5)
			elif self.config.problem_type == "single_label_classification":
				loss_fct = CrossEntropyLoss()
				loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
			elif self.config.problem_type == "multi_label_classification":
				loss_fct = BCEWithLogitsLoss()
				#loss = loss_fct(logits, labels)
				
				loss_age = loss_fct(logits_age.view(-1, len(labels_age)), labels_age.view(-1)) # todo: len(labels_age) might be wrong
				loss_vocalization = loss_fct(logits_vocal.view(-1, len(labels_voc)), labels_vocal.view(-1))
				loss = (loss_age * 0.5)  + (loss_vocalization * 0.5)

		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		#return 1
		return SpeechClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)