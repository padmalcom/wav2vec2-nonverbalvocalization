from transformers.models.wav2vec2.modeling_wav2vec2 import (
	Wav2Vec2PreTrainedModel,
	Wav2Vec2Model
)
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
			outputs = torch.mean(hidden_states, dim=1)
		elif mode == "sum":
			outputs = torch.sum(hidden_states, dim=1)
		elif mode == "max":
			outputs = torch.max(hidden_states, dim=1)[0]
		else:
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
		print(1)
		return_dict = return_dict if return_dict is not None else self.config.use_return_dict
		outputs = self.wav2vec2(
			input_values,
			attention_mask=attention_mask,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=return_dict,
		)
		hidden_states = outputs[0]
		hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
		print(2)
		
		# split hidden_state by task. what datatype is it? what dimension is expected?
		hidden_states_age = []
		hidden_states_vocal = []
		labels_age = []
		labels_vocal = []
		#for hs in hidden_states:
		#	print("current hidden state:", hs)
		#	(hidden_states_vocal, hidden_states_age)[hs[0] == 0].append(hs)
		#	(labels_vocal, labels_age)[hs[1] == 0].append(hs)
		if not len(task) == len(hidden_states):
			print("Length of tasks and hiddenstates differ.")
		for idx, t in enumerate(task):
			if t == 0:
				hidden_states_vocal.append(hidden_states[idx])
			elif t == 1:
				hidden_states_age.append(hidden_states[idx])
			else:
				print("Unknown task id: ", t)
			
		print("Hiddenstates 1:", hidden_states_age, "hiddenstates 2:", hidden_states_vocal)
				
		logits_age = self.classifier_age(torch.tensor(hidden_states_age))
		logits_vocal = self.classifier_vocalization(torch.tensor(hidden_states_vocal))
		print(3)
		loss = None
		if labels_age and labels_vocal is not None:
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
				loss = loss_fct(logits, labels)

		if not return_dict:
			output = (logits,) + outputs[2:]
			return ((loss,) + output) if loss is not None else output

		return SpeechClassifierOutput(
			loss=loss,
			logits=logits,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)