# src: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb#scrollTo=0Tl6iKAUR4EL

import os
import json
import random
from datasets import load_dataset, Audio
from transformers import (
	AutoConfig,
	Wav2Vec2Processor,
	Wav2Vec2CTCTokenizer,
	EvalPrediction,
	TrainingArguments,
	Trainer,
	is_apex_available
)

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.models.wav2vec2.modeling_wav2vec2 import (
	Wav2Vec2PreTrainedModel,
	Wav2Vec2Model
)

import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any

import transformers
from transformers import Wav2Vec2Processor
from packaging import version

from Wav2Vec2ForSpeechClassification import Wav2Vec2ForSpeechClassification

if is_apex_available():
	from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
	_is_native_amp_available = True
	from torch.cuda.amp import autocast

RAW_DATA_FILE = 'NonverbalVocalization/Nonverbal_Vocalization.json'
TRAIN_FILE = "train.json"
TEST_FILE = "test.json"
AGE_LABELS_FILE = "age_labels.json"
VOC_LABELS_FILE = "voc_labels.json"

INPUT_COLUMN = "path"
OUTPUT_COLUMN_1 = "voc"
OUTPUT_COLUMN_2 = "age"

#model_name_or_path = "facebook/wav2vec2-base" # does not work since not trained on attention mask. see: https://github.com/huggingface/transformers/issues/12934
model_name_or_path = "facebook/wav2vec2-large-960h-lv60-self"
#model_name_or_path = "patrickvonplaten/wav2vec2_tiny_random" # does not work
#model_name_or_path = "hf-internal-testing/tiny-random-wav2vec2-conformer" # does not work
#model_name_or_path = "patrickvonplaten/tiny-wav2vec2-no-tokenizer" # does not work
#model_name_or_path = "patrickvonplaten/wav2vec2_tiny_random_robust" # does not work
#model_name_or_path = "hf-internal-testing/tiny-random-Wav2Vec2ConformerForCTC" # works but bad results

pooling_mode = "mean"
is_regression = False

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
	
class CTCTrainer(Trainer):
	def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
		model.train()
		inputs = self._prepare_inputs(inputs)
		self.use_amp = True

		if self.use_amp:
			with autocast():
				loss = self.compute_loss(model, inputs)
		else:
			loss = self.compute_loss(model, inputs)

		if self.args.gradient_accumulation_steps > 1:
			loss = loss / self.args.gradient_accumulation_steps

		if self.use_amp:
			self.scaler.scale(loss).backward()
		elif self.use_apex:
			with amp.scale_loss(loss, self.optimizer) as scaled_loss:
				scaled_loss.backward()
		elif self.deepspeed:
			self.deepspeed.backward(loss)
		else:
			loss.backward()

		return loss.detach()		
		
@dataclass
class DataCollatorCTCWithPadding:
	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True
	max_length: Optional[int] = None
	max_length_labels: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	pad_to_multiple_of_labels: Optional[int] = None

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		print("Features:", features[1].keys(), "len:", len(features), "type:", type(features), "sample label:", features[0]["label"])
		input_features = [{"input_values": feature["input_values"]} for feature in features]
		
		label1_features = [feature["label"][0] for feature in features] # swap 0 and 1?
		label2_features = [feature["label"][1] for feature in features]
		
		print("Label 1 features:", label1_features, "label 2 features:", label2_features)

		# use the same d_type since both features are int
		d_type = torch.long if isinstance(label1_features[0], int) else torch.float

		# the audio array is the same feature for both labels (age, vocalization)
		features_x2 = input_features + input_features
		batch = self.processor.pad(
			features_x2,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)

		batch["labels"] = torch.tensor(label1_features + label2_features, dtype=d_type)
		
		task_list = len(label1_features) * [0] + len(label2_features) * [1]
		batch["task"] = torch.tensor(task_list, dtype=torch.long)
		
		return batch

def compute_metrics(p: EvalPrediction):
	preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
	preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

	if is_regression:
		return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
	else:
		return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

def prepare_data():
	data = []
	labels_vocalization = {}
	labels_age = {}
	with open(RAW_DATA_FILE, 'r') as f:
		json_data = json.load(f)
		
		for class_name in json_data:
			print("Processing:", class_name)
			for sample in json_data[class_name]:
				current_sample = json_data[class_name][sample]
				
				# add label id <-> name
				labels_vocalization[int(current_sample['label'])] = class_name	
				labels_age[int(current_sample['age'])] = str(current_sample['age'])
				
				# add sample
				formatted_sample = {}
				formatted_sample['wav'] = os.path.join('NonverbalVocalization', class_name, sample)
				formatted_sample['voc'] = current_sample['label']
				formatted_sample['speakerID'] = current_sample['speakerID']
				formatted_sample['age'] = current_sample['age']
				formatted_sample['sex'] = current_sample['sex']
				
				# audio is cast later on
				formatted_sample['audio'] = os.path.join('NonverbalVocalization', class_name, sample)
				
				data.append(formatted_sample)
	random.shuffle(data)
	print("Found", len(data), "samples. Example: ", data[:5])
	
	train = data[:int(len(data)*0.8)]
	test = data[len(train):]
	
	print("Length train:", len(train), "length test:", len(test))
	
	with open(TRAIN_FILE, 'w') as json_file:
		json.dump(train, json_file)
	with open(TEST_FILE, 'w') as json_file:
		json.dump(test, json_file)
	with open(AGE_LABELS_FILE, 'w') as json_file:
		json.dump(labels_age, json_file)
	with open(VOC_LABELS_FILE, 'w') as json_file:
		json.dump(labels_vocalization, json_file)
	

def preprocess_function(examples):
	speech_list = [audio["array"] for audio in examples["audio"]]
	
	labels = ["voc", "age"]
	
	labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
	labels_matrix = np.zeros((len(examples["audio"]), len(labels)))
	
	for idx, label in enumerate(labels):
		labels_matrix[:, idx] = labels_batch[label]

	result = processor(speech_list, sampling_rate=16000)
	
	result["label"] = labels_matrix.tolist()
		
	print("Len labels:", len(result["label"]), "len features:", len(result["input_values"]), "len examples:", len(examples["audio"]))

	return result
	
if __name__ == "__main__":
	if not (os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE) and os.path.exists(AGE_LABELS_FILE) and os.path.exists(VOC_LABELS_FILE)):
		prepare_data()
		
	# load labels
	with open(AGE_LABELS_FILE) as json_file:
		age_labels = json.load(json_file)
	with open(VOC_LABELS_FILE) as json_file:
		voc_labels = json.load(json_file)
		
	# load datasets
	print("Loading datasets...")
	dataset = load_dataset('json', data_files={'train': TRAIN_FILE, 'test': TEST_FILE}).cast_column("audio", Audio())
		
	print(dataset)
	print("There are ", len(voc_labels), "vocalization labels: ", voc_labels)
	print("There are ", len(age_labels), "age labels: ", age_labels)
	
	all_labels = voc_labels | age_labels
	
	# create config
	config = AutoConfig.from_pretrained(
		model_name_or_path,
		num_labels=len(all_labels),
		label2id={all_labels[label]: i for i, label in enumerate(all_labels)},
		id2label={i: all_labels[label] for i, label in enumerate(all_labels)},
		finetuning_task="wav2vec2_clf",
	)
	setattr(config, 'pooling_mode', "mean")
	#config.pooling_mode = "mean"
	print("label2id", config.label2id)
	print("id2label", config.id2label)
	
	processed_dataset = dataset.map(preprocess_function, batch_size=100, batched=True, num_proc=4)
	
	model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path,config=config)
	
	model.freeze_feature_extractor()
	
	training_args = TrainingArguments(output_dir="./wav2vec/",
		per_device_train_batch_size=4,
		per_device_eval_batch_size=4,
		gradient_accumulation_steps=2,
		evaluation_strategy="steps",
		num_train_epochs=20.0,
		#fp16=True,
		fp16=False,
		save_steps=20,
		eval_steps=10,
		logging_steps=10,
		learning_rate=1e-4,
		save_total_limit=2,
	)
	
	data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

	trainer = CTCTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=processed_dataset['train'],
		eval_dataset=processed_dataset['test'],
		tokenizer=processor.feature_extractor
	)
	
	tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name_or_path)
	tokenizer.save_pretrained("myrun3") 

	trainer.train()
	trainer.save_model("myrun3")