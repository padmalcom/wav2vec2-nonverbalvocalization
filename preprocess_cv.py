import os
import json
import csv
import random
import string
from tqdm import tqdm
from pydub import AudioSegment
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer, FSMTForConditionalGeneration, FSMTTokenizer, pipeline
import numpy as np
from scipy.special import softmax
import urllib.request

RAW_DATA_FILE = os.path.join('common-voice-12','validated.tsv')
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

### irony
IRONY_MODEL_NAME = f"cardiffnlp/twitter-roberta-base-irony"
irony_tokenizer = AutoTokenizer.from_pretrained(IRONY_MODEL_NAME)
irony_model = AutoModelForSequenceClassification.from_pretrained(IRONY_MODEL_NAME)

def irony(text):
	labels = ['no irony', 'irony']
	encoded_input = tokenizer(text, return_tensors='pt')
	output = model(**encoded_input)
	scores = output[0][0].detach().numpy()
	scores = softmax(scores)
	ranking = np.argsort(scores)
	ranking = ranking[::-1]
	max_label = 0
	max_score = 0
	for i in range(scores.shape[0]):
		l = labels[ranking[i]]
		s = scores[ranking[i]]
		if s > max_score:
			max_label = ranking[i]
			max_score = s
	return max_label
	
### emotion
EMOTION_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
audio_classifier = pipeline(task="audio-classification", model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


### translation
TRANSLATION_MODEL_NAME = "facebook/wmt19-de-en"
translation_tokenizer = FSMTTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
translation_model = FSMTForConditionalGeneration.from_pretrained(TRANSLATION_MODEL_NAME)

def translate(text):
	input_ids = translation_tokenizer.encode(text, return_tensors="pt")
	outputs = translation_model.generate(input_ids)
	return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

### preparation
def prepare_data():
	data = []
	labels = {}
	with open(RAW_DATA_FILE) as f:
		row_count = sum(1 for line in f)
		print("There are", row_count, "rows in the dataset.")
	
	with open(RAW_DATA_FILE, 'r', encoding="utf8") as f:
		tsv = csv.DictReader(f, delimiter="\t")
		data = []
		
		if not os.path.exists(os.path.join('common-voice-12', "wavs")):
			os.mkdir(os.path.join('common-voice-12', "wavs"))
			
		i = 0
		for line in tqdm(tsv, total=row_count):
			formatted_sample = {}
			formatted_sample['speaker'] = line['client_id']
			formatted_sample['file'] = line['path']
			formatted_sample['sentence'] = line['sentence'].translate(str.maketrans('', '', string.punctuation))
			formatted_sample['age'] = line['age']
			formatted_sample['gender'] = line['gender']
			formatted_sample['language'] = line['locale']
			formatted_sample['file'] = line['path']
			# '0' non irony, '1' irony
			formatted_sample['irony'] = irony(translate(formatted_sample['sentence']))
			
			mp3FullPath = os.path.join('common-voice-12', "clips", line['path'])
			filename, _ = os.path.splitext(os.path.basename(mp3FullPath))
			sound = AudioSegment.from_mp3(mp3FullPath)
			if sound.duration_seconds > 0:
					sound = sound.set_frame_rate(16000)
					sound = sound.set_channels(1)
					wav_path = os.path.join('common-voice-12', "wavs", filename + ".wav")
					sound.export(wav_path, format="wav")
					
					# emotion classification
					preds = audio_classifier(wav_path)
					for p in preds:
						print(p)
					#return [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
					formatted_sample['emotion'] = p["label"]
					# 'neu' => 0, 'ang' => 1, 'hap' => 2, 'sad' => 3
					
					# 
					data.append(formatted_sample)
					i += 1
			if i == 100:
				break
			
		random.shuffle(data)
		print("Found", len(data), "samples. Example: ", data[:5])
		
		train = data[:int(len(data)*0.8)]
		test = data[len(train):]
		
		print("Length train:", len(train), "length test:", len(test))
		
		with open(TRAIN_FILE, 'w', newline='', encoding="utf8") as f: 
			w = csv.DictWriter(f, train[0].keys())
			w.writeheader()
			for t in train:
				w.writerow(t)
				
		with open(TEST_FILE, 'w', newline='', encoding="utf8") as f: 
			w = csv.DictWriter(f, test[0].keys())
			w.writeheader()
			for t in test:
				w.writerow(t)	
	
		
if __name__ == '__main__':
	prepare_data()