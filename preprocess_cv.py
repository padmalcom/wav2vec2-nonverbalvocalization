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
	encoded_input = irony_tokenizer(text, return_tensors='pt')
	output = irony_model(**encoded_input)
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
EMOTION_MODEL_NAME = "padmalcom/wav2vec2-large-emotion-detection-german"
emotions = {'anger':0, 'boredom':1, 'disgust':2, 'fear':3, 'happiness':4, 'sadness':5, 'neutral':6}
audio_classifier = pipeline(task="audio-classification", model=EMOTION_MODEL_NAME)

def emotion(audio_file):
	preds = audio_classifier(audio_file)
	max_score = 0
	max_label = 6
	max_label_text = ""
	for p in preds:
		if p['score'] > max_score and p['score'] > 0.3:
			max_score = p['score']
			max_label = emotions[p["label"]]
			max_label_text = p["label"]
			print("There is an emotional file:", max_label_text)
	return max_label
	
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
		try:
			for line in tqdm(tsv, total=row_count):
				formatted_sample = {}
				formatted_sample['speaker'] = line['client_id']
				formatted_sample['file'] = line['path']
				formatted_sample['sentence'] = line['sentence'].translate(str.maketrans('', '', string.punctuation))
				formatted_sample['age'] = line['age']
				formatted_sample['gender'] = line['gender']
				formatted_sample['language'] = line['locale']
				
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
						formatted_sample['file'] = filename + ".wav"
						
						# emotion classification
						formatted_sample['emotion'] = emotion(wav_path)

						data.append(formatted_sample)
						i += 1
		except KeyboardInterrupt:
			print("Keyboard interrupt called. Writing files and exiting")
		
		random.shuffle(data)
		print("Found", len(data), "samples. Example: ", data[:1])
		
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