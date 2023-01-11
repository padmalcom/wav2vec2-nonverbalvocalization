import torch
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2Processor

from Wav2Vec2ForSpeechClassification import Wav2Vec2ForSpeechClassification

MY_MODEL = "myrun3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = AutoConfig.from_pretrained(MY_MODEL)
processor = Wav2Vec2Processor.from_pretrained(MY_MODEL)
sampling_rate = processor.feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(MY_MODEL).to(device)

def speech_file_to_array_fn(path, sampling_rate):
	speech_array, _sampling_rate = torchaudio.load(path)
	resampler = torchaudio.transforms.Resample(_sampling_rate)
	speech = resampler(speech_array).squeeze().numpy()
	return speech


def predict(path, sampling_rate):
	speech = speech_file_to_array_fn(path, sampling_rate)
	features = processor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
	
	input_values = features.input_values.to(device)
	attention_mask = features.attention_mask.to(device)

	with torch.no_grad():
		logits = model(input_values, attention_mask=attention_mask).logits

	scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
	outputs = [{"Vocalization": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)]
	return outputs
	
res = predict("test.wav", 16000)
max = max(res, key=lambda x: x['Score'])
print("Expected lip popping:", max)