from torch import nn
import torch

class Wav2Vec2ClassificationHead(nn.Module):
	"""Head for wav2vec classification task."""

	def __init__(self, hidden_size, final_dropout, num_labels):
		super().__init__()
		self.dense = nn.Linear(hidden_size, hidden_size)
		self.dropout = nn.Dropout(final_dropout)
		self.out_proj = nn.Linear(hidden_size, num_labels)

	def forward(self, features, **kwargs):
		x = features
		x = self.dropout(x)
		x = self.dense(x)
		x = torch.tanh(x)
		x = self.dropout(x)
		x = self.out_proj(x)
		return x