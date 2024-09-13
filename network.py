import torch
import torch.nn as nn

class DummyModel(nn.Module):

	def __init__(self, input_shape=(3, 32, 32), num_classes=10):
		super(DummyModel, self).__init__()
		self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
		self.pool = nn.AdaptiveMaxPool2d((1, 1))
		self.fc = nn.Linear(64, num_classes)

	def __str__(self):
		return self.__class__.__name__

	def forward(self, x):
		x = self.conv1(x)
		x = torch.relu(x)
		x = self.conv2(x)
		x = torch.relu(x)
		x = self.pool(x)
		x = torch.flatten(x, 1)
		x = self.fc(x)
		return x

def get_network(input_shape=(3, 32, 32), num_classes=10, weights_fp=None):
	model = DummyModel(input_shape, num_classes)
	if weights_fp is not None:
		model.load_state_dict(torch.load(weights_fp))
	return model