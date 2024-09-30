import flwr as fl
import torch
import collections
import torchmetrics
import torch.optim as optim 

class Client(fl.client.NumPyClient):

	def __init__(self, dataset, cid, model_loader, data_loader, device='cuda'):
		self.cid = cid
		self.data, self.num_classes, self.num_samples = data_loader()
		self.input_shape = self.get_dataset_config(dataset)
		self.model_loader = model_loader
		self.device = device

	def set_parameters(self, parameters, config):
		if not hasattr(self, 'model'):
			self.model = self.model_loader(input_shape=self.input_shape, num_classes=self.num_classes).to(self.device)
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

	def get_parameters(self, config={}):
		return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

	def fit(self, parameters, config):
		self.set_parameters(parameters, config)
		# SGD optimizer
		lr = config.get('lr', 1e-3) 
		optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
		# Adam optimizer
		#optimizer = torch.optim.Adam(self.model.parameters(), lr=config['lr'])
		h = __class__.train(ds=self.data, model=self.model, epochs=config['epochs'], optimizer=optimizer, num_classes=self.num_classes)
		return self.get_parameters(), self.num_samples, h

	def evaluate(self, parameters, config):
		raise NotImplementedError('Client-side evaluation is not implemented!')
	
	def get_dataset_config(self, dataset):
		if dataset.lower() == "cifar10" or "svhn":
			input_shape=(3, 32, 32)
		elif dataset.lower() == "pathmnist" or "dermamnist":
			input_shape=(1, 28, 28)
		else:
			raise NotImplementedError(f"Dataset '{dataset}' is not supported.")
		return input_shape

	@staticmethod
	def train(ds, model, epochs, optimizer, num_classes, metrics=None, loss=torch.nn.CrossEntropyLoss(), verbose=False):
		device = next(model.parameters()).device
		if metrics is None:
			metrics = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
		loss_scores = []
		model.train()
		for epoch in range(epochs):
			train_loss = 0.0
			for _, (x, y) in enumerate(ds):
				x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True).long()
				optimizer.zero_grad()
				preds = model(x)
				_loss = loss(preds, y)
				_loss.backward()
				optimizer.step()
				train_loss += _loss.item()
				metrics(preds.max(1)[-1], y)
			train_loss /= len(ds)
			loss_scores.append(train_loss)
			acc = metrics.compute()
			if verbose:
				print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Accuracy: {100. * acc:.2f}%")
		return {'loss': loss_scores, 'accuracy': acc}
