# testing the saved model
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import tqdm

import model
import loader

cnn = torch.load("SimpleCNN.pt")
cnn.eval()

data = loader.ImgDataSet("test.csv", transformer=model.basic_transformer)
dataloader = torch.utils.data.DataLoader(data, batch_size=1)

criterion = nn.CrossEntropyLoss()
predictions = []
actual_labels = []

for inputs, labels in tqdm.tqdm(dataloader):
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        _, pred = torch.max(outputs, dim=1)
        actual_labels += labels.view(-1).cpu().numpy().tolist()
        predictions += pred.view(-1).cpu().numpy().tolist()

print("predictions: ", predictions)
print("labels: ", actual_labels)
