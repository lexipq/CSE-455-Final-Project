# testing the saved model
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import tqdm

import model
import loader

cnn = torch.load("RN18.pt")
cnn.eval()

data = loader.ImgDataSet("test.csv", transformer=model.basic_transformer)
dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

predictions = []
actual_labels = []

for inputs, labels in tqdm.tqdm(dataloader):
        outputs = cnn(inputs)
        _, pred = torch.max(outputs, dim=1)
        actual_labels += labels.view(-1).cpu().numpy().tolist()
        predictions += pred.view(-1).cpu().numpy().tolist()

# print("actual labels: ", actual_labels)
# print("predicted labels: ", predictions)   
acc = np.sum(np.array(actual_labels) == np.array(
        predictions)) / len(actual_labels)
print("testing accuracy:", (acc * 100))
