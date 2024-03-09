# modified copy of hw5's helper.py
import torch
import torch.nn as nn
import numpy as np
import tqdm


def run(mode, dataloader, model, optimizer=None, use_cuda = True):
    """
    mode: either "train" or "valid". If the mode is train, we will optimize the model
    """
    running_loss = []
    criterion = nn.CrossEntropyLoss()

    actual_labels = []
    predictions = []
    for inputs, labels in tqdm.tqdm(dataloader):
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss.append(loss.item())

        actual_labels += labels.view(-1).cpu().numpy().tolist()
        _, pred = torch.max(outputs, dim=1)

        predictions += pred.view(-1).cpu().numpy().tolist()

        if mode == "train":
            # zero the parameter gradients
            optimizer.zero_grad() # so we don't accumlate loss
            loss.backward() # back propagation
            optimizer.step() # updating weights

    # sums up labels and predictions that equal each other
    # arr == diff_arr returns an array of booleans where True means
    # arr[i] == diff_arr[i] is True or 1
    acc = np.sum(np.array(actual_labels) == np.array(
        predictions)) / len(actual_labels)
    print(mode, "Accuracy:", acc)

    loss = np.mean(running_loss)

    return loss, acc
