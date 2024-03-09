# testing the saved model
import tqdm
import torch
import random
import numpy as np
import pandas as pd
import tkinter as tk
import torch.utils.data
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

import model
import loader

def process_image():
    global img_tk
    global count

    # random image to process
    idx = idxs[count]
    # note: unsafe access if count >= num_of_images
    count += 1

    path = df.loc[idx, 'path']
    label = df.loc[idx, 'label']
    actual_name = df.loc[idx, 'class']

    # set the image we're displaying
    img = Image.open(path)
    img_tk = ImageTk.PhotoImage(img)
    canvas.itemconfig(img_id,image=img_tk)

    # turn the image into a 4D tensor
    input = model.basic_transformer(img)
    input = torch.unsqueeze(input, 0)

    # pass it through the network
    outputs = cnn(input)
    _, pred = torch.max(outputs, dim=1)
    predicted = pred.view(-1).cpu().numpy().tolist()[0]
    val = f"""
    actual label: {label}
    name: {actual_name}

    predicted label: {predicted}
    name: {ndf.loc[predicted,'class']}"""
    label_var.set(value=val)

cnn = torch.load("models/RN18.pt")
cnn.eval()

# randomly shuffle the indicies for the images in the csv file
num_of_images = 1250
idxs = [x for x in range(0, num_of_images)]
random.shuffle(idxs)

df = pd.read_csv('csv_files/test.csv')
ndf = pd.read_csv('csv_files/names.csv')
count = 0

window = tk.Tk()
window.geometry('500x400')

# image to display, none for now
img_tk = None
canvas = tk.Canvas(height=224, width=224)
img_id = canvas.create_image(0, 0, anchor='nw')
canvas.pack()

label_var = tk.StringVar()
label = tk.Label(textvariable=label_var)
label.pack()

button = tk.Button(text='new image', command=process_image)
button.pack()

# update images and labels for first prediction
process_image()

window.mainloop()

# code to test training accuracy
data = loader.ImgDataSet("csv_files/test.csv", transformer=model.basic_transformer)
dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)

predictions = []
actual_labels = []

for inputs, labels in tqdm.tqdm(dataloader):
        outputs = cnn(inputs)
        _, pred = torch.max(outputs, dim=1)
        actual_labels += labels.view(-1).cpu().numpy().tolist()
        predictions += pred.view(-1).cpu().numpy().tolist()

acc = np.sum(np.array(actual_labels) == np.array(
        predictions)) / len(actual_labels)
print("testing accuracy:", (acc * 100))

# code for plotting visualization window (still work in progress)
plt.xlabel('predicted label')
plt.ylabel('actual label')
plt.title(f"testing accuracy: {(acc * 100)}%")
plt.scatter(predictions, actual_labels)
plt.show()
