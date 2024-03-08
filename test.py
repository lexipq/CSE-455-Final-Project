# testing the saved model
import torch
import random
import pandas as pd
import tkinter as tk
import torch.utils.data
from PIL import Image, ImageTk

import model
# import loader

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

cnn = torch.load("RN18.pt")
cnn.eval()
    
# randomly shuffle the indicies for the images in the csv file
num_of_images = 1250
idxs = [x for x in range(0, num_of_images)]
random.shuffle(idxs)

df = pd.read_csv('test.csv')
ndf = pd.read_csv('names.csv')
count = 0

window = tk.Tk()
window.geometry('500x400')

canvas = tk.Canvas(height=224, width=224)
img_id = canvas.create_image(0, 0, anchor='nw')
canvas.pack()

# random initial image
img_tk = None
label_var = tk.StringVar()
process_image()

label = tk.Label(textvariable=label_var)
label.pack()

button = tk.Button(text='new image', command=process_image)
button.pack()

window.mainloop()

# code to test training accuracy
# data = loader.ImgDataSet("test.csv", transformer=model.basic_transformer)
# dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)

# predictions = []
# actual_labels = []

# for inputs, labels in tqdm.tqdm(dataloader):
#         outputs = cnn(inputs)
#         _, pred = torch.max(outputs, dim=1)
#         actual_labels += labels.view(-1).cpu().numpy().tolist()
#         predictions += pred.view(-1).cpu().numpy().tolist()

# print("actual labels: ", actual_labels)
# print("predicted labels: ", predictions)   
# acc = np.sum(np.array(actual_labels) == np.array(
#         predictions)) / len(actual_labels)
# print("testing accuracy:", (acc * 100))
