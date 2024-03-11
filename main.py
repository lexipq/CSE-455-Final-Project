# code to test and train the model
# edited copy of hw5's main.py
import argparse

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt

import loader
import helper
import model

# %%
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ResNet Final Project')
    parser.add_argument(
        '-m',
        '--model',
        default='SimpleCNN',
        help='The Model you want to use')
    parser.add_argument('-t', '--transform', default='basic',
                        help='How do you want to transform your input data?')
    parser.add_argument(
        '-d',
        '--dropout',
        default=False,
        type=str2bool,
        help='Do you want to use dropout during training phase? (True or False)')
    parser.add_argument(
        '-b',
        '--batch-size',
        default=64,
        type=int,
        help='mini-batch size')
    parser.add_argument(
        '-e',
        '--epoch',
        default=10,
        type=int,
        help='Number of Epoches')
    
    parser.add_argument("-c", "--use-cuda", type=str2bool, default=False)
    
    args = parser.parse_args()
    use_mps = False
    
    torch.manual_seed(1)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(1)
    
    if args.model == "RN18":
        net_model = model.ResNet_18()
    elif args.model == "RN18-GPU":
        use_mps = True
        # if we want to use the GPU it must be available
        if torch.backends.mps.is_available():   
            net_model = model.ResNet_18(True)
        else:
            print("error: your gpu is not available for use")
            exit(1)
    elif args.model == "RN34":
        net_model = model.ResNet_34()
    elif args.model == "SimpleCNN":
        net_model = model.SimpleCNN()
    elif args.model == "SimpleCNN-GPU":
        use_mps = True
        if torch.backends.mps.is_available():   
            net_model = model.SimpleCNN(True)
        else:
            print("error: your gpu is not available for use")
            exit(1)
        
    num_params = sum(p.numel() for p in net_model.parameters() if p.requires_grad)
    print("There are", num_params, "parameters in this model") 
    
    print("Use %s transformer for training" % args.transform)
    if args.transform == "basic":
        train_transform = valid_transform = model.basic_transformer
    elif args.transform == "aug":
        train_transform = model.aug_transformer
        valid_transform = model.basic_transformer
    
    trainloader, validloader = loader.get_data_loader(train_transform, valid_transform, args.batch_size)
    
    use_cuda = torch.cuda.is_available() and args.use_cuda
    
    if use_cuda:
        net_model = net_model.cuda()
    
    if use_mps:
        mps_device = torch.device("mps")
        net_model = net_model.to(mps_device)
    
    # %%
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        print(f'epoch: {epoch + 1} / {args.epoch}')
        learning_rate = 0.01 * 0.8 ** epoch
        learning_rate = max(learning_rate, 1e-6)
        optimizer = optim.SGD(net_model.parameters(), lr=learning_rate, momentum=0.9)
    
        loss, acc = helper.run("train", trainloader, net_model, optimizer, use_cuda=use_cuda, use_mps=use_mps)
        train_losses.append(loss)
        train_accs.append(acc)
        with torch.no_grad():
            loss, acc = helper.run("valid", validloader, net_model, use_cuda=use_cuda, use_mps=use_mps)
            valid_losses.append(loss)
            valid_accs.append(acc)
    
    print("-"*60)
    print("best validation accuracy is %.4f percent" % (np.max(valid_accs) * 100) )
    
    torch.save(net_model, "models/%s.pt" % args.model)  # save the model for future reference

    # plots two figures with loss and accuracy (still not working)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.plot(train_losses)
    ax1.plot(valid_losses)
    ax1.legend(["training loss", "validation loss"])

    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.plot(train_accs)
    ax2.plot(valid_accs)
    ax2.legend(["training accuracy", "validation accuracy"])

    plt.show()
