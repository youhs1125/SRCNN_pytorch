from SRCNN_model import SRCNN
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import dataset
import loop
import matplotlib.pyplot as plt
import numpy as np

def doTrain():
    # get dataloader
    train_dataloader, val_dataloader = dataset.getDataset()

    # use GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # define model
    model = SRCNN().to(device)

    # define loss function
    loss_fn = nn.MSELoss()

    # define optimizer
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # define tensorboard Logger
    writer = SummaryWriter()

    # define epoch
    max_epoch = 300

    for epoch in range(1, max_epoch + 1):
        # train
        print(f"EPOCH: {epoch} \n\n")
        model.train()
        loop.train_loop(model, train_dataloader, loss_fn, optimizer, writer, epoch)

        # valid
        model.eval()
        with torch.no_grad():
            loop.val_loop(model, val_dataloader, writer, epoch)

    writer.close()
    torch.save(model.state_dict(),"models/SRCNN_model_state_dict.pt")
    torch.save(model,"models/SRCNN_model.pt")

if __name__ == "__main__":
    doTrain()