from tqdm import tqdm
import torch
import numpy as np
import cv2
import utils

# define train_loop

def train_loop(model, dataloader, loss_fn, optimizer,writer, epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    size = len(dataloader.dataset)

    # the last layer's learning rate should be 1e-5
    optimizer.param_groups[-1]["lr"] = 1e-5

    cur_loss = 0.0
    for iter, (X, y) in enumerate(tqdm(dataloader, position=0, leave=True, desc="train")):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_loss += loss.item()

        if iter % 100 == 0:
            train_loss = cur_loss / 100
            num_pred = pred.cpu().detach().numpy()
            num_y = y.cpu().numpy()
            train_PSNR = cv2.PSNR(num_pred*255, num_y*255)

            print(f"\n{iter}-train_loss: {train_loss} PSNR: {train_PSNR}")
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("PSNR/train", train_PSNR, epoch)

            cur_loss = 0


# define validation_loop

def val_loop(model, dataloader,writer, epoch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    val_PSNR = 0.0
    for iter, (X, y) in enumerate(tqdm(dataloader, position=0, leave=True, desc="validation")):
        pred = model(X.to(device))
        num_pred = pred.cpu().numpy()
        num_y = y.numpy()
        val_PSNR += cv2.PSNR(num_pred*255, num_y*255)

    val_PSNR /= len(dataloader)
    print(f"RSNR: {val_PSNR}")

    # writer.add_scalar("PSNR/val", val_PSNR, epoch)

# define test_loop

def test_loop(model, ds):
    pred = []

    model.eval()
    with torch.no_grad():
        for img in ds:
            input = torch.FloatTensor(img)
            pred.append(model(input).numpy())

    return pred

def getBicubic(ds):
    bicubic = []

    for img in ds:
        img = utils.backChannel(img)
        bicubic.append(cv2.resize(img,dsize=(img.shape[1],img.shape[0]),interpolation=cv2.INTER_CUBIC))


    return bicubic