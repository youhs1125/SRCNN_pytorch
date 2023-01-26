import torch
import dataset
import loop
import utils

def doTest():

    # get dataloader
    origin, ds = dataset.getTestData()

    # load modelData
    srcnn = torch.load("models/SRCNN_model.pt")
    srcnn.cpu()

    preds = loop.test_loop(srcnn,ds)

    bicubic = loop.getBicubic(ds)

    utils.comparePSNR(origin,bicubic,preds)


if __name__ == "__main__":
    doTest()