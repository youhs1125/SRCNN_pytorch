import cv2
import numpy as np
import matplotlib.pyplot as plt

def comparePSNR(origins, bicubic, preds1, preds2=None, preds3 = None):
    # compare predicts with bicubic

    mPSNR = 0
    bPSNR = 0
    for i in range(len(preds1)):
        pred_num = preds1[i].reshape(preds1[i].shape[1],preds1[i].shape[2],preds1[i].shape[0])
        pred_num = pred_num*255

        testi = origins[i]*255
        bicubici = bicubic[i]*255

        predPSNR = cv2.PSNR(pred_num, testi)
        bicubicPSNR = cv2.PSNR(bicubici, testi)
        # print("MODEL --- PSNR: ", predPSNR)
        # print("BICUBIC - PSNR: ", bicubicPSNR)

        mPSNR += predPSNR
        bPSNR += bicubicPSNR

        if i == 5:
            print(f"PSNR(of images below) SRCNN : {predPSNR} Bicubic Interpolation: {bicubicPSNR}")
            # fig, axes = plt.subplots(1,4, figsize = (20,48))

            pred_num /= 255.0
            bicubici /= 255.0
            testi /= 255.0

            pred_num = np.clip(pred_num, 0.0, 1.0)
            bicubici = np.clip(bicubici, 0.0, 1.0)

            fig, axes = plt.subplots(1, 3, figsize = (20,36))
            axes[0].imshow(testi)
            axes[0].set_title("original")
            axes[1].imshow(bicubici)
            axes[1].set_title("Bicubic Interpolation")
            axes[2].imshow(pred_num)
            axes[2].set_title("SRCNN")
            plt.show()

    print(f"PSNR(avg) SRCNN: {mPSNR / len(preds1)},  Bicubic Interpolation: {bPSNR / len(preds1)}")