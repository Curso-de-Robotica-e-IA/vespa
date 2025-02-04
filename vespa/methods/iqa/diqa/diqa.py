import math
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from pathlib import Path

from PIL import Image
from skimage import feature

from vespa.methods.iqa.diqa.model.diqa_model import DIQAModel


class DIQA:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__device = torch.device(device)

        self.__model = DIQAModel()
        self.__attached_model = self.__model.to(self.__device)
        self.checkpoint_path = Path(r"\\192.168.155.240\Robotica\Vespa\weights\iqa\diqa")

    def load(self,
             model_path: str=r"\\192.168.155.240\Robotica\Vespa\weights\iqa\diqa\pretrained\modelo_DIQA75b64.pth")\
            -> None:
        self.__model.load_state_dict(torch.load(model_path, weights_only=True))
        self.__attached_model = self.__model.to(self.__device)

    def predict(self, image_path: str) -> float:
        image = self.__load_images(image_path)
        prediction = None

        with torch.no_grad():
            target_image = image.to(self.__device)
            self.__attached_model.eval()
            output = self.__attached_model(target_image)
            prediction = output.item()

        prediction = 1 if prediction > 1 else prediction
        return prediction

    def find_issues(self, std_score, image_path):
        dark, bright, blur, edge, haze = self.__fix_score(
            image_path,
            std_score,
        )
        msg = "Not found"

        if dark or bright or blur or edge or haze:
            if dark:
                msg = "Dark image or with many dark regions"
            if bright:
                msg = "Bright image or with many bright areas"
            if blur or haze:
                msg = "Blurred or Hazed image"
            if edge:
                msg = "Edgy image or many fine details"

        return std_score, msg

    # internal function
    def __load_images(self, image_path: str):
        target_image = Image.open(image_path)

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        target_image = transform(target_image)
        target_image = target_image.unsqueeze(0)

        return target_image

    def __fix_score(self, image_path, score):
        th_Mean_Blur = 97
        low_score = 0.5
        high_score = 0.7
        th_Desvio_Haze = 45
        Mean_Haze = 5
        Median_Haze = 5
        Desvio_Blur = 30
        Mean_Bright = 160
        im_original = cv.imread(image_path)
        im = cv.cvtColor(im_original, cv.COLOR_BGR2GRAY)

        lin, col = im.shape  # Number of lines and columns of images

        NL = im.max() + 1  # Maximum grey level in the image

        # Histogram of probabilities
        N = lin * col
        hist = np.zeros(NL, int)
        Gray_vector = np.arange(1, NL + 1)
        for i in range(lin):
            for j in range(col):
                valor = im[i][j]
                hist[valor] = hist[valor] + 1
        fim1 = NL - 100
        fim2 = fim1 + 20
        soma1 = sum(hist[0:20]) * 100 / N
        soma2 = sum(hist[21:40]) * 100 / N
        soma3 = sum(hist[41:fim1]) * 100 / N
        soma4 = sum(hist[fim1:fim2]) * 100 / N
        soma5 = sum(hist[fim2:NL]) * 100 / N
        Prob = hist / N
        Mean = sum(Prob * Gray_vector)
        Median = np.median(Gray_vector)
        Variance = sum(Prob * (Gray_vector - Mean) ** 2)
        Desvio = np.sqrt(Variance)

        # ========================================

        # Detection of Dark Images

        # Mudança 1 vem aqui

        th_Dark1 = math.ceil(0.11 * NL)  # 30; -> Vira 11% de NL;
        th_Dark2 = math.ceil(0.23 * NL)  # 60; -> 23% de NL
        th_Dark3 = math.ceil(0.27 * NL)  # 70; -> 27% de NL
        th_Dark4 = math.ceil(0.47 * NL)  # 120; -> 47% de NL
        if NL < 150:
            soma = (soma1 + soma2 + soma3 + soma4) / 2
        else:
            soma = soma1 + soma2

        dark = 0  # Flag

        # Mudança 2
        if (
            ((Mean < th_Dark1) or (Mean + Desvio < th_Dark2))
            or (((Mean > th_Dark1) and (Mean < th_Dark3)) or (Mean + Desvio < th_Dark4))  # noqa
            or (NL < 175)
        ):
            if soma >= th_Dark1:
                if score > low_score:
                    dark = 1

        # ========================================

        haze = 0
        # Haze detection
        if not dark:
            # Mudança 3 no if abaixo
            # Evita erros e atua apenas nas hazed mesmo
            if Desvio < th_Desvio_Haze and (NL < 250):
                # Esse código todo do WhitePatch vai precisar no blur também!
                # Mesmo assim, é melhor repetir
                # porque pode não precisar rodar nos dois
                WP = self.__white_patch2(im_original)
                im_original_grey = cv.cvtColor(im_original, cv.COLOR_BGR2GRAY)

                lin2, col2 = im_original_grey.shape
                im_dif_WP = cv.cvtColor(WP, cv.COLOR_BGR2GRAY)

                im_dif2 = np.absolute(im_dif_WP - im_original_grey)
                Mean2, Median2, Variance2, Desvio2 = self.__measures(
                    im_dif2,
                    lin2,
                    col2,
                )  # No final

                if Mean2 != 1 and Median2 != 1:
                    if Mean2 < Mean_Haze and Median2 <= Median_Haze:
                        haze = 1

        blur = 0
        edge = 0
        if not dark and not haze and (Mean > th_Mean_Blur):
            desvio2 = self.__detect_Blur(im, lin, col)

            # Mudança 4 vem aqui
            WP = self.__white_patch2(im_original)
            im_original_grey = cv.cvtColor(im_original, cv.COLOR_BGR2GRAY)

            lin2, col2 = im_original_grey.shape
            im_dif_WP = cv.cvtColor(WP, cv.COLOR_BGR2GRAY)

            im_dif2 = np.absolute(im_dif_WP - im_original_grey)
            Mean2, Median2, Variance2, Desvio2 = self.__measures(
                im_dif2,
                lin2,
                col2,
            )
            # Mudança 5
            # Tudo estará dentro desse if agora; vou indicar o final dele
            if (
                (abs(desvio2 - Desvio2) < 2)
                or (Desvio2 == 0)
                or (desvio2 > Desvio_Blur)
                or ((desvio2 < Desvio_Blur / 3))
            ):
                # Mudança 6: uma nova análise entra aqui.
                #  É um if completo independente do próximo
                if (desvio2 > 10) and (desvio2 < 15):
                    if score > high_score:
                        blur = 1
                # Acrescentei o not blur
                if desvio2 > Desvio_Blur and not blur:
                    edge = 1
                else:
                    # Modifiquei em 07/11
                    if (desvio2 < Desvio_Blur / 3) and not blur:
                        blur = 1

        # Detection of bright images
        bright = 0
        if not dark and not haze and not blur:
            if NL > 250:
                if Mean > Mean_Bright:
                    if (
                        soma5 > 20
                        and soma5 > soma1
                        and abs(soma5 - soma3) > 8
                        and abs(soma5 - soma4) > 30
                        and (Desvio > 50)
                    ):  # (soma5 > 7)
                        # Scores mais baixos já indicam imagens muitos ruins
                        if score > low_score and score < (high_score + 0.1):
                            bright = 1
                        elif score >= (high_score + 0.1):
                            bright = 1

        return dark, bright, blur, edge, haze

    def __white_patch2(self, I):  # noqa
        B, G, R = cv.split(I)
        Kr = 255 / (R.max())
        Kg = 255 / (G.max())
        Kb = 255 / (B.max())
        R = Kr * R
        G = Kg * G
        B = Kb * B
        OUT = cv.merge([B, G, R]).astype(np.uint8)
        return OUT

    def __measures(self, im_dif, lin, col):
        NL = im_dif.max() + 1
        N = lin * col
        hist = np.zeros(NL, int)
        Gray_vector = np.arange(1, NL + 1)

        for i in range(lin):
            for j in range(col):
                valor = im_dif[i][j]
                hist[valor] = hist[valor] + 1

        Prob = hist / N
        Mean = sum(Prob * Gray_vector)
        Median = np.median(Gray_vector)
        Variance = sum(Prob * (Gray_vector - Mean) ** 2)
        Desvio = np.sqrt(Variance)
        return Mean, Median, Variance, Desvio

    def __detect_Blur(self, im, lin, col):
        # Blur detection

        sigma = 19
        kernel = 2 * (2 * sigma) + 1
        im2 = cv.GaussianBlur(im, ksize=(kernel, kernel),
                              sigmaX=sigma, borderType=cv.BORDER_REPLICATE)
        im_dif = np.absolute(im.astype(np.int32) - im2).astype(np.uint8)
        Mean, Median, Variance, Desvio = self.__measures(im_dif, lin, col)

        # Define um mínimo achei zero na imagem da garrafa roxa de Sofia.
        if Desvio == 0:
            Desvio = 0.1
        return Desvio

    def __blur_macro(self, nome, score):
        # Function to detect blur in first plane in Macro photos
        # OBS: Se tudo estiver blurred, já passou pelo teste de blur
        # Este teste deve vir depois dos testes normais, apenas para Macro
        score0 = score
        im = cv.imread(nome)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        lin, col = im.shape
        kernel = 2 * (2 * 23) + 1
        x_center = round(lin / 2)
        y_center = round(col / 2)

        # Working just with vertical images
        # If not vertical, rotate.
        if col > lin:
            tmp = lin
            lin = col
            col = tmp
            tmp = x_center
            x_center = y_center
            y_center = tmp

        raio = round(x_center / 2)
        square_size = round(raio * np.sqrt(2) / 3)

        method = "canny"
        th1 = 0.1
        th2 = 0.3
        sigma = 2

        # Center ROI - Region of Interest

        roi = im[
            (x_center - square_size) : (x_center + square_size),
            (y_center - square_size) : (y_center + square_size),
        ]
        # roi_2 = imgaussfilt(roi, 23)
        # # Same filter that weas used in corrige_score
        roi_2 = cv.GaussianBlur(
            roi,
            ksize=(kernel, kernel),
            sigmaX=23,
            borderType=cv.BORDER_REPLICATE,
        )
        roi_2 = cv.medianBlur(roi_2, 3)
        dif = np.absolute(roi.astype(np.int32) - roi_2.astype(np.int32)).astype(np.uint8)  # noqa
        bw = feature.canny(
            dif,
            low_threshold=th1,
            high_threshold=th2,
            sigma=sigma,
        )
        bw = cv.Canny(dif, th1, th2)
        # Number of colors in the Histogram (code just below)
        num_colors_roi = self.__imhisto(roi)
        num_colors_roi_2 = self.__imhisto(roi_2)
        num_colors_dif = self.__imhisto(dif)
        num_colors_bw = self.__imhisto(bw)

        # Getting more areas for comparison of features
        # Second ROI
        roi2 = im[1:(2 * square_size + 1),
                  (y_center - square_size):(y_center + square_size)]
        # roi2_2 = imgaussfilt(roi2, 23) # Same as before
        roi2_2 = cv.GaussianBlur(
            roi2,
            ksize=(kernel, kernel),
            sigmaX=23,
            borderType=cv.BORDER_REPLICATE,
        )
        roi2_2 = cv.medianBlur(roi2_2, 3)
        dif2 = np.absolute(roi2.astype(np.int32) - roi2_2.astype(np.int32)).astype(np.uint8)  # noqa
        bw2 = feature.canny(
            dif2,
            low_threshold=th1,
            high_threshold=th2,
            sigma=sigma,
        )
        num_colors_roi2 = self.__imhisto(roi2)
        num_colors_roi2_2 = self.__imhisto(roi2_2)
        num_colors_dif2 = self.__imhisto(dif2)
        num_colors_bw2 = self.__imhisto(bw2)

        # Third ROI
        roi3 = im[(x_center - square_size):(x_center + square_size),
                  1:(2 * square_size + 1)]
        # roi3_2 = imgaussfilt(roi3, 23) # Same as before
        roi3_2 = cv.GaussianBlur(
            roi3,
            ksize=(kernel, kernel),
            sigmaX=23,
            borderType=cv.BORDER_REPLICATE,
        )
        roi3_2 = cv.medianBlur(roi3_2, 3)
        dif3 = np.absolute(roi3.astype(np.int32) - roi3_2.astype(np.int32)).astype(np.uint8)  # noqa
        bw3 = feature.canny(
            dif3,
            low_threshold=th1,
            high_threshold=th2,
            sigma=sigma,
        )
        num_colors_roi3 = self.__imhisto(roi3)
        num_colors_roi3_2 = self.__imhisto(roi3_2)
        num_colors_dif3 = self.__imhisto(dif3)
        num_colors_bw3 = self.__imhisto(bw3)

        change = 0
        # ROI2 (or 3) is valid only if it has more than X colors.
        # The number of colors of ROI is also necessary to evaluate.
        if (
            num_colors_roi > 20
        ):  # Any comparison is made just if ROI has more than 20 colors
            # - example, a dark ROI is useless
            # - even blurred regions have more colors:
            if num_colors_roi2 < 30:  # Do not use ROI2:
                # Do not use ROI3. In this case, just ROI is analysed.:
                if num_colors_roi3 < 30:
                    if (num_colors_dif < 60) and (num_colors_bw == 1):
                        # The score must be penalized, probably blurred center
                        change = 1
                else:  # It is needed to check ROI3
                    if num_colors_bw == 1:
                        if num_colors_dif < 0.75 * num_colors_dif3:
                            change = 1
            else:  # Using ROI2
                if num_colors_roi3 < 30:  # We will use ROI and ROI2:
                    if num_colors_bw == 1:
                        if num_colors_dif < 0.75 * num_colors_dif2:
                            change = 1
                else:  # Using ROI2 and ROI3
                    if num_colors_bw == 1:
                        if (
                            (num_colors_dif < 0.75 * num_colors_dif2)
                            or (num_colors_dif < 0.75 * num_colors_dif3)
                        ):
                            change = 1
        else:
            change = 0  # No penalty for the score
        change_score_macro = False
        if score > 0.7 and change:
            change_score_macro = True

        change_score_macro = False  # Mudança teste em Março de 2023
        return change_score_macro

    def __imhisto(self, im):
        # Returns the number of non zero elements in the histogram
        lin, col = im.shape
        hist = np.zeros(256, int)
        for i in range(lin):
            for j in range(col):
                elem = im[i, j]
                hist[elem] = hist[elem] + 1

        c = [x for x in hist if x != 0]
        num_elem_nozero = len(c)
        return num_elem_nozero


if __name__ == "__main__":
    diqa = DIQA()
    diqa.load()
    print(diqa.predict(r"C:\Users\phavm\Documents\dev\python\ARNIQA\assets\01.png"))
