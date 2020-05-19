import numpy as np
import cv2


class ReinhardNormalizer:
    """
    Normalize a patch stain to the target image using the method of:
    E. Reinhard, M. Adhikhmin, B. Gooch, and P. Shirley, ‘Color transfer between images’, IEEE Computer Graphics and Applications, vol. 21, no. 5, pp. 34–41, Sep. 2001.
    """

    def __init__(self):
        super().__init__()
        self.target_concentrations = np.array([[148.60, 41.56], [169.30, 9.01], [105.97, 6.67]])

    def fit(self, values):
        if values:
            self.target_concentrations = values
        return

    def transform(self, I, mask):
        """
        Transform an image
        :param I:
        :param mask
        :return:
        """

        whitemask = ~mask
        imagelab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)

        imageL, imageA, imageB = cv2.split(imagelab)
        # mask is valid when true
        imageLM = np.ma.MaskedArray(imageL, whitemask)
        imageAM = np.ma.MaskedArray(imageA, whitemask)
        imageBM = np.ma.MaskedArray(imageB, whitemask)
        # Sometimes STD is near 0, or 0; add epsilon to avoid div by 0 -NI

        epsilon = 1e-11
        imageLMean = imageLM.mean()
        imageLSTD = imageLM.std() + epsilon
        imageAMean = imageAM.mean()
        imageASTD = imageAM.std() + epsilon

        imageBMean = imageBM.mean()
        imageBSTD = imageBM.std() + epsilon

        # normalization in lab
        imageL = (imageL - imageLMean) / imageLSTD * self.target_concentrations[0][1] + self.target_concentrations[0][0]
        imageA = (imageA - imageAMean) / imageASTD * self.target_concentrations[1][1] + self.target_concentrations[1][0]
        imageB = (imageB - imageBMean) / imageBSTD * self.target_concentrations[2][1] + self.target_concentrations[2][0]

        imagelab = cv2.merge((imageL, imageA, imageB))
        imagelab = np.clip(imagelab, 0, 255)
        imagelab = imagelab.astype(np.uint8)

        # Back to RGB space
        returnimage = cv2.cvtColor(imagelab, cv2.COLOR_LAB2RGB)
        # Replace white pixels
        returnimage[whitemask] = I[whitemask]

        return returnimage

    def get_mean_std(self, I, mask):
        """
        Get mean and standard deviation of each channel
        :param I: uint8
        :param mask: mask
        :return:
        """

        whitemask = ~mask
        imagelab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
        imageL, imageA, imageB = cv2.split(imagelab)
        # mask is valid when true
        imageLM = np.ma.MaskedArray(imageL, whitemask)
        imageAM = np.ma.MaskedArray(imageA, whitemask)
        imageBM = np.ma.MaskedArray(imageB, whitemask)

        epsilon = 1e-11
        imageLMean = imageLM.mean()
        imageLSTD = imageLM.std() + epsilon
        imageAMean = imageAM.mean()
        imageASTD = imageAM.std() + epsilon

        imageBMean = imageBM.mean()
        imageBSTD = imageBM.std() + epsilon

        return [imageLMean, imageAMean, imageBMean], [imageLSTD, imageASTD, imageBSTD]

    def get_norm_method(self):
        return "reinhard"