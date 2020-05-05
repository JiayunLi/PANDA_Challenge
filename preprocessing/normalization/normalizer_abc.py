"""
Normalizer abstract base classes
"""

from __future__ import division

from abc import ABC, abstractmethod
import preprocessing.normalization.utils.misc_utils as mu
import spams
import numpy as np


class Normaliser(ABC):

    @abstractmethod
    def fit(self, target):
        """Fit the normalizer to an target image"""

    @abstractmethod
    def transform(self, I):
        """Transform an image to the target stain"""

    @abstractmethod
    def get_norm_method(self):
        """return the normalization method for current normalizer"""


class FancyNormalizer(Normaliser):

    def __init__(self):
        self.stain_matrix_target = None

    @abstractmethod
    def get_stain_matrix(self, I, *args):
        """Estimate stain matrix given an image and relevant method parameters"""

    @abstractmethod
    def get_norm_method(self):
        """return the normalization method for current normalizer"""

    @staticmethod
    def get_concentrations(I, stain_matrix, lamda=0.01):
        """
        Get the concentration matrix. Suppose the input image is H x W x 3 (uint8). Define Npix = H * W.
        Then the concentration matrix is Npix x 2 (or we could reshape to H x W x 2).
        The first element of each row is the Hematoxylin concentration.
        The second element of each row is the Eosin concentration.

        We do this by 'solving' OD = C*S (Matrix product) where OD is optical density (Npix x 3),\
        C is concentration (Npix x 2) and S is stain matrix (2 x 3).
        See docs for spams.lasso.

        We restrict the concentrations to be positive and penalise very large concentration values,\
        so that background pixels (which can not easily be expressed in the Hematoxylin-Eosin basis) have \
        low concentration and thus appear white.

        :param I: Image. A np array HxWx3 of type uint8.
        :param stain_matrix: a 2x3 stain matrix. First row is Hematoxylin stain vector, second row is Eosin stain vector.
        :return:
        """
        # param = {
        #     'lambda1': 0.15,  # not more than 20 non-zeros coefficients
        #     'numThreads': -1,  # number of processors/cores to use, the default choice is -1
        #     # and uses all the cores of the machine
        #     'mode': spams.PENALTY}  # penalized formulation
        # alpha = spams.lasso(X,D = D,return_reg_path = False,**param)

        OD = mu.RGB_to_OD(I).reshape((-1, 3))  # convert to optical density and flatten to (H*W)x3.
        return spams.lasso(OD.T, D=stain_matrix.T, mode=2, numThreads=6, lambda1=lamda, pos=True).toarray().T

    def fit(self, target):
        """
        Fit to a target image
        :param target:
        :return:
        """
        target = mu.standardize_brightness(target)
        self.stain_matrix_target = self.get_stain_matrix(target)

    def transform(self, I):
        """
        Transform an image
        :param I:
        :return:
        """
        I = mu.standardize_brightness(I)
        stain_matrix_source = self.get_stain_matrix(I)
        source_concentrations = self.get_concentrations(I, stain_matrix_source)
        return (255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target).reshape(I.shape))).astype(
            np.uint8)

    def fetch_target_stains(self):
        """
        Fetch the target stain matrix and convert from OD to RGB.
        Must call fit first (this builds the stain matrix)
        :return:
        """
        assert self.stain_matrix_target is not None, 'Run fit method first.'
        return mu.OD_to_RGB(self.stain_matrix_target)

    def hematoxylin(self, I):
        """
        Hematoxylin channel
        :param I:
        :return:
        """
        I = mu.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = self.get_stain_matrix(I)
        source_concentrations = self.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H
