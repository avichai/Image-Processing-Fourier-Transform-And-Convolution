__author__ = 'Avichai'

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread as imread
from skimage.color import rgb2gray
from scipy import signal as sig
import os


IMAGE_DIM = 2
IDENTITY_KERNEL_SIZE = 1
BINOMIAL_MAT = [0.5, 0.5]
DER_VEC = [1, 0, -1]

GRAY = 1
RGB = 2
NORM_PIX_FACTOR = 255
DIM_RGB = 3
MAX_PIX_VALUE = 256
MIN_PIX_VALUE = 0
Y = 0
ROWS = 0
COLS = 1


def read_image(filename, representation):
    """this function reads a given image file and converts it into a given
    representation:
    filename - string containing the image filename to read.
    representation - representation code, either 1 or 2 defining if the
                     output should be either a grayscale image (1) or an
                     RGB image (2).
    output - the image in the given representation when the pixels are
             of type np.float32 and normalized"""
    filename = os.path.abspath(filename)
    if not os.path.exists(filename):
        return
    im = imread(filename)
    if im.dtype == np.float32:
        '''I don't handle this case, we asume imput in uint8 format'''
        return
    if representation == GRAY:
        im = rgb2gray(im).astype(np.float32)
        return im
    im = im.astype(np.float32)
    im /= NORM_PIX_FACTOR
    return im


def DFT(signal):
    ''' function that transform a 1D discrete signal to its fourier
    representation. (without loops)
    :param signal: an array or matrix of dtype float32
    :return: an array or matrix of dtype complex128 with the same shape
    as fourier_signal where it is the fourier tranform of
    signal if it's a vector and it's the fourier tranform of
    fourier_signal cols if it's a matrix.
    '''
    dims = signal.shape
    shouldConvert = False
    if len(dims) == 1:
        shouldConvert = True
        signal = np.array(signal).reshape(signal.shape[ROWS], 1)
    dims = signal.shape
    if len(dims) != IMAGE_DIM:
        '''this is against instruction'''
        raise NameError('houston we have a problem')
    N = dims[ROWS]
    u = np.array(np.arange(N).reshape(N, 1))
    xMulU = np.transpose(u) * u
    transMat = np.exp((-2j * np.pi * xMulU) / N).astype(np.complex128)
    if (shouldConvert):
        return np.dot(transMat, signal).reshape(N).astype(np.complex128)
    return np.dot(transMat, signal).astype(np.complex128)


def IDFT(fourier_signal):
    '''
    function that transform a 1D discrete Fourier_signal to its regular
    representation. (without loops)
    :param fourier_signal:  an array or matrix of dtype float32
    :return: an array or matrix of dtype complex128 with the same shape
    as fourier_signal where it is the inverse fourier tranform of
    fourier_signal if it's a vector and it's the inverse fourier tranform of
    fourier_signal cols if it's a matrix.
    '''
    dims = fourier_signal.shape
    shouldConvert = False
    if len(dims) == 1:
        shouldConvert = True
        fourier_signal = np.array(fourier_signal).reshape(
            fourier_signal.shape[ROWS], 1)
    dims = fourier_signal.shape
    if len(dims) != IMAGE_DIM:
        '''this is against instruction'''
        raise NameError('houston we have a problem')
    N = dims[ROWS]
    x = np.array(np.arange(N).reshape(N, 1))
    xMulU = np.transpose(x) * x
    transMat = (np.exp((2j * np.pi * xMulU) / N) / N).astype(np.complex128)
    signal = np.dot(transMat, fourier_signal)
    if (shouldConvert):
        return signal.reshape(N).astype(np.complex128)
    ''''we have been specifically requested to return complex number!'''''
    return signal.astype(np.complex128)


def DFT2(image):
    '''
    functions that convert a 2D discrete signal to its Fourier representation
    (without loops)
    :param image: grayscale image of dtype float32
    :return:  2D array of dtype complex128 the fourier of image
    '''
    midImage = DFT(image)
    invFourier = DFT(np.transpose(midImage))
    return np.transpose(invFourier)


def IDFT2(fourier_image):
    '''
    o functions that convert Fourier representation to its 2D discrete signal
    (without loops)
    :param fourier_image: a 2D array of dtype complex128.
    :return:  grayscale image of dtype float32 the inv fourier transform of
    fourier_image
    '''
    midImage = IDFT(np.transpose(fourier_image))
    return IDFT(np.transpose(midImage))


def getMagnitude(derX, derY):
    '''
    gets the magnitude using the derivative of x and y.
    :param derX: derivative x
    :param derY: derivative y
    :return: the magnitude of the image
    '''
    return (np.sqrt(np.abs(derX) ** 2 + np.abs(derY) ** 2)).astype(np.float32)


def conv_der(im):
    '''
    getting the magnitude of an image using convolution
    :param im:  grayscale images of type float32.
    :return:  grayscale images of type float32 which is the magnitude of
    ima.
    '''
    maskX = np.array(DER_VEC, ndmin=2)
    maskY = np.transpose(maskX)
    derX = sig.convolve(im, maskX, mode='same')
    derY = sig.convolve(im, maskY, mode='same')
    return getMagnitude(derX, derY)


def fourier_der(im):
    '''
    a function that computes the magnitude of image derivatives using
    Fourier transform
    :param im: float32 grayscale image
    :return: the magnitude of im in float32
    '''

    '''implemnting the algorithm learned in class'''
    fourierImage = DFT2(im)
    shiftFourierImage = np.fft.fftshift(fourierImage)
    dimX = shiftFourierImage.shape[ROWS]
    dimY = shiftFourierImage.shape[COLS]
    uVec = np.arange(np.ceil((-1 * dimX) / 2), np.ceil(dimX / 2), 1).reshape(
        dimX, 1)
    vVec = np.arange((-1 * dimY) // 2, dimY // 2, 1).reshape(1, dimY)
    fourMulU = np.multiply(shiftFourierImage, uVec)
    fourMulV = np.multiply(shiftFourierImage, vVec)
    unShiftX = np.fft.ifftshift(fourMulU)
    unShiftY = np.fft.ifftshift(fourMulV)
    derX = (2j * np.pi) / dimX * IDFT2(unShiftX)
    derY = (2j * np.pi) / dimY * IDFT2(unShiftY)

    return getMagnitude(derX, derY)


def checkValidKernelSize(kernel_size):
    if not (isinstance(kernel_size, int)) or kernel_size % 2 == 0 or \
                    kernel_size < IDENTITY_KERNEL_SIZE:
        raise NameError("bad input for kernel size")


def getBlurVec(kernel_size):
    '''
    gets the blurring vector in the length of the kernel size
    :param kernel_size: the length of the wished kernel
    :return: the 1d vector we want
    '''
    if kernel_size == IDENTITY_KERNEL_SIZE:
        return [1]
    return sig.convolve(BINOMIAL_MAT, getBlurVec(kernel_size - 1))


def getBlurMat(kernel_size):
    '''
    getting a blur kernel of size kernel_sise^2
    :param kernel_size: the size of the wished kernel in
    each dimension (an odd integer)
    :return: blur kernel of size kernel_sise^2
    '''
    '''geeting the blure vec in 1d'''
    blurVec = getBlurVec(kernel_size)
    '''creating the 2d kernel'''
    blurAsMat = np.array(blurVec)
    blurMat = sig.convolve2d(blurAsMat.reshape(kernel_size, 1),
                             blurAsMat.reshape(1, kernel_size))
    return blurMat


def blur_spatial(im, kernel_size):
    '''
    function that performs image blurring using 2D convolution
    between the image f and a gaussian kernel g.
    :param im: image to be blurred (grayscale float32 image).
    :param kernel_size: is the size of the gaussian kernel in
    each dimension (an odd integer)
    :return:  the output blurry image (grayscale float32 image).
    '''
    checkValidKernelSize(kernel_size)
    '''the kernel will do nothing'''
    if kernel_size == IDENTITY_KERNEL_SIZE:
        return im
    '''getting the bluring matrix'''
    blurMat = getBlurMat(kernel_size)
    return sig.convolve2d(im, blurMat, mode='same', boundary="wrap").astype(
        np.float32)


def blur_fourier(im, kernel_size):
    '''
    a function that performs image blurring with gaussian kernel in
    Fourier space.
    :param im: is the input image to be blurred (grayscale float32 image).
    :param kernel_size: - is the size of the gaussian in each dimension
    (an odd integer).

    :return: the output blurry image (grayscale float32 image)
    '''
    checkValidKernelSize(kernel_size)
    if kernel_size == IDENTITY_KERNEL_SIZE:
        return im
    rows = im.shape[ROWS]
    cols = im.shape[COLS]
    '''getting the bluring matrix'''
    blurMat = getBlurMat(kernel_size)
    '''padding the matrix with zeros so it'll be in the size of im'''
    padMat = np.zeros(im.shape).astype(np.float32)
    '''putting in the center 1'''
    padMat[rows // 2, cols // 2] = 1
    filterGaus = sig.convolve2d(padMat, blurMat, mode='same')
    '''shifting the filter so it would work with the DFT of im'''
    shiftFilterGaus = np.fft.fftshift(filterGaus)

    fourierShiftFilterGaus = DFT2(shiftFilterGaus)
    shiftFourierIm = DFT2(im)
    fourierBlurIm = np.multiply(shiftFourierIm, fourierShiftFilterGaus)

    return IDFT2(fourierBlurIm).astype(np.float32)


FILE_1 = "C:\\Users\\Avichai\\Desktop\\ImageP\\ex1\\external\\jerusalem.jpg"
FILE_2 = "C:\\Users\\Avichai\\Desktop\\ImageP\\ex1\\external\\LowContrast.jpg"
FILE_3 = "C:\\Users\\Avichai\\Desktop\\ImageP\\ex1\\external\\monkey.jpg"