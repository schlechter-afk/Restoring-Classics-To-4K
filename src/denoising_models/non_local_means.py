import numpy as np

def findAllNeighbours(padImg, small_window, big_window, h, w):
    """Function to preprocess neighbours (small_window x small_window) for each pixel"""
    smallWidth = small_window//2
    bigWidth = big_window//2
    neighbours = np.zeros((padImg.shape[0],padImg.shape[1],small_window,small_window))

    for i in range(bigWidth,bigWidth + h):
        for j in range(bigWidth,bigWidth + w):   
            neighbours[i,j] = padImg[
                (i - smallWidth):(i + smallWidth + 1), 
                (j - smallWidth):(j + smallWidth + 1)
            ]

    return neighbours


def evaluateNorm(pixelWindow, neighborWindow, Nw):
    """Function to calculate the weighted average value (Ip) for each pixel"""
    Ip_Numerator = 0
    Z = 0

    for i in range(neighborWindow.shape[0]):
      for j in range(neighborWindow.shape[1]):

        q_window = neighborWindow[i,j]
        q_x, q_y = q_window.shape[0]//2, q_window.shape[1]//2
        Iq = q_window[q_x, q_y]

        w = np.exp(-1 * ((np.sum((pixelWindow - q_window)**2)) / Nw))
        Ip_Numerator = Ip_Numerator + (w*Iq)
        Z = Z + w

    return Ip_Numerator/Z


class NLMeans():

    def solve(self, img, h=30, small_window=7, big_window=21):
        """Helper Function to denoise the image using Non-Local Means Denoising"""
        padImg = np.pad(img, big_window//2, mode='reflect')
        return self.NLM(padImg, img, h, small_window, big_window)


    @staticmethod
    def NLM(padImg, img, h, small_window, big_window):
        """Function to denoise the image using Non-Local Means Denoising"""
        Nw = (h**2)*(small_window**2)
        _, w = img.shape
        result = np.zeros(img.shape)
        bigWidth = big_window//2
        neighbours = findAllNeighbours(padImg, small_window, big_window, h, w) 

        for i in range(bigWidth, bigWidth + h):
            for j in range(bigWidth, bigWidth + w):
                pixelWindow = neighbours[i,j]
                neighborWindow = neighbours[(i - bigWidth):(i + bigWidth + 1) , (j - bigWidth):(j + bigWidth + 1)]
                Ip = evaluateNorm(pixelWindow, neighborWindow, Nw)
                result[i - bigWidth, j - bigWidth] = max(min(255, Ip), 0)

        return result