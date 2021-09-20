import cv2 
import numpy as np
from math import sqrt, pi

def gauss(sigma): # get kernel of gaussian, sizes depends on sigma
    half_size = 3*sigma

    dim = 2*half_size+1

    k = np.zeros((dim, dim), np.single)

    for i in range(-half_size, half_size, 1):
        for j in range(-half_size, half_size, 1):
            k[i,j] = np.exp((i**2 + j**2) / (-2*sigma)) / (sigma*2*pi)

    return k

def filteredGradient(im, sigma):
    # Computes the smoothed horizontal and vertical gradient images for a given
    # input image and standard deviation. The convolution operation should use
    # 0 padding.
    #
    # im: 2D float32 array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.

    # Returns:
    # Fx: 2D double array with shape (height, width). The horizontal
    #     gradients.
    # Fy: 2D double array with shape (height, width). The vertical
    #     gradients.

    kernel = gauss(sigma)

    # deriv of gaussian in x direction
    xconv = np.array([[-1, 1, 0]])
    dx = cv2.filter2D(kernel,-1,xconv)
    Fx = cv2.filter2D(im,-1,dx)
    # print(np.amax(Fx))
    # cv2.imwrite("fx.jpeg", Fx)

    # deriv of gaussian in y direction
    yconv = np.array([[-1], [1], [0]])
    dy = cv2.filter2D(kernel,-1,yconv)
    Fy = cv2.filter2D(im,-1,dy)
    # cv2.imwrite("fy.jpeg", Fy)
    # print(np.amax(Fy))
    return Fx, Fy


def edgeStrengthAndOrientation(Fx, Fy):
    # Given horizontal and vertical gradients for an image, computes the edge
    # strength and orientation images.
    #
    # Fx: 2D double array with shape (height, width). The horizontal gradients.
    # Fy: 2D double array with shape (height, width). The vertical gradients.

    # Returns:
    # F: 2D double array with shape (height, width). The edge strength
    #        image.
    # D: 2D double array with shape (height, width). The edge orientation
    #        image.

    F = np.sqrt(np.square(Fx) + np.square(Fy))
    div = Fy / (Fx+(1e-13))
    D = np.arctan(div)
#     print(orient)
    D = np.where(D > 0, D, D+pi)
#     print(orient)
    return F, D


def suppression(F, D):
    # Runs nonmaximum suppression to create a thinned edge image.
    #
    # F: 2D double array with shape (height, width). The edge strength values
    #    for the input image.
    # D: 2D double array with shape (height, width). The edge orientation
    #    values for the input image.

    # Returns:
    # I: 2D double array with shape (height, width). The output thinned
    #        edge image.

    # For each pixel, find the direction 𝐷∗∈0,𝜋/4,𝜋/2,3𝜋/4 that is closest to the orientation 𝐷 at that pixel.
    angles = np.array([0, pi/4, pi/2, 3*pi/4])
    D_prime = np.zeros(D.shape)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            D_prime[i][j] = angles[np.abs(angles - D[i][j]).argmin()]
    # print(D_prime)

    # If the edge strength 𝐹(𝑥,𝑦) is smaller than at least one of its neighbors along 𝐷∗, 
    # set 𝐼(𝑥,𝑦)=0, else set 𝐼(𝑥,𝑦)=𝐹(𝑥,𝑦).
    I = np.zeros(F.shape)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if D_prime[i][j] == 0:
                n1, n2 = (i, max(0,j-1)), (i, min(I.shape[1]-1,j+1))
            elif D_prime[i][j] == pi/4:
                n1, n2 = (max(0,i-1), min(I.shape[1]-1,j+1)), (min(I.shape[0]-1, i+1), max(0,j-1))
            elif D_prime[i][j] == pi/2:
                n1, n2 = (max(0,i-1), j), (min(I.shape[0]-1, i+1), j)
            else:
                n1, n2 = (max(0,i-1), max(0,j-1)), (min(I.shape[0]-1, i+1), min(I.shape[1]-1,j+1))
                
            if F[i][j] > F[n1] or F[i][j] > F[n2]:
                I[i][j] = F[i][j]

    return I


def hysteresisThresholding(I, D, tL, tH):
    # Runs hysteresis thresholding on the input image.

    # I: 2D double array with shape (height, width). The input's edge image
    #    after thinning with nonmaximum suppression.
    # D: 2D double array with shape (height, width). The edge orientation
    #    image.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary array with shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0. 

    edgeMap = np.zeros(I.shape)
    I_normal = I / np.amax(I)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if I_normal[i][j] > tH:
                edgeMap[i][j] = 1
                to_check = [(i,j)]
                while(to_check):
                    p = to_check.pop(0)
                    if edgeMap[p] == 0:
                        if D[p[0]][p[1]] == 0:
                            n1, n2 = (max(0,p[0]-1), p[1]), (min(I.shape[0]-1, p[0]+1), p[1])
                        elif D[p[0]][p[1]] == pi/4:
                            n1, n2 = (max(0,p[0]-1), max(0,p[1]-1)), (min(I.shape[0]-1, p[0]+1), min(I.shape[1]-1,p[1]+1))
                        elif D[p[0]][p[1]] == pi/2:
                            n1, n2 = (p[0], max(0,p[1]-1)), (p[0], min(I.shape[1]-1,p[1]+1))
                        else:
                            n1, n2 = (max(0,p[0]-1), min(I.shape[1]-1,p[1]+1)), (min(I.shape[0]-1, p[0]+1), max(0,p[1]-1))
                        if I[n1] > tL:
                            to_check.append(n1)
                            edgeMap[n1] = 1
                        if I[n2] > tL:
                            to_check.append(n2)
                            edgeMap[n2] = 1

    return edgeMap

def cannyEdgeDetection(im, sigma, tL, tH):
    # Runs the canny edge detector on the input image. This function should
    # not duplicate your implementations of the edge detector components. It
    # should just call the provided helper functions, which you fill in.
    #
    # IMPORTANT: We have broken up the code this way so that you can get
    # better partial credit if there is a bug in the implementation. Make sure
    # that all of the work the algorithm does is in the proper helper
    # functions, and do not change any of the provided interfaces. You
    # shouldn't need to create any new .py files, unless they are for testing
    # these provided functions.
    # 
    # im: 2D double array with shape (height, width). The input image.
    # sigma: double. The standard deviation of the gaussian blur kernel.
    # tL: double. The low threshold for detection.
    # tH: double. The high threshold for detection.

    # Returns:
    # edgeMap: 2D binary image of shape (height, width). Output edge map,
    #          where edges are 1 and other pixels are 0.


    # TODO: Implement me!

    Fx, Fy = filteredGradient(im, sigma)
    F, D = edgeStrengthAndOrientation(Fx, Fy)
    I = suppression(F, D)
    edgeMap = hysteresisThresholding(I, D, tL, tH)
    # print(edgeMap.shape)

    return edgeMap
