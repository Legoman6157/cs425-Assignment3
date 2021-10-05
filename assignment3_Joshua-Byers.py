# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 00:00:00 1970

@author: jbyers3
"""

import numpy as np
from skimage import io
import os
import matplotlib.pyplot as plt
import time

def getRandomPositions(n, nrows, ncols):
    """
        I'll try to explain seed as best I can, from inside out:
            1.) time.time(): Get the time with nanoseconds (float)
            2.) *10000000: Bring the nanoseconds to the one's place
            3.) %(2**32): Have the number be always be below the max
                integer for the seed of np.random.RandomState() (2**32)
            4.) int(): np.random.RandomState() only accepts integers
        All this so that you can run this back-to-back and get different
            results each time. (Uses nanoseconds instead of seconds)
    """
    seed = int((time.time()*10000000)%(2**32))
    rand = np.random.RandomState(seed)

    positions = []

    for i in range(n):
        a = rand.randint(0, nrows)
        b = rand.randint(0, ncols)
        positions.append([a, b])

    return positions

def kmeans(k, image):

    nrows = image.shape[0]
    ncols = image.shape[1]

    clusterCenters = getRandomPositions(k, nrows, ncols)

    positionAssignments = np.empty((nrows, ncols, 2))

    #Algorithm for K-Means(D, K):
    #repeat
    #   for n=1 to N do
    #       z_n <- argmin_k ||{mew}_k - x_n||   //assign example n to closest
    #                                               center
    #   end for
    #   for k=1 to K do
    #       X_k <- { x_N: z_n = k }             //points assigned to cluster k
    #       {mew}_k <- mean(X_k)                //re-estimate center of
    #                                               cluster k
    #   end for
    #until {mew}s stop changing                 //Until largest change is < 1,
    #                                               but stop after 24
    #                                               iterations
    #return z                                   //return cluster assignments
    
    for center in clusterCenters:
        print(center)
    return
    
    

if __name__ == "__main__":
    images = {}
    imageTypes = [".jpg"]
    files = os.listdir(".")
    figures = []
    
    for file in files:
        for imgType in imageTypes:
            if file.endswith(imgType):
                images[file] = io.imread(file)
                continue
    
    k_values = [4, 16, 32]
    
    for key in images:
        fig, axs = plt.subplots(2, 2)
        figures.append(fig)
        for k_value in k_values:
            print(key, images[key].shape)
            kmeans(k_value, images[key])
        axs[0][0].imshow(images[key])