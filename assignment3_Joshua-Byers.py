# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 00:00:00 1970

@author: jbyers3
"""

"""
    Pick k random spots.
    Get the color of those spots.
    For all positions, find the closest matching color.
    Set all positions to have the closest color (smallest change in rgb) once
        you begin changing.
"""

import numpy as np
from skimage import io
import os
import time

#Returns the distance between [position1] and [position2]
def getDistance(position1, position2):
    a = abs(position2[0] - position1[0])
    b = abs(position2[1] - position1[1])
    c = abs(position2[2] - position1[2])
    d = (a**2 + b**2 + c**2)**.5
    return d

#Relative to point [p], returns the closest position from the given list of
#   [positions]
def getClosestColor(p, positions):
    numPositions = len(positions)

    distance = 0
    smallestDistance = getDistance(p, positions[0])
    closestPosition = positions[0]

    for i in range(1, numPositions):
        distance = getDistance(p, positions[i])
        if (smallestDistance > distance):
            smallestDistance = distance
            closestPosition = positions[i]

    return closestPosition

#Returns [n] random positions whose maximum x value is [nrows]-1 and maximum y
#   value is [ncols]-1
def generateRandomPositions(n, nrows, ncols):
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

def getColors(positions, image):
    colors = np.empty(len(positions))
    for position in positions:
        colors.append(image[ position[0] ][ position[1] ])
    return colors

def kmeans(k, image):

    
    nrows = image.shape[1]
    ncols = image.shape[0]

    randomPositions = generateRandomPositions(k, nrows, ncols)
    clusterCenters = getColors(randomPositions, image)

    clusterAssignments = np.empty((nrows, ncols, 2), dtype="int")
    
    #smallestChanges just keeps track of the smallest changes in the Red,
    #   Green, and Blue values.
    smallestChanges = np.empty((3), dtype="int")
    stopIterating = 0
    
    

    #Algorithm for K-Means(D, K):
    #repeat
    # while True:
    #   for n=1 to N do
    for c in range(ncols):
        for r in range(nrows):
                #The character limit for good-looking code can be difficult at
                #   times.
                #   row -> r,
                #   col -> c
            clusterAssignments[c][r]=getClosestColor([c, r], clusterCenters)
    #       z_n <- argmin_k ||{mew}_k - x_n||   //assign example n to closest
    #                                               center
    #   end for
    #   for k=1 to K do
    #       X_k <- { x_N: z_n = k }             //points assigned to cluster k
    #       {mew}_k <- mean(X_k)                //re-estimate center of
    #                                               cluster k
    #   end for
    #until {mew}s stop changing                 //Until largest change is < 1,
        # for change in smallestChanges:
        #     if change < 1:
        #         stopIterating = 1
        # if stopIterating:
        #     break;
    #                                               but stop after 24
    #                                               iterations
    #return z                                   //return cluster assignments
    return clusterAssignments
    

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
    allAssignments = []
    
    for key in images:
        # fig, axs = plt.subplots(2, 2)
        # figures.append(fig)
        for k_value in k_values:
            print(key, images[key].shape, "k={}".format(k_value))
            allAssignments.append(kmeans(k_value, images[key]))
        # axs[0][0].imshow(images[key])