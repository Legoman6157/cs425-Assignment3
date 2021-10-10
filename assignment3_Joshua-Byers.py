# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 00:00:00 1970

@author: jbyers3
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
import time

#Returns the distance between [color1] and [color2]
def getDistance(color1, color2):
    a = abs(int(color2[0]) - int(color1[0]))
    b = abs(int(color2[1]) - int(color1[1]))
    c = abs(int(color2[2]) - int(color1[2]))
    d = (a**2 + b**2 + c**2)**.5
    return d

#From the given [image], return the colors at the given [positions]
def getColorsFromImage(image, positions):
    colors = np.empty((len(positions), 3), dtype="int")
    for i in range(len(positions)):
        colors[i] = image[ positions[i][0] ][ positions[i][1] ]
    return colors
        

#Get the index of the given [c]olor from the given array of [colors]
def getCentroidNum(c, colors):
    for i in range(len(colors)):
        if np.array_equiv(c, colors[i]):
            return i
    return -1

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

    positions = np.full((n, 2), -1)

    for i in range(n):
        a = rand.randint(0, nrows)
        b = rand.randint(0, ncols)
        while [a, b] in positions:
            a = rand.randint(0, nrows)
            b = rand.randint(0, ncols)
        positions[i] = [a, b]

    return positions

#Assignment function.
def kmeans(k, image):
    allChanges = []

    #Number of rows and columns for the given image
    nrows = image.shape[0]
    ncols = image.shape[1]

    #Get random positions to start out with and get their respective colors
    #   in the image.
    randomPositions = generateRandomPositions(k, nrows, ncols)
    clusterCenters = getColorsFromImage(image, randomPositions)

    #newImage is the new image to be returned with only k colors
    newImage = np.empty((nrows, ncols, 3), dtype="int")
    stopIterating = 0

    numIterations = 0

    #Algorithm for K-Means(D, K):
    #repeat
    while True:
        print("\niteration number: {}".format(numIterations))
        for i in range(k):
            print("clusterCenters[{}]:".format(i), clusterCenters[i])


        clusterCenterSums = np.zeros((k), dtype="int")

    #   for n=1 to N do
        clusterSums = np.zeros((k, 3), dtype="float")
        clusterSizes = np.zeros((k), dtype="int")
        # for r in range(nrows):
        #     for c in range(ncols):
        #       z_n <- argmin_k ||{mew}_k - x_n||   //assign example n to closest
        #                                               center
        #   end for
        #   for k=1 to K do
        for r in range(nrows):
            for c in range(ncols):
        #       X_k <- { x_N: z_n = k }             //points assigned to cluster k
                newImage[r][c]=getClosestColor(image[r][c], clusterCenters)
                centroidNum = getCentroidNum(newImage[r][c], clusterCenters)
                clusterSums[centroidNum] += image[r][c]
                clusterSizes[centroidNum] += 1
                
        #       {mew}_k <- mean(X_k)                //re-estimate center of
        #                                               cluster k
        
        for i in range(k):
            print("clusterSums[{}]:".format(i), clusterSums[i])
        for i in range(k):
            print("clusterSizes[{}]:".format(i), clusterSizes[i])
        
        means = np.empty((k,3), dtype="float")
        changes = np.empty((k, 3), dtype="float")
    
        #largestChanges track of the largest changes in the Red,
        #   Green, and Blue values.
        largestChanges = np.empty((3), dtype="float")

        #Set up the base means and their respective changes relative to the
        #   original positions
        
        for i in range(k):
            if (clusterSizes[i] != 0):
                means[i] = clusterSums[i]/clusterSizes[i]
            else:
                means[i] = 0

            changes[i] = abs(clusterCenters[i] - means[i])
        
        allChanges.append([changes])
        
        for i in range(k):
            print("means[{}]:".format(i), means[i])
        for i in range(k):
            print("changes[{}]:".format(i), changes[i])

        for i in range(3):
            largestChanges[i] = max(changes.T[i])
        print("largestChanges:", largestChanges)

    #   end for
    #until {mew}s stop changing                 //Stop after the largest
    #                                           //  change is < 1, or you
    #                                           //  have gone through 24
    #                                           //  iterations.
        #Check if the largest change is less than one.
        if max(largestChanges[0], largestChanges[1], largestChanges[2]) < 1:
            stopIterating = 1

        if stopIterating or (numIterations == 24):
            break;
        else:
            for i in range(k):
                #Set the new center of the clusters as the means of the
                #    previous groups.
                clusterCenters[i] = means[i]
            numIterations += 1
    #return z                                   //return cluster assignments
    return newImage, allChanges, clusterSizes
    

if __name__ == "__main__":
    images = []
    imageTypes = [".jpg"]
    files = os.listdir(".")
    figures = []
    
    for file in files:
        for imgType in imageTypes:
            if file.endswith(imgType):
                images.append(io.imread(file))
                continue
    
    #Required k values for assignment

    k_values = [4, 16, 32]
    
    newImage = []
    allChanges = []
    allSizes = []
    

    i = 0

    #For all found images of the allowed imageTypes,
    for image in images:
        print("Image number {}".format(i))
        allImages = []
        allImages.append(image)
        changes = []
        sizes = []

        #For all k values required by the assignment,
        for k_value in k_values:
            print("k={}".format(k_value))
            #kmeans returns the image, the list of all changes for that image,
            #   and a list of the colors (clusterCenters) for that image.
            newImage, changes, sizes = kmeans(k_value, image)
            allImages.append(newImage)
            allChanges.append(changes)
            allSizes.append(sizes)


        #Show all images
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        ax1.imshow(allImages[0])
        ax2.imshow(allImages[1])
        ax3.imshow(allImages[2])
        ax4.imshow(allImages[3])
        plt.show()
        
        plt.close()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1,3)
        ax1.set_xticks([1, 2, 3, 4])
        ax1.bar([1, 2, 3, 4], allSizes[0])
        ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        ax2.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], allSizes[1])
        ax3.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
        ax3.bar([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], allSizes[2])
        
        plt.show()
        
        i += 1