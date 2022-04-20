#!/usr/bin/python

import random
import collections
import math
import sys
import os
from typing import List

from util import *


############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    words = x.split()
    freq = collections.defaultdict(int)
    for word in words:
        freq[word] += 1
    return freq

    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''
    weights = {}  # feature => weight

    def gradient(x, y):
        if dotProduct(weights, featureExtractor(x)) * y < 1:
            return -y
        else:
            return 0

    def predictor(x):
        return 1 if dotProduct(weights, featureExtractor(x)) > 0 else -1

    for x, y in trainExamples:
        for feature in featureExtractor(x):
            weights[feature] = 0
    for i in range(numIters):
        for x, y in trainExamples:
            increment(weights, (-1) * gradient(x, y) * eta, featureExtractor(x))
        #print('iteration {}: w = {}, train error = {}'.format(i, weights, evaluatePredictor(trainExamples, predictor)))
        #print('iteration {}: w = {}, test error = {}'.format(i, weights, evaluatePredictor(testExamples, predictor)))


    return weights


############################################################
# Problem 3c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a nonzero score under the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    def generateExample():
        phi = {feature : random.random() for feature in random.sample(list(weights.keys()), random.randint(1, len(weights)))}
        y = 1 if dotProduct(weights, phi) > 0 else -1
        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3e: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''

    def extract(x):
        x = x.replace(" ", "")
        features = collections.defaultdict(int)
        for i in range(len(x) - (n-1)):
            features[x[i:i+n]] += 1
        return features

    return extract


############################################################
# Problem 4: k-means
############################################################


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''

    # expand the square formula and pass in precomputed square terms
    def fastSquare(a: dict, b: dict, aSquared, bSquared):
        return aSquared + bSquared - 2 * dotProduct(a, b)

    def squareDist(a: dict, b: dict):
        keys = set(a.keys())
        keys.update(b.keys())
        ans = 0
        for key in keys:
            ans += (a.get(key, 0) - b.get(key, 0)) ** 2
        return ans

    def fastPickClosest(point: dict, point_square: float, centroids: List[dict], centroid_squares: List[float]):
        index, val = -1, float("inf")
        for i in range(len(centroids)):
            dist = fastSquare(point, centroids[i], point_square, centroid_squares[i])
            if dist < val:
                index = i
                val = dist
        return index


    def pickClosest(point: dict, centroids: List[dict]):
        index, val = -1, float("inf")
        for i in range(len(centroids)):
            dist = squareDist(point, centroids[i])
            if dist < val:
                index = i
                val = dist
        return index


    assignments = [0] * len(examples)
    random.seed(42)
    centroids = random.sample(examples, K)
    centroid_squares = [dotProduct(centroid, centroid) for centroid in centroids]
    loss = 0
    example_squares = [dotProduct(ex, ex) for ex in examples]

    for _ in range(maxIters):
        cluster_counts = collections.defaultdict(int)
        new_centroid_squares = [dotProduct(centroid, centroid) for centroid in centroids]
        for i in range(len(examples)):
            #z = pickClosest(examples[i], centroids)
            z = fastPickClosest(examples[i], example_squares[i], centroids, new_centroid_squares)
            assignments[i] = z
            cluster_counts[z] += 1

        new_centroids = [{} for _ in range(K)]

        for i in range(len(examples)):
            z = assignments[i]
            increment(new_centroids[z], 1 / cluster_counts[z], examples[i])

        if centroids == new_centroids:
            break
        else:
            centroids = new_centroids
            centroid_squares = new_centroid_squares


    for i in range(len(assignments)):
        # loss += squareDist(centroids[assignments[i]], examples[i])
        loss += fastSquare(centroids[assignments[i]], examples[i],
                           centroid_squares[assignments[i]], example_squares[i])

    return (centroids, assignments, loss)