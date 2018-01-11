import itertools
import logging
import math
import collections
import warnings
import matplotlib.pyplot as plt
import numpy as np

# Data for segments is Segment : SegmentWight
segments = { }

# Data will be Segment: segmentEG
clusters = { }

# Radio area in meters 1200mx1200m
grid = [1200][1200]

radioRanges = (50, 60, 100)
numSegments = (3, 6, 9, 12, 24)
interSegDataVolume = (5, 15, 25, 35, 45)
interSegDataStD = (0, 3)
energyMotion = "1 Joule/meter"
energyComm = "2 Joule/Mbit"
initEnergyMDC = "1000 Joules"
speedMDC = "6 meter/min"
wirelessBandwidth = (0.1, 0.2, 0.5)

def findStartingEG(segmentList)


def findEG(clusterList)


def makeNonCentralClusters(clusterList)


def initCluster(segmentList)


def makeCentralCluster(clusterList)



