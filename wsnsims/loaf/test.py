import os
import sys
import math

#x coord, y coord, mbit speed

node1 = [0,4,1]
node2 = [5,0,1]
node3 = [14,0,2]
node4 = [5,6,21]
node5 = [11,4,7]
node6 = [17,4,1]
node7 = [2,9,29]
node8 = [8,9,50]
node9 = [12,9,17]
node10 = [17,9,13]
node11 = [4,12,31]
node12 = [9,13,34]

segment = list()
cluster = list()
nonEssentialClusters = list()
centralCluster = list()

segment.append(node1)
segment.append(node2)
segment.append(node3)
segment.append(node4)
segment.append(node5)
segment.append(node6)
segment.append(node7)
segment.append(node8)
segment.append(node9)
segment.append(node10)
segment.append(node11)
segment.append(node12)

# Find EG coordinates of the segment
# Center of mass for segments
def findEG(segments):

    eGs = list()
    totalData = 0
    yweight = 0
    xweight = 0
    
    for seg in segments:
        x = seg[0]
        y = seg[1]
        data = seg[2]
        
        xweight += x * data
        yweight += y * data
        totalData += data
        
    Cx = xweight / totalData
    Cy = yweight / totalData
     
    eGs.append(Cx)
    eGs.append(Cy)
    
    return eGs

## Need to find non-central clusters
## First though, we need to organize clusters in the first place
def findEucDist(segment, eG):
    x = segment[0]
    y = segment[1]

    # eg is a list, get each one individual
    eG_x = eG[0]
    eG_y = eG[1]

    #We now have 2 vectors we can do the math on and get the distance
    temp1 = (x - eG_x) ** 2
    temp2 = (y - eG_y) ** 2

    distance = math.sqrt(temp1 + temp2)
    
    return distance


def findNonEssentialClusters(segments):

    eG = findEG(segments)
    radio_range = 100

    # loop through segments, verify the EUC distance between the two points in a segment
    # if distance is less than or equal to the radio range
    # Add to nonEssentialClusters
    for seg in segments:
        if findEucDist(seg, eG) <= radio_range:
            nonEssentialClusters.append(seg)
            
    return true
        
        


#print(findEG(segment))
