import os
import sys
import math

# List to hold all the segments
mainSegments = list()

#x coord, y coord, mbit speed
node1 = [0,4,1]
node2 = [5,0,1]
#
node3 = [14,0,2]
node4 = [5,6,21]
#
node5 = [11,4,7]
node6 = [17,4,1]
#
node7 = [2,9,29]
node8 = [8,9,50]
#
node9 = [12,9,17]
node10 = [17,9,13]
#
node11 = [4,12,31]
node12 = [9,13,34]

segment = list()
seg1 = list()
seg2 = list()
seg3 = list()
seg4 = list()
seg5 = list()
seg6 = list()

seg1.append(node1)
seg1.append(node2)

seg2.append(node3)
seg2.append(node4)

seg3.append(node5)
seg3.append(node6)

seg4.append(node7)
seg4.append(node8)

seg5.append(node9)
seg5.append(node10)

seg6.append(node11)
seg6.append(node12)

# Now add all segmetns to main segment list
mainSegments.append(seg1)
mainSegments.append(seg2)
mainSegments.append(seg3)
mainSegments.append(seg4)
mainSegments.append(seg5)
mainSegments.append(seg6)

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

radio1  = 100
radio2 = 50
radio3 = 25
radio4 = 30

mdc_count = 12
seg_count = len(segment)

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

    clusters = list()
    xEG = list()
    yEG = list()
    mainEG = list()

    #findEG returns coordinate list of EG center of mass
    # Loop through main segments
    for seg in segments:
        # each segment has two nodes
        eG = findEG(seg)

        xEG = eG[0]
        yEG = eG[1]
        
        # Make new cluster with eG and specific segmetn
        # Cluster0 = Segment between Node 1 and Node 2, eG
        cluster = list()
        cluster.append(seg)
        cluster.append(eG)

        clusters.append(cluster)

        # we're getting eG for seach segment

    tempx = 0
    for x in xEG:
        tempx += x

    mainEG.append(tempx)

    tempy = 0
    for y in yEG:
        tempy += y

    # List of the mainEG (x,y)
    mainEG.append(tempy)

    # loop Through cluster list to get temporary central cluster

    # Assuming central cluster will be at eG
    tempCentralCluster = mainEG
    
    for clust in clusters:
        seg = clust[0]
        eG = clust[1]

        # assign temporary central cluster
        # using radio1 as a temporary measure
        if(findEucDist(seg, mainEG) <= radio1):
           tempCentralCluster = clust

    # We now need to remove the central cluster frmo the cluster list
    clusters.remove(tempCentralCluster)
    
    mdcE1 = 125
    mdcE2 = 50
    mdcE3 = 90
    mdcE4 = 23

    comE1 = 50
    comE2 = 22
    comE3 = 97
    comE4 = 30

    # We now have a cluster per segment.. we need to group the clusters based on minimum energy
           
                
            #E_c = Communication Energy required
            # ME = energy required for MDC to serve
            
            
    return true

# Find central cluster based on eG of existing clusters
def findCentralCluster(clusters):


    return true
        


#print(findEG(segment))
