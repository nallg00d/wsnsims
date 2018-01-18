import os
import sys
import math
from scipy.spatial import distance


#x coord, y coord, mbit speed
S0 = [9,13,34]
S1 = [4,12,31]
S2 = [2,9,29]
S3 = [7,9,50]
S4 = [12,9,17]
S5 = [17,9,13]
S6 = [17,4,1]
S7 = [14,0,2]
S8 = [11,4,7]
S9 = [5,0,1]
S10 = [5,6,22]
S11 = [0,4,1]

nodes = list()
nodes.append(S0)
nodes.append(S1)
nodes.append(S2)
nodes.append(S3)
nodes.append(S4)
nodes.append(S5)
nodes.append(S6)
nodes.append(S7)
nodes.append(S8)
nodes.append(S9)
nodes.append(S10)
nodes.append(S11)



seg10_11 = list()
seg10_8 = list()
seg9_8 = list()
seg7_8 = list()
seg8_4 = list()
seg8_5 = list()
seg6_5 = list()
seg0_5 = list()
seg0_3 = list()
seg3_4 = list()
seg3_2 = list()
seg1_2 = list()
seg1_10 = list()

seg10_11.append(S10)
seg10_11.append(S11)

seg10_8.append(S10)
seg10_8.append(S8)

seg9_8.append(S9)
seg9_8.append(S8)

seg7_8.append(S7)
seg7_8.append(S8)

seg8_4.append(S4)
seg8_4.append(S8)

seg8_5.append(S8)
seg8_5.append(S5)

seg6_5.append(S6)
seg6_5.append(S5)

seg0_5.append(S0)
seg0_5.append(S5)

seg0_3.append(S0)
seg0_3.append(S3)

seg3_4.append(S3)
seg3_4.append(S4)

seg3_2.append(S3)
seg3_2.append(S2)

seg1_2.append(S1)
seg1_2.append(S2)

seg1_10.append(S1)
seg1_10.append(S10)


segmentList = list()

# Now add all segmetns to main segment list
segmentList.append(seg10_8)
segmentList.append(seg10_11)
segmentList.append(seg9_8)
segmentList.append(seg7_8)
segmentList.append(seg8_4)
segmentList.append(seg8_5)
segmentList.append(seg6_5)
segmentList.append(seg0_5)
segmentList.append(seg0_3)
segmentList.append(seg3_4)
segmentList.append(seg3_2)
segmentList.append(seg1_2)
segmentList.append(seg1_10)



radio1  = 100
radio2 = 50
radio3 = 25
radio4 = 30

mdc_count = 12
seg_count = len(segmentList)

# Find EG coordinates of the segment
# Center of mass for segments
def findEG(nodeList):
    eGs = list()
    totalData = 0
    yweight = 0
    xweight = 0

    # Segment has 2 nodes, so get data from each node
    for node in nodeList:
        x = node[0]
        y = node[1]
        data = node[2]

        xweight += x * data
        yweight += y * data
        totalData += data

    # rounding the data to no decimal points here for simplicity
    Cx = (xweight / totalData)
    Cy = (yweight / totalData)

    eGs.append(Cx)
    eGs.append(Cy)

    # coordinates for center of energy 
    return eGs

# Calculates data parameters for energy on a cluster
def dataCalc(cluster):

    # get Node count
    nodeCount = len(cluster)
    sum = 0
    total = 0
    # Get total Data for all nodes in cluster
    for clust in cluster:
        sum += clust[2]

    for clust in cluster:
        nodeData = clust[2]

        total += (nodeCount * nodeData) / sum
    
    return total

def getMergedData(clust_x, clust_y):

    sum = 0
    xtotal = 0
    ytotal = 0

    for x in clust_x:
        xtotal += x[2]

    for y in clust_y:
        ytotal += y[2]

    sum = xtotal + ytotal
    
    return sum


# Finds the euclidian distance between a node and eG for a segment
# We can use ditance.euclidian(a,b) instead and probaly should as part of scipy
def findEucDist(eG,node):

    distance = 0
    x = node[0]
    y = node[1]
    
    eG_x = eG[0]
    eG_y = eG[1]

    
    temp1 = (x - eG_x) ** 2
    temp2 = (y - eG_y) ** 2

    sum = temp1 + temp2

    sqrt = math.sqrt(temp1 + temp2)

    distance = sqrt
            

    return distance

# Finds tour length for a single node in a segment to eG
def findTourNode(eG, node):

    # In paper, R is 30meters
    R = 30

    distance = findEucDist(eG, node)

    tour_length = distance - (2 * R)

    return tour_length

# Gets the entire tour length of a cluster
def findTourCluster(eG,cluster):

    distance = 0

    # Loop through all ndoes in a cluster and sum the tour lengths from each node to eG
    for node in cluster:
        distance += findTourNode(eG,node)

    # Result will be negative, we need to take the absolute value
    return abs(distance)


# Sums the clusters aggregated data of x and Y
def summation(eG, cluster_x, cluster_y, clusterList):

    sum = 0
    mergedData = 0
    for clust in clusterList:
        mergedData = cluster_x[0][2] + cluster_y[0][2]
                
        sum += mergedData
        
    return sum

# initialize and form segments into clusters
def initClusters(segments):

    start_clusters = list()
    # Loop through segments 

    for seg in segments:
        start_clusters.append(seg)
        
    return start_clusters


def mergeClusters(nodes):

    return true


def hamilCycle(nodes):


    return true



# main code here

# starting out, center cluster is at eG
eG = findEG(nodes)

center_cluster_coord = eG

# initialize clusters, each segment is in it's own cluster

startClusters = initClusters(nodes)

# Central cluster starts at eG, which is eG

centralCluster = list()
centralCluster.append(S3)

# Remove from our node list that is at eG
nodes.remove(S3)

singleClusterList = list()

# Add each node to it's own cluster
for node in nodes:
    cluster = list()
    cluster.append(node)
    singleClusterList.append(cluster)

## Phase 1 results should be
# MDC Count = 5
# Cluster 0 = S0, S1  - Tour length: 320
# Cluster 1 = S2, S9, S10, S11 - Tour Length: 526
# Cluster 2 = S6, S7, S8 - Tour Length: 470
# Cluster 3 = S4, S5 - Tour Length: 330
#
# Cluster 4 = S3 ## eG

test0 = list()
test1 = list()
test2 = list()
test3 = list()

test0.append(S0)
test0.append(S1)

test1.append(S2)
test1.append(S9)
test1.append(S10)
test1.append(S11)

test2.append(S6)
test2.append(S7)
test2.append(S8)

test3.append(S4)
test3.append(S5)

print(getMergedData(test0, test1))

## Get tour lengths of clusters
print(findTourCluster(eG, test0))
print(findTourCluster(eG, test1))
print(findTourCluster(eG, test2))
print(findTourCluster(eG, test3))



sys.exit(0)
# k in the paper
mdcCount = 5

#k - 1 in the paper
numClusters = mdcCount -1

k = numClusters

listOfClusters = list()
cenClust = list()
cenClust.append(S3)

    
# Simulating a do/while loop
lowest = 0
while True:
    sum = 0
    round = 0
    energyCurrCluster = 0
    mergedCluster = list()

    # brute forcing every combination of clusters
    for clust_x in singleClusterList:
        for clust_y in singleClusterList:
            # avoid comparing same cluster to itself
            if clust_x == clust_y:
                continue
            
            ## Loop for summation
            if round == 0:
                lowest = summation(eG, clust_x, clust_y, singleClusterList)
                continue

            # Get the energy of the current cluster
            
            # This gets the energy summation for a cluster
            sum = summation(eG, clust_x, clust_y, singleClusterList)

            print("Cluster pair: ", clust_x, clust_y, " sum: ", sum, ", lowest: ", lowest)
            
            if sum < lowest:
                # New pair is found, clear out old list
                mergedCluster.append(clust_x)
                mergedCluster.append(clust_y)
                lowest = sum
        
        round += 1
        
    listOfClusters.append(mergedCluster)

    # We're done marking clusters for merging
    mergedCluster.clear()
    k -= 1
    if k == 0:
        break

    
# Just to make sure center cluster is last in list
listOfClusters.append(cenClust)

#for clust in listOfClusters:
 ###   print("Cluster: ", clust)

    


            
        
        
        



          
        
