import os
import sys
import math
import numpy as np
import scipy.spatial as sp
from scipy.spatial import distance
from wsnsims.core import linalg



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

# This will acutally compute a tour between the points of a cluster
def computeTourCluster(eG, cluster):

    npList = np.array((eG[0], eG[1]))

    new = list()

    new.append(eG)

    # Create list of x and y of each node in cluster, since the node format is [x, y, mbit]
    for node in cluster:
        x = node[0]
        y = node[1]

        w = list()
        w.append(x)
        w.append(y)
        new.append(w)

    # Should have new list
    # eG coordinates, node1 in cluster coords, node 2 in coords, ...
    # radio_range default = 30
    R = 30

    if len(new) == 2:
        vertices = np.array([0,1])
    else:
        # Create convex hull object
        hull = sp.ConvexHull(new, qhull_options='QJ Pp')
        vertices = hull.vertices

    tour = list(vertices)

    # make a np array based on our existing list
    npNew = np.array(new)

    collection_points = np.empty_like(npNew)
    center_of_mass = linalg.centroid(npNew[vertices])

    
    
    for vertex in vertices:
        if np.all(np.isclose(center_of_mass, npNew[vertex])):
            collection_points[vertex] = np.copy(npNew[vertex])
            continue

        cp = center_of_mass - npNew[vertex]
        cp /= np.linalg.norm(cp)
        cp *= R
        cp += npNew[vertex]
        collection_points[vertex] = cp



    interior = np.arange(start=0, stop=len(npNew), step=1)
    interior = np.delete(interior, vertices, 0)

    for point_idx in interior:

        closest_segment = -1
        closest_distance = np.inf
        closest_perp = np.zeros((1,2))

        p = npNew[point_idx]

        tail = len(tour) - 1
        head = 0
        while head < len(tour):

            start_idx = tour[tail]
            end_idx = tour[head]

            start = collection_points[start_idx]
            end = collection_points[end_idx]

            perp_len, perp_vec = linalg.closest_point(start, end, p)

            if perp_len < closest_distance:
                closest_segment = head
                closest_distance = perp_len
                closest_perp = perp_vec

            tail = head
            head += 1

        tour.insert(closest_segment, point_idx)
        collect_point = closest_perp - p

        radius = np.linalg.norm(collect_point)

        if radius > R:
            collect_point /= radius
            collect_point *= R

        collect_point += p
        collection_points[point_idx] = collect_point

    tour.append(tour[0])

        
    # length calculations
    total = 0
    tail = 0
    head = 1

    while head < len(vertices):
        start = collection_points[vertices[tail]]
        stop = collection_points[vertices[head]]

        total += np.linalg.norm(stop - start)
        tail += 1
        head += 1

    return total

def findCommEnergy(eG, node):

    # data volume * communication cost
    # Taken form wsnsims/core/Environment.py
    comm_cost = 2.0

    # get data volume (index 2) of node
    data_volume = node[2]
    
    #comm_energy = data_volume * comm_cost
    comm_energy = (10*data_volume)/5
    
    return comm_energy

def findCommEnergyCluster(eG, cluster):

    sum = 0
    for node in cluster:
        sum += findCommEnergy(eG, node)
        
    return sum

def findMoveEnergyCluster(eG, cluster):

    sum = 0
    # Taken from wsnsims/core/Environment.py
    move_cost = 1
    
    sum = computeTourCluster(eG, cluster) * move_cost

    return sum


def totalEnergyCluster(eG, cluster):

    # TotalCommEnergyCluster + TotalMoveEnergyCluster
    return findMoveEnergyCluster(eG, cluster) + findCommEnergyCluster(eG, cluster)


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



def mergeClusters(listOfClusters):

    mdc_count = 5
    numClusters = mdc_count - 1
    k = numClusters

    mergedCluster = list()
    finalCluster = list()

    sum = 0
    round = 0
    lowest = 0

    #do/while
    while True:
        for clust_x in listOfClusters:
            for clust_y in listOfClusters:

                sum = totalEnergyCluster(eG, clust_x) + totalEnergyCluster(eG, clust_y)

              #  print("Energy of ", clust_x, " and ", clust_y, ": ", sum)
                # make sure they aren't the same
                if clust_x == clust_y:
                    continue
                if round == 0:
                    lowest = sum
                    mergedCluster.append(clust_x)
                    mergedCluster.append(clust_y)
                else:
                    if sum < lowest:
                        lowest = sum
                        # Clear out merged clusters
                        mergedCluster.clear()
                        mergedCluster.append(clust_x)
                        mergedCluster.append(clust_y)
            round = round + 1
        k = k-1
        if k == 0:
            break
    
    return mergedCluster

def printClusters(cluster_list):

    i = 0
    for clust in cluster_list:
        print("Cluster", i, ":", clust)
        i += 1
        
    return True


### phase2 code here

def getAvgEnergyCluster(eG, cluster):


    return True

def getStanDevCluster(eG, cluster):


    return True


def getCoreOfMass(eG, cluster):


    return True


def moveRPIn(cluster):


    return True


def moveRPOut(cluster):


    return True



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
   # cluster.append(eG)
    cluster.append(node)
    # append eG to each cluster per the paper
    singleClusterList.append(cluster)

## Phase 1 results should be
# MDC Count = 5
# Cluster 0 = S0, S1  - Energy: 320
# Cluster 1 = S2, S9, S10, S11 - Energy:  526
# Cluster 2 = S6, S7, S8 - Energy: 470
# Cluster 3 = S4, S5 - Energy:  330
#
# Cluster 4 = S3 ## eG


### DEBUGGING

clust0 = list()
clust1 = list()
clust2 = list()
clust3 = list()
clust4 = list()

clust0.append(S0)
clust0.append(S1)

clust1.append(S2)
clust1.append(S9)
clust1.append(S10)
clust1.append(S11)

clust2.append(S6)
clust2.append(S7)
clust2.append(S8)

clust3.append(S4)
clust3.append(S5)

clust4.append(S3)

####


listOfClusters = list()
cenClust = list()
cenClust.append(S3)

#printClusters(singleClusterList)
print(mergeClusters(singleClusterList))


          
        



          
        
