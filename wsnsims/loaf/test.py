import os
import sys
import math


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
    Cx = round(xweight / totalData)
    Cy = round(yweight / totalData)

    eGs.append(Cx)
    eGs.append(Cy)

    # coordinates for center of energy 
    return eGs

# Finds the euclidian distance between a node and eG
def findEucDist(node, eG):

    x = node[0]
    y = node[1]
    
    eG_x = eG[0]
    eG_y = eG[1]

    
    temp1 = (x - eG_x) ** 2
    temp2 = (y - eG_y) ** 2

    distance = round(math.sqrt(temp1 + temp2))
            
    
    return distance

# initialize and form segments into clusters
def initClusters(segments):

    start_clusters = list()
    # Loop through segments 

    for seg in segments:
        start_clusters.append(seg)
        
    return start_clusters


# Find the non-essential clusters
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




# main code here

# starting out, center cluster is at eG
eG = findEG(nodes)

center_cluster_coord = eG

# initialize clusters, each segment is in it's own cluster

startClusters = initClusters(nodes)

# Central cluster starts at eG, which is eG

centralCluster = list()
centralCluster.append(S3)

# Remove from our node list
nodes.remove(S3)

nonEssClust = list()

# Add each node to it's own cluster
# add cluster to list of non-essential clusters
for node in nodes:
    cluster = list()
    cluster.append(node)
    nonEssClust.append(cluster)

# Now we need to merge clusters

toMerge = list()
total = 0
prev = 0
round = 0

# Clusters -> cluster -> node
# list -> List -> list

mergedClust = list()
# Loop through nodes
# determine which two will be grouped into a cluster
for outer in nodes:
    for inner in nodes:
        if inner == outer:
            continue
        else:
            outerWeight = outer[2]
            innerWeight = inner[2]

            if round == 0:
                round += 1
                prev = outerWeight + innerWeight
                continue
            else:
                total = outerWeight + innerWeight
                if total < prev:
                    toMerge.clear()
                    prev = total
                    print("Prev: ", prev, "Total: ", total)
                    toMerge.append(outer)
                    toMerge.append(inner)

                    cluster = list()
                    for m in toMerge:
                        cluster.append(m)
                        nodes.remove(m)
                        mergedClust.append(cluster)
                    round += 1
    
            # We now have a list of clusters to merge, we we assign them to a new cluster
                    

                        
            
        
        
        



          
        
