import sys
import random
import traceback
import math
import copy

"""
KMeansClusterer.py - an interface for the Model AI Assignments k-Means Clustering exercises, based on the Java class written by Todd Neller
"""

class KMeansClusterer:
    def __init__(self, kMin=2, kMax=2, iterations=1):
        """Construct a KMeansClusterer object.
        
        Keyword arguments:
        kMin -- the minimum k value (default 2)
        kMax -- the maximum k value (default 2)
        iterations -- the number of iterations of the algorithm to execute (default 1)
        """
        self.dim = 2            # The number of dimensions in the data
        self.k = kMin           # The allowable range of the clusters (kMin to kMax)
        self.kMin = kMin
        self.kMax = kMax
        self.iter = iterations  # The number of k-Means Clustering iterations per k
        self.data = None        # The data vectors for clustering -- Should eventually be set to a 2-D list of floats by setData()
        self.centroids = None   # The cluster centroids -- Should eventually be set to a 2-D list of floats by computeNewCentroids()
        self.clusters = None    # Assigned clusters for each data point -- Should eventually be set to a list of ints by assignNewClusters()

    def readData(self, filename):
        """Read the specified data input format from the given file and return a 2-D list of floats, with each row being a data point and each column being a dimension of the data.

        Arguments:
        filename -- A string representing the file path (may be relative or global)

        Returns: A 2-D list of floats 
        """
        numPoints = 0
        try:
            with open(filename) as file:
                try:
                    self.dim = int(file.readline().split()[1])
                    numPoints = int(file.readline().split()[1])
                except Exception:
                    print('Invalid data file format. Exiting.')
                    traceback.print_exception()
                    sys.exit(1)

                data = [[0]*self.dim for _ in range(numPoints)]
                for i in range(numPoints):
                    line = file.readline()
                    values = line.split()
                    for j in range(self.dim):
                        data[i][j] = float(values[j])

            return data

        except Exception as e:
            print('Could not locate source file. Exiting.')
            traceback.print_exception()
            sys.exit(1)

    def setData(self, data):
        """Set the given data as the clustering data as a 2-D list of floats with each row being a data point and each column being a dimension of the data.

        Arguments:
        data -- The given clustering data
        """
        self.data = data
        self.dim = len(data[0])

    def getData(self):
        """Return the clustering data as a 2-D list of floats with each row being a data point and each column being a dimension of the data.
        
        Returns: A 2-D list of floats containing the clustering data
        """
        return self.data

    def getDim(self):
        """Return the number of dimensions of the clustering data
        
        Returns: The integer number of dimensions of the clustering data
        """
        return self.dim

    def setKRange(self, kMin, kMax):
        """Set the minimum and maximum allowable number of clusters k. If a single given k is to be used, then kMin == kMax. If kMin < kMax, then all k from kMin to kMax
        inclusive will be compared using the gap statistic. The minimum WCSS run of the k with the maximum gap will be the result.
        
        Arguments:
        kMin -- Minimum number of clusters
        kMax -- Maximum number of clusters
        """
        self.kMin = kMin
        self.kMax = kMax
        self.k = kMin

    def getK(self):
        """Return the number of clusters k. After calling kMeansCluster() with a range from kMin to kMax, this value will be the k yielding the maximum gap statistic.
        
        Returns: the number of clusters k (int)
        """
        return self.k

    def setIter(self, it):
        """Set the number of iterations to perform k-Means Clustering and choose the minimum WCSS result.
        
        Arguments:
        it -- The number of iterations to perform k-Means Clustering
        """
        self.iter = it
    
    def getCentroids(self):
        """Return the 2-D list of centroids indexed by cluster number and centroid dimension

        Returns: The 2-D list of centroids indexed by cluster number and centroid dimension
        """
        return self.centroids

    def getClusters(self):
        """Return a parallel list of cluster assignments such that self.data[i] belongs to the cluster self.clusters[i] with centroid self.centroids[self.clusters[i]].
        
        Returns: A parallel list of cluster assignments
        """
        return self.clusters
    
    def getDataMinimums(self):
        num_dimensions = len(self.data[0])
        minimums = [float('inf')] * num_dimensions

        for row in self.data:
            for i in range(num_dimensions):
                if row[i] < minimums[i]:
                    minimums[i] = row[i]

        return minimums


    def getDataMaximums(self):
        num_dimensions = len(self.data[0])
        maximums = [float('-inf')] * num_dimensions

        for row in self.data:
            for i in range(num_dimensions):
                if row[i] > maximums[i]:
                    maximums[i] = row[i]

        return maximums
    

    def getDistance(self, p1, p2):
        """Return the Euclidean distance between the two given point vectors.

        Arguments:
        p1 -- Point vector 1
        p2 -- Point vector 2

        Returns: The Euclidean distance between the two given point vectors (float)
        """
        sumOfSquareDiffs = 0
        for i in range(len(p1)):
            diff = p1[i] - p2[i]
            sumOfSquareDiffs += (diff * diff)
        
        return math.sqrt(sumOfSquareDiffs)

    def getWCSS(self):
        """Return the minimum Within-Clusters Sum-of-Squares measure for the chosen k number of clusters.

        Returns: the minimum Within-Clusters Sum-of-Squares measure (float)
        """
        wcss = 0.0
        for i, point in enumerate(self.data):
            cluster_id = self.clusters[i]
            centroid = self.centroids[cluster_id]
            dist = self.getDistance(point, centroid)
            wcss += dist * dist
        return wcss

    def assignNewClusters(self, data) -> bool:
        """Assign each data point to the nearest centroid and return whether or not any cluster assignments changed.

        Returns: Whether or not any cluster assignments changed (bool)
        """
        changed = False
        new_clusters = []

        for point in data:
            # Compute distance to each centroid
            distances = [self.getDistance(point, centroid) for centroid in self.centroids]
            # Find the index of the closest centroid
            closest_cluster = distances.index(min(distances))
            new_clusters.append(closest_cluster)

        # Check for changes
        for i in range(len(data)):
            if self.clusters[i] != new_clusters[i]: # Note: self.clusters[i] may initially be None; comparison still works safely
                changed = True
                break  # Stop early if any assignment changed

        self.clusters = new_clusters
        return changed

    def computeNewCentroids(self, k, data):
        """Compute new centroids at the mean point of each cluster of points."""
        dim = len(data[0])
        # Initialize lists to hold sums and counts
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k

        # Sum up coordinates for each cluster
        for i, cluster_id in enumerate(self.clusters):
            for d in range(dim):
                sums[cluster_id][d] += data[i][d]
            counts[cluster_id] += 1

        # Calculate mean for each cluster
        self.centroids = []
        for i in range(k):
            if counts[i] == 0:
                # Handle empty cluster — reinitialize randomly
                self.centroids.append(random.choice(data))
            else:
                mean = [s / counts[i] for s in sums[i]]
                self.centroids.append(mean)

    def computeSampleData(self, minimums, maximums, dimensions, numPoints):
        dataSet = []
        for i in range(numPoints):
            currentPoint = []
            for i in range(dimensions):
                currentPoint.append(random.uniform(minimums[i], maximums[i] + 1))
            dataSet.append(currentPoint)
        return dataSet

    def kMeansCluster(self):
        """Perform k-means clustering with Forgy initialization and return the 0-based cluster assignments for corresponding data points.
        If self.iter > 1, choose the clustering that minimizes the WCSS measure.
        If kMin < kMax, select the k maximizing the gap statistic using 100 uniform samples uniformly across given data ranges.
        """
        if self.kMin == self.kMax: # If there is only one k provided

            self.kMeansSingleK(self.kMin, self.data)

        else:

            gapK = float('-inf')
            bestK = None
            bestClusters = None
            bestCentroids = None

            # Data info
            minimums = self.getDataMinimums()
            maximums = self.getDataMaximums()
            dimensions = self.dim
            numPoints = len(self.data)
            
            for k in range(self.kMin, self.kMax+1, 1):

                # I like to uncomment this so I can see that the algorithm is running
                # print(f"k: {k}")

                self.kMeansSingleK(k, self.data)

                logMinWCSS = math.log(self.getWCSS())
                associatedClusters = copy.deepcopy(self.clusters)
                associatedCentroids = copy.deepcopy(self.centroids)

                totalLogRandWCSS = 0
                for i in range(100):
                    # I like to uncomment this so I can see that the algorithm is running
                    # print(f"    i: {i}")
                    data = self.computeSampleData(minimums, maximums, dimensions, numPoints)
                    self.kMeansOneIter(k, data)
                    totalLogRandWCSS += math.log(self.getWCSS())

                avgRandWCSS = totalLogRandWCSS / 100
                newGapK = avgRandWCSS - logMinWCSS

                if gapK < newGapK:
                    gapK = newGapK
                    bestK = k
                    bestClusters = associatedClusters
                    bestCentroids = associatedCentroids

            self.k = bestK
            self.clusters = bestClusters
            self.centroids = bestCentroids
            


    def kMeansSingleK(self, k, data):

        old_wcss = float('inf')

        for i in range(self.iter):
            
            self.kMeansOneIter(k, data)

            # wcss comparison
            new_wcss = self.getWCSS()
            if new_wcss < old_wcss:
                best_clusters = copy.deepcopy(self.clusters)
                best_centroids = copy.deepcopy(self.centroids)
                old_wcss = new_wcss

        self.clusters = best_clusters
        self.centroids = best_centroids


    def kMeansOneIter(self, k ,data):
        # Choose k centroids using Forgy initialization (random sample from data)
        self.centroids = random.sample(data, k)

        # Initialize cluster assignments
        self.clusters = [None for _ in range(len(data))]

        still_changing = True
        while still_changing:
            still_changing = self.assignNewClusters(data)  # Assign points and check for changes
            self.computeNewCentroids(k, data)                # Move centroids to the mean of their clusters

    def writeClusterData(self, filename):
        """Export cluster data in the given data output format to the file provided.

        Arguments:
        filename -- A string representing the file path (may be relative or global)
        """
        try:
            with open(filename, 'w') as file:
                file.write('%% '+str(self.dim)+' dimensions\n')
                file.write('%% '+str(len(self.data))+' points\n')
                file.write('%% '+str(self.k)+' clusters/centroids\n')
                file.write('%% '+str(self.getWCSS())+' within-cluster sum of squares\n')

                for i in range(self.k):
                    file.write(str(i) + ' ')
                    for j in range(self.dim):
                        endtoken = ' ' if j < (self.dim - 1) else '\n'
                        file.write(str(self.centroids[i][j])+endtoken)
                
                for i in range(len(self.data)):
                    file.write(str(self.clusters[i]) + ' ')
                    for j in range(self.dim):
                        endtoken = ' ' if j < (self.dim - 1) else '\n'
                        file.write(str(self.data[i][j])+endtoken)

        except Exception:
            print('Error writing to file')
            traceback.print_exception()
            sys.exit(1)
    

def main():
    """Read UNIX-style command line parameters as to specify the type of k-Means clustering algorithm applied to the formatted data specified.
    "-k int" specifies both the minimum and maximum number of clusters. "-kmin int" specifies the minimum number of clusters. "-kmax int" specifies the maximum number of clusters.
    "-iter int" specifies the number of times k-Means Clustering is performed in iteration to find a lower local minimum.
    "-in filename" specifies the source file for input data. "-out filename" specifies the destination file for cluster data.
    """
    kMin = 2
    kMax = 2
    iterations = 1
    attributes = []
    values = []
    i = 1
    infile = ''
    outfile = ''
    while i < len(sys.argv):
        if sys.argv[i] == "-k" or sys.argv[i] == "-kmin" or sys.argv[i] == "-kmax" or sys.argv[i] == '-iter':
            attributes.append(sys.argv[i][1:])
            i += 1
                
            if i == len(sys.argv):
                print('No integer value for '+str(attributes[-1])+".")
                sys.exit(1)
            
            try:
                values.append(int(sys.argv[i]))
                i += 1
            except Exception:
                print("Error parsing "+str(sys.argv[i])+" as an integer.")
                sys.exit(2)
        elif sys.argv[i] == "-in":
            i += 1
            if i == len(sys.argv):
                print('No string vlaue provided for input source.')
                sys.exit(1)
            infile = sys.argv[i]
            i += 1
        elif sys.argv[i] == "-out":
            i += 1
            if i == len(sys.argv):
                print('No string value provided for output source.')
                sys.exit(1)
            outfile = sys.argv[i]
            i += 1

    for i in range(len(attributes)):
        attribute = attributes[i]
        if attribute == "k":
            kMin = values[i]
            kMax = values[i]
        elif attribute == "kmin":
            kMin = values[i]
        elif attribute == "kmax":
            kMax = values[i]
        elif attribute == "iter":
            iterations = values[i]
    
    km = KMeansClusterer(kMin, kMax, iterations)
    km.setData(km.readData(infile))
    km.kMeansCluster()
    km.writeClusterData(outfile)


if __name__ == "__main__":
    main()