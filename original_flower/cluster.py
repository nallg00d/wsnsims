import itertools
import logging

from original_flower import data
from original_flower import params
from original_flower import tour
from original_flower.point import WorldPositionMixin


class ClusterError(Exception):
    pass


class ClusterMixin(WorldPositionMixin):
    def __init__(self):
        WorldPositionMixin.__init__(self)

        self.cluster_id = params.NOT_CLUSTERED
        self._relay_node = None
        self._nodes = list()
        self._tour = list()
        self._tour_length = 0

    def calculate_tour(self):
        if not self._nodes:
            self._tour = []
            self._tour_length = 0
        else:
            self._tour = tour.find_tour(list(self._nodes), radius=0)
            self._tour_length = tour.tour_length(self._tour)

    def update_location(self):
        if self._nodes:
            com = tour.centroid(self._nodes)
            self.x = com.x
            self.y = com.y

    @property
    def tour_length(self):
        return self._tour_length

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value
        self.update_location()

    # def append(self, *args):
    #     self._nodes.append(args)
    #     self.update_location()

    def add(self, node):
        if node not in self._nodes:
            logging.debug("Adding %s to %s", node, self)
            self._nodes.append(node)
            self.update_location()
        else:
            logging.warning("Re-added %s to %s", node, self)

    def remove(self, node):
        # if node not in self._nodes:
        #     logging.warning("Invalid attempt to remove %s from %s %s", node, self, self._nodes)
        #     raise ClusterError()
        # else:

        logging.debug("Removing %s from %s", node, self)
        self._nodes.remove(node)
        self.update_location()

    def tour(self):
        return [n.collection_point for n in self._tour]

    def tour_nodes(self):
        return self._tour

    def motion_energy(self):
        cost = params.MOVEMENT_COST * self.tour_length
        return cost

    def data_volume_bits(self, all_clusters, all_nodes):
        # Volume in megabits
        data_volume = self.data_volume_mbits(all_clusters, all_nodes)

        # Volume in bits
        data_volume *= 1024 * 1024
        return data_volume

    def data_volume_mbits(self, all_clusters, all_nodes):
        raise NotImplementedError()

    def communication_energy(self, all_clusters, all_nodes):
        # Volume in bits
        data_volume = self.data_volume_bits(all_clusters, all_nodes)

        e_c = data_volume * (params.ALPHA + params.BETA * pow(params.COMMUNICATION_RANGE, params.DELTA))
        # e_c = data_volume * 2.0 * pow(10, -6)

        return e_c

    def total_energy(self, all_clusters, all_nodes):
        total = self.motion_energy() + self.communication_energy(all_clusters, all_nodes)
        return total

    def combined(self, other):
        if not self._nodes:
            return 0

        new_cluster = type(self)()
        new_cluster.nodes = list(set(self.nodes + other.nodes))
        return new_cluster


class FlowerCluster(ClusterMixin):
    def __init__(self):
        super(FlowerCluster, self).__init__()

        self.completed = False
        self.anchor = None
        self.central_cluster = None
        self._recent = None

    def __str__(self):
        return "Flower Cluster {}".format(self.cluster_id)

    def __repr__(self):
        return "FC{}".format(self.cluster_id)

    @property
    def cells(self):
        return self.nodes

    @cells.setter
    def cells(self, value):
        self.nodes = value

    @property
    def recent(self):
        return self._recent

    def add(self, cell):
        super(FlowerCluster, self).add(cell)
        cell.cluster = self
        cell.cluster_id = self.cluster_id

    def remove(self, cell):
        super(FlowerCluster, self).remove(cell)
        cell.cluster = None
        cell.cluster_id = params.NOT_CLUSTERED

    @recent.setter
    def recent(self, value):
        logging.debug("Setting RC(%r) to %r", self, value)
        self._recent = value

        if value not in self.nodes:
            logging.warning("%r is not a member of %r", value, self)

    def calculate_tour(self):
        if not self.cells:
            self._tour = []
            self._tour_length = 0
            return

        if len(self.cells) == 1:
            cells = self.cells + self.central_cluster.cells
        elif self.anchor:
            cells = self.cells + [self.anchor]
        else:
            cells = list(self.cells)

        self._tour = tour.find_tour(cells, radius=0)
        self._tour_length = tour.tour_length(self._tour)

    def update_anchor(self):
        # First, remove any cells that don't have our cluster id
        # new_cells = [c for c in self.cells if c.cluster_id == self.cluster_id]
        # if not new_cells:
        #     return

        if not self.cells:
            self.anchor = None
            return

        # Now, find the closest cell in the central cluster, and add it to ourselves
        _, anchor = closest_nodes(self, self.central_cluster, cell_distance=False)
        # new_cells.append(anchor)
        # self.cells = new_cells

        # self.central_cluster.anchors[self] = anchor
        self.anchor = anchor

    def combined(self, other):
        new_cluster = super(FlowerCluster, self).combined(other)
        new_cluster.central_cluster = self.central_cluster
        return new_cluster

    def data_volume_mbits(self, all_clusters, all_cells):
        if not self.cells:
            return 0

        # Handle all intra-cluster data
        cluster_cells = [c for c in all_cells if c.cluster_id == self.cluster_id]
        cluster_segs = [seg for segs in [c.segments for c in cluster_cells] for seg in segs]
        intracluster_seg_pairs = [(src, dst) for src in cluster_segs for dst in cluster_segs if src != dst]
        data_vol = sum([data.data(src, dst) for src, dst in intracluster_seg_pairs])

        # Handle inter-cluster data at the anchor
        other_cells = [c for c in all_cells if c.cluster_id != self.cluster_id]
        other_segs = [seg for segs in [c.segments for c in other_cells] for seg in segs]
        intercluster_seg_pairs = [(src, dst) for src in cluster_segs for dst in other_segs]
        intercluster_seg_pairs += [(src, dst) for src in other_segs for dst in cluster_segs]

        # data volume for inter-cluster traffic
        data_vol += sum(data.data(src, dst) for src, dst in intercluster_seg_pairs)

        return data_vol


class FlowerVirtualCluster(FlowerCluster):
    def __init__(self):
        super(FlowerVirtualCluster, self).__init__()

        self.virtual_cluster_id = params.NOT_CLUSTERED

    def __str__(self):
        return "Flower Virtual Cluster {}".format(self.virtual_cluster_id)

    def __repr__(self):
        return "FVC{}".format(self.virtual_cluster_id)

    def calculate_tour(self):
        if not self.cells:
            self._tour = []
            self._tour_length = 0
            return

        cells = self.cells + self.central_cluster.cells
        self._tour = tour.find_tour(cells, radius=0)
        self._tour_length = tour.tour_length(self._tour)

    def data_volume_mbits(self, all_clusters, all_cells):
        if not self.cells:
            return 0

        # Handle all intra-cluster data
        cluster_cells = [c for c in all_cells if c.virtual_cluster_id == self.virtual_cluster_id]
        cluster_segs = [seg for segs in [c.segments for c in cluster_cells] for seg in segs]
        intracluster_seg_pairs = [(src, dst) for src in cluster_segs for dst in cluster_segs if src != dst]
        data_vol = sum([data.data(src, dst) for src, dst in intracluster_seg_pairs])

        # Handle inter-cluster data at the anchor
        other_cells = [c for c in all_cells if c.virtual_cluster_id != self.virtual_cluster_id]
        other_segs = [seg for segs in [c.segments for c in other_cells] for seg in segs]
        intercluster_seg_pairs = [(src, dst) for src in cluster_segs for dst in other_segs]
        intercluster_seg_pairs += [(src, dst) for src in other_segs for dst in cluster_segs]

        # data volume for inter-cluster traffic
        data_vol += sum(data.data(src, dst) for src, dst in intercluster_seg_pairs)

        return data_vol


class FlowerHub(FlowerCluster):
    def __init__(self):
        super(FlowerHub, self).__init__()
        # self.anchors = {}

    def calculate_tour(self):
        # cells = self.cells + list(self.anchors.values())
        cells = list(self.cells)
        self._tour = tour.find_tour(cells, radius=0)
        self._tour_length = tour.tour_length(self._tour)

    def __str__(self):
        return "Flower Hub Cluster"

    def __repr__(self):
        return "FH"

    def data_volume_mbits(self, all_clusters, all_cells):
        if not self.cells:
            return 0

        # Handle all intra-cluster data for the hub
        hub_cells = [c for c in all_cells if c.cluster_id == self.cluster_id]
        hub_segs = [seg for segs in [c.segments for c in hub_cells] for seg in segs]
        hub_seg_pairs = [(src, dst) for src in hub_segs for dst in hub_segs if src != dst]
        data_vol = sum([data.data(src, dst) for src, dst in hub_seg_pairs])

        # Handle inter-cluster data for other clusters
        anchor_cells = [a for a in [c.anchor for c in all_clusters if c.cluster_id != self.cluster_id]]
        anchor_cells = list(set(anchor_cells))
        for cell in anchor_cells:
            # Get the segments served by this anchor
            local_clusters = [c for c in all_clusters if c.anchor == cell]
            local_cells = [c for c in all_cells if c.cluster_id in [clust.cluster_id for clust in local_clusters]]
            local_segs = [seg for segs in [c.segments for c in local_cells] for seg in segs]

            # Get the segments not served by this anchor
            remote_clusters = [c for c in all_clusters if c.anchor != cell and c != self]
            remote_cells = [c for c in all_cells if c.cluster_id in [clust.cluster_id for clust in remote_clusters]]
            remote_segs = [seg for segs in [c.segments for c in remote_cells] for seg in segs]

            # Generate the pairs of local-to-remote segments
            seg_pairs = [(seg_1, seg_2) for seg_1 in local_segs for seg_2 in remote_segs]

            # Generate the pairs of remote-to-local segments
            seg_pairs += [(seg_1, seg_2) for seg_1 in remote_segs for seg_2 in local_segs]

            # Add the inter-cluster data volume
            data_vol += sum(data.data(src, dst) for src, dst in seg_pairs)

        # Handle inter-cluster data for the hub itself
        # This is done by the above loop

        return data_vol


class FlowerVirtualHub(FlowerVirtualCluster):
    def calculate_tour(self):
        cells = list(self.cells)

        self._tour = tour.find_tour(cells, radius=0)
        self._tour_length = tour.tour_length(self._tour)

    def __str__(self):
        return "Flower Virtual Hub Cluster"

    def __repr__(self):
        return "FVH"

    def data_volume_mbits(self, all_clusters, all_cells):
        return 0


class ToCSCluster(ClusterMixin):
    def __init__(self):
        super(ToCSCluster, self).__init__()

        self.rendezvous_point = None

    @property
    def segments(self):
        return self._nodes

    @segments.setter
    def segments(self, value):
        self.nodes = value

    def calculate_tour(self):
        cells = list(self.nodes)

        if self.rendezvous_point:
            cells.append(self.rendezvous_point)
            self._tour = tour.find_tour(cells, radius=params.COMMUNICATION_RANGE, start=self.rendezvous_point)
        else:
            self._tour = tour.find_tour(cells, radius=params.COMMUNICATION_RANGE)

        self._tour_length = tour.tour_length(self._tour)

    def add(self, segment):
        super(ToCSCluster, self).add(segment)
        segment.cluster = self

    def remove(self, segment):
        super(ToCSCluster, self).remove(segment)
        segment.cluster = None

    def data_volume_mbits(self, all_clusters, all_segments):
        if not self.segments:
            return 0

        # Handle all intra-cluster data
        cluster_segs = self.segments
        intracluster_seg_pairs = [(src, dst) for src in cluster_segs for dst in cluster_segs if src != dst]
        data_vol = sum([data.data(src, dst) for src, dst in intracluster_seg_pairs])

        # Handle inter-cluster data at the rendezvous point
        other_segs = [c for c in all_segments if c.cluster != self]
        intercluster_seg_pairs = [(src, dst) for src in cluster_segs for dst in other_segs]
        intercluster_seg_pairs += [(src, dst) for src in other_segs for dst in cluster_segs]

        # data volume for inter-cluster traffic
        data_vol += sum(data.data(src, dst) for src, dst in intercluster_seg_pairs)

        return data_vol

    def __str__(self):
        return "ToCS Cluster {}".format(self.cluster_id)

    def __repr__(self):
        return "TC{}".format(self.cluster_id)

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


class ToCSCentroid(ToCSCluster):
    def __init__(self, position):
        super(ToCSCentroid, self).__init__()
        self.virtual_center = position
        self.x = position.x
        self.y = position.y
        self.rendezvous_points = {}

    def calculate_tour(self):
        cells = list(self.rendezvous_points.values()) + self.segments
        self._tour = tour.find_tour(cells, radius=params.COMMUNICATION_RANGE)
        self._tour_length = tour.tour_length(self._tour)

    def add(self, segment):
        super(ToCSCentroid, self).add(segment)
        segment.cluster = self

    def remove(self, segment):
        super(ToCSCentroid, self).remove(segment)
        segment.cluster = None

    def data_volume_mbits(self, all_clusters, all_segments):

        # Handle all intra-cluster data for the hub
        if self.segments:
            hub_segs = self.segments
            hub_seg_pairs = [(src, dst) for src in hub_segs for dst in hub_segs if src != dst]
            data_vol = sum([data.data(src, dst) for src, dst in hub_seg_pairs])
        else:
            data_vol = 0

        # Handle inter-cluster data for other clusters
        for clust, _ in self.rendezvous_points.items():
            local_segs = clust.segments
            remote_segs = [seg for seg in all_segments if seg.cluster != clust]

            # Generate the pairs of local-to-remote segments
            seg_pairs = [(seg_1, seg_2) for seg_1 in local_segs for seg_2 in remote_segs]

            # Generate the pairs of remote-to-local segments
            seg_pairs += [(seg_1, seg_2) for seg_1 in remote_segs for seg_2 in local_segs]

            # Add the inter-cluster data volume
            data_vol += sum(data.data(src, dst) for src, dst in seg_pairs)

        # Handle inter-cluster data for the hub itself
        # This is done by the above loop

        return data_vol

    def __str__(self):
        return "ToCS Centroid"

    def __repr__(self):
        return "TCentroid"


def combine_clusters(clusters, centroid):
    index = 0
    decorated = list()

    cluster_pairs = [(c1, c2) for c1 in clusters for c2 in clusters if c1 != c2]
    for c_i, c_j in cluster_pairs:
        temp_cluster_1 = c_i.combined(c_j).combined(centroid)
        temp_cluster_1.calculate_tour()

        temp_cluster_2 = c_i.combined(centroid)
        temp_cluster_2.calculate_tour()

        combination_cost = temp_cluster_1.tour_length - temp_cluster_2.tour_length
        decorated.append((combination_cost, index, c_i, c_j))
        index += 1

    cost, _, c_i, c_j = min(decorated)
    logging.info("Combining %s and %s (Cost: %f)", c_i, c_j, cost)

    new_clusters = list(clusters)
    new_cluster = c_i.combined(c_j)
    new_cluster.calculate_tour()

    for node in new_cluster.nodes:
        node.cluster = new_cluster

    new_clusters.remove(c_i)
    new_clusters.remove(c_j)
    new_clusters.append(new_cluster)
    return new_clusters


def closest_nodes(cluster_1, cluster_2, cell_distance=True):
    if isinstance(cluster_1, ClusterMixin):
        node_list_1 = cluster_1.nodes
    else:
        node_list_1 = cluster_1

    if isinstance(cluster_2, ClusterMixin):
        node_list_2 = cluster_2.nodes
    else:
        node_list_2 = cluster_2

    pairs = itertools.product(node_list_1, node_list_2)

    if cell_distance:
        decorated = [(cell_1.cell_distance(cell_2), i, cell_1, cell_2) for i, (cell_1, cell_2) in enumerate(pairs)]
    else:
        decorated = [(cell_1.distance(cell_2), i, cell_1, cell_2) for i, (cell_1, cell_2) in enumerate(pairs)]

    closest = min(decorated)
    cells = closest[2], closest[3]
    return cells


def closest_points(points_1, points_2):
    pairs = itertools.product(points_1, points_2)
    decorated = [(point_1.distance(point_2), i, point_1, point_2) for i, (point_1, point_2) in enumerate(pairs)]
    closest = min(decorated)
    points = closest[2], closest[3]
    return points