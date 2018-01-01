import logging

from wsnsims.core.cluster import BaseCluster

logger = logging.getLogger(__name__)


class LoafCluster(BaseCluster):
    def __init__(self, environment):
        """

        :param environment:
        :type environment: core.environment.Environment
        """
        super(LoafCluster, self).__init__(environment)

        self.completed = False
        self.recent = None

    @property
    def cluster_id(self):
        return super(LoafCluster, self).cluster_id

    @cluster_id.setter
    def cluster_id(self, value):
        self._cluster_id = value
        for cell in self.cells:
            cell.cluster_id = self.cluster_id

    @property
    def cells(self):
        return self.nodes

    @cells.setter
    def cells(self, value):
        self.nodes = value

    @property
    def segments(self):
        cluster_segments = list()
        for cell in self.cells:
            cluster_segments.extend(cell.segments)

        cluster_segments = list(set(cluster_segments))
        return cluster_segments

    @property
    def anchor(self):
        return self.relay_node

    @anchor.setter
    def anchor(self, value):
        self.relay_node = value

    def add(self, cell):
        """

        :param cell:
        :type cell: loaf.cell.Cell
        :return:
        """
        super(LoafCluster, self).add(cell)
        self.recent = cell

    def remove(self, cell):
        """

        :param cell:
        :type cell: loaf.cell.Cell
        :return:
        """
        super(LoafCluster, self).remove(cell)
        if cell == self.recent:
            self.recent = None

    def __str__(self):
        return "Loaf Cluster {}".format(self.cluster_id)

    def __repr__(self):
        return "FC{}".format(self.cluster_id)


class LoafVirtualCluster(LoafCluster):
    def __init__(self, environment):
        """

        :param environment:
        :type environment: core.environment.Environment
        """
        super(LoafVirtualCluster, self).__init__(environment)

    def __str__(self):
        return "Loaf Virtual Cluster {}".format(self.cluster_id)

    def __repr__(self):
        return "FVC{}".format(self.cluster_id)

    @property
    def cluster_id(self):
        return super(LoafVirtualCluster, self).cluster_id

    @cluster_id.setter
    def cluster_id(self, value):
        self._cluster_id = value
        for cell in self.cells:
            cell.virtual_cluster_id = self.cluster_id


class LoafHub(LoafCluster):
    def __init__(self, environment):
        """

        :param environment:
        :type environment: core.environment.Environment
        """
        super(LoafHub, self).__init__(environment)

    def __str__(self):
        return "Loaf Hub Cluster"

    def __repr__(self):
        return "FH"


class LoafVirtualHub(LoafVirtualCluster):
    def __init__(self, environment):
        """

        :param environment:
        :type environment: core.environment.Environment
        """
        super(LoafVirtualHub, self).__init__(environment)

    def __str__(self):
        return "Loaf Virtual Hub Cluster"

    def __repr__(self):
        return "FVH"
