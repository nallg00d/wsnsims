import logging

import numpy as np

from wsnsims.loaf.energy import LOAFEnergyModel
from wsnsims.loaf.movement import LOAFMovementModel

from wsnsims.loaf import data

logger = logging.getLogger(__name__)


class LOAFRunnerError(Exception):
    pass


class LOAFRunner(object):
    def __init__(self, sim, environment):
        """

        :param sim: The simulation after a run of ToCS
        :type sim: loaf.loaf_sim.LOAF
        :param environment:
        :type environment: core.environment.Environment
        """

        self.sim = sim
        self.env = environment

        self.movement_model = LOAFMovementModel(self.sim, self.env)
        self.energy_model = LOAFEnergyModel(self.sim, self.env)

    def maximum_communication_delay(self):
        """
        Compute the average communication delay across all segments.

        :return: The delay time in seconds
        :rtype: pq.quantity.Quantity
        """

        cell_pairs = ((src, dst) for src in self.sim.cells for dst in
                      self.sim.cells if src != dst)

        delays = []
        for src, dst in cell_pairs:
            delay = self.communication_delay(src, dst)
            delays.append(delay)

        delays = np.array(delays)
        max_delay = np.max(delays)
        # max_delay *= pq.second

        return max_delay

    def cell_cluster(self, cell):
        for cluster in [self.sim.hub] + self.sim.clusters:
            if cell in cluster.cells:
                return cluster

        raise LOAFRunnerError("No cluster found for %s", cell)

    def communication_delay(self, begin, end):
        """
        Compute the communication delay between any two segments. This is done
        as per Equation 1 in LOAF.

        :param begin:
        :type begin: loaf.cell.Cell
        :param end:
        :type end: loaf.cell.Cell

        :return: The total communication delay in seconds
        :rtype: pq.second
        """

        travel_delay, path = self.movement_model.shortest_distance(begin, end)
        travel_delay /= self.env.mdc_speed

        begin_cluster = self.cell_cluster(begin)
        end_cluster = self.cell_cluster(end)

        if begin_cluster == end_cluster:
            transmission_count = 1
        elif begin_cluster == self.sim.hub or end_cluster == self.sim.hub:
            transmission_count = 2
        else:
            transmission_count = 3

        transmission_delay = transmission_count
        transmission_delay *= data.cell_volume(begin, end, self.env)
        transmission_delay /= self.env.comms_rate

        relay_delay = self.holding_time(begin, end)

        total_delay = travel_delay + transmission_delay + relay_delay
        return total_delay

    def holding_time(self, begin, end):
        """

        :param begin:
        :type begin: loaf.cell.Cell
        :param end:
        :type end: loaf.cell.Cell
        :return:
        :rtype: pq.second
        """

        begin_cluster = self.cell_cluster(begin)
        end_cluster = self.cell_cluster(end)

        if begin_cluster == end_cluster:
            return 0.  # * pq.second

        if begin_cluster.anchor == end_cluster.anchor:
            delay = self.tour_time(end_cluster)
            return delay

        elif begin_cluster == self.sim.hub:
            delay = self.tour_time(end_cluster)
            return delay

        elif end_cluster == self.sim.hub:
            delay = self.tour_time(end_cluster)
            return delay

        else:
            delay = self.tour_time(self.sim.hub)
            delay += self.tour_time(end_cluster)
            return delay

    def tour_time(self, cluster):
        """

        :param cluster:
        :type cluster: loaf.cluster.LOAFCluster
        :return:
        :rtype: pq.second
        """

        travel_time = cluster.tour_length / self.env.mdc_speed

        # Compute the time required to upload and download all data from each
        # segment in the cluster. This has to include both inter- and intra-
        # cluster data. Because of this, we're going to simply enumerate all
        # segments and for each one in the cluster, sum the data sent to the
        # segment. Similarly, for each segment in the cluster, we will sum the
        # data sent from the segment to all other segments.

        if cluster == self.sim.hub:
            data_volume = self.energy_model.hub_data_volume(cluster)
        else:
            data_volume = self.energy_model.cluster_data_volume(cluster)

        transmit_time = data_volume / self.env.comms_rate
        total_time = travel_time + transmit_time
        return total_time

    def energy_balance(self):
        """

        :return:
        :rtype: pq.J
        """

        energy = list()
        for cluster in self.sim.clusters + [self.sim.hub]:
            energy.append(self.energy_model.total_energy(cluster.cluster_id))

        if self.sim.em_is_large or self.sim.ec_is_large:
            # Estimate to simulate spreading work out between the hub MDC and
            # other cluster MDCs
            energy.append(0.)

        balance = np.std(energy)
        return balance

    def average_energy(self):
        """

        :return:
        :rtype: pq.J
        """
        energy = list()
        for cluster in self.sim.clusters + [self.sim.hub]:
            energy.append(self.energy_model.total_energy(cluster.cluster_id))

        if self.sim.em_is_large or self.sim.ec_is_large:
            # Estimate to simulate spreading work out between the hub MDC and
            # other cluster MDCs
            energy.append(0.)

        average = np.mean(energy)
        return average

    def neighbors(self, anchor):

        clusters = list()
        for cluster in self.sim.clusters:
            if cluster.anchor == anchor:
                clusters.append(cluster)

        return clusters

    def max_buffer_size(self):

        data_volumes = list()
        for anchor in self.sim.hub.cells:

            volume = 0.  # * pq.bit
            clusters = self.neighbors(anchor)

            for cluster in clusters:
                volume += self.energy_model.cluster_data_volume(cluster,
                                                                intercluster_only=True)

            data_volumes.append(volume)

        max_data_volume = np.max(data_volumes)  # * pq.bit
        return max_data_volume
