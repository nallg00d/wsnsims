from wsnsims.core.segment import Segment


class LoafSegment(Segment):
    def __init__(self, nd):
        super(LoafSegment, self).__init__(nd)
        self.cell_id = -1
        self.segment_volume = 0

    @property
    def volume(self):
        return self.segment_volume

    @volume.setter
    def volume(self, volume):
        self.segment_volume = volume
        
    def __str__(self):
        return "LOAF Segment {}".format(self.segment_id)

    def __repr__(self):
        return "LSEG {}".format(self.segment_id)
