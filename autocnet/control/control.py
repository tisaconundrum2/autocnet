import collections
from time import gmtime, strftime

class Point(object):
    """

    """
    __slots__ = '_subpixel', 'point_id', 'point_type', 'correspondences'

    def __init__(self, pid, point_type=2):
        self.point_id = pid
        self._subpixel = False
        self.point_type = point_type
        self.correspondences = []

    def __repr__(self):
        return str(self.point_id)

    def __eq__(self, other):
        return self.point_id == other

    def __hash__(self):
        return hash(self.point_id)

    @property
    def subpixel(self):
        return self._subpixel

    @subpixel.setter
    def subpixel(self, v):
        if isinstance(v, bool):
            self._subpixel = v
        if self._subpixel is True:
            self.point_type = 3


class Correspondence(object):

    __slots__ = 'id', 'x', 'y', 'measure_type', 'serial'

    def __init__(self, id, x, y, measure_type=2, serial=None):
        self.id = id
        self.x = x
        self.y = y
        self.measure_type = measure_type
        self.serial = serial


    def __repr__(self):
        return str(self.id)

    def __eq__(self, other):
        return self.id == other

    def __hash__(self):
        return hash(self.id)


class CorrespondenceNetwork(object):
    def __init__(self):
        self.point_to_correspondence = collections.defaultdict(list)
        self.correspondence_to_point = {}
        self.point_id = 0
        self.creationdate = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        self.modifieddate = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    @property
    def n_points(self):
        return len(self.point_to_correspondence.keys())

    @property
    def n_measures(self):
        return len(self.correspondence_to_point.keys())

    def add_correspondences(self, edge, matches):
        # Convert the matches dataframe to a dict
        df = matches.to_dict()
        source_image = next(iter(df['source_image'].values()))
        destination_image = next(iter(df['destination_image'].values()))

        # TODO: Handle subpixel registration here
        s_kps = edge.source.get_keypoint_coordinates().values
        d_kps = edge.destination.get_keypoint_coordinates().values

        # Load the correspondence to point data structure
        for k, source_idx in df['source_idx'].items():
            p = Point(self.point_id)
            destination_idx = df['destination_idx'][k]

            sidx = Correspondence(source_idx, *s_kps[source_idx], serial=edge.source.isis_serial)
            didx = Correspondence(destination_idx, *d_kps[destination_idx], serial=edge.destination.isis_serial)

            p.correspondences = [sidx, didx]

            self.correspondence_to_point[(source_image, source_idx)] = self.point_id
            self.correspondence_to_point[(destination_image, destination_idx)] = self.point_id

            self.point_to_correspondence[p].append((source_image, sidx))
            self.point_to_correspondence[p].append((destination_image, didx))

            self.point_id += 1
        self._update_modified_date()

    def _update_modified_date(self):
        self.modifieddate = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    def to_dataframe(self):
        pass
