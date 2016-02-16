import numpy as np

from autocnet.utils.utils import make_homogeneous
from autocnet.utils import evaluation_measures


class Homography(np.ndarray):
    """
    A homography or planar transformation matrix

    Attributes
    ----------
    determinant : float
                  The determinant of the matrix

    condition : float
                The condition computed as SVD[0] / SVD[-1]

    rmse : float
           The root mean square error computed using a set of
           given input points

    """
    def __new__(cls, inputarr, x1, x2):
        obj = np.asarray(inputarr).view(cls)

        if not isinstance(inputarr, np.ndarray):
            raise TypeError('The homography must be an ndarray')
        if not inputarr.shape[0] == 3 and not inputarr.shape[1] == 3:
            raise ValueError('The homography must be a 3x3 matrix.')

        obj.x1 = make_homogeneous(x1)
        obj.x2 = make_homogeneous(x2)

        return obj

    @property
    def determinant(self):
        if not hasattr(self, '_determinant'):
            self._determinant = np.linalg.det(self)
        return self._determinant

    @property
    def condition(self):
        if not hasattr(self, '_condition'):
            s = np.linalg.svd(self, compute_uv=False)
            self._condition = s[0] / s[1]
        return self._condition

    @property
    def rmse(self):
        if not hasattr(self, '_rmse'):

            # TODO: Vectorize this for performance
            t_kps = np.empty((self.x1.shape[0], 3))
            for i, j in enumerate(self.x1):
                proj_point = self.dot(j)
                proj_point /= proj_point[-1]  # normalize
                t_kps[i] = proj_point
            self._rmse = evaluation_measures.rmse(self.x2, t_kps)
        return self._rmse

