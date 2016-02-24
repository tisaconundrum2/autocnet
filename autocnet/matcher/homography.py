import numpy as np
import pandas as pd

from autocnet.utils.utils import make_homogeneous


class Homography(np.ndarray):
    """
    A homography or planar transformation matrix

    Attributes
    ----------
    determinant : float
                  The determinant of the matrix

    condition : float
                The condition computed as SVD[0] / SVD[-1]

    error : dataframe
            describing the error of the points used to
            compute this homography
    """
    def __new__(cls, inputarr, x1, x2, index=None):
        obj = np.asarray(inputarr).view(cls)

        if not isinstance(inputarr, np.ndarray):
            raise TypeError('The homography must be an ndarray')
        if not inputarr.shape[0] == 3 and not inputarr.shape[1] == 3:
            raise ValueError('The homography must be a 3x3 matrix.')

        obj.x1 = make_homogeneous(x1)
        obj.x2 = make_homogeneous(x2)
        obj.pd_index = index

        cls.__array_finalize__(cls, obj)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.x1 = getattr(obj, 'x1', None)
        self.x2 = getattr(obj, 'x2', None)
        self.pd_index = getattr(obj, 'pd_index', None)

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
    def error(self):
        if not hasattr(self, '_error'):
            self._error = self.compute_error(self.x1,
                                             self.x2,
                                             self.pd_index)
        return self._error

    def compute_error(self, a, b, index=None):
        """
        Give this homography, compute the planar reprojection error
        between points a and b.

        Parameters
        ----------
        a : ndarray
            n,2 array of x,y coordinates

        b : ndarray
            n,2 array of x,y coordinates

        index : ndarray
                Index to be used in the returned dataframe

        Returns
        -------
        df : dataframe
             With columns for x_residual, y_residual, rmse, and
             error contribution.  The dataframe also has cumulative
             x, t, and total RMS statistics accessible via
             df.x_rms, df.y_rms, and df.total_rms, respectively.
        """
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        if not isinstance(b, np.ndarray):
            b = np.asarray(b)

        if a.shape[1] == 2:
            a = make_homogeneous(a)
        if b.shape[1] == 2:
            b = make_homogeneous(b)

        # ToDo: Vectorize for performance
        for i, j in enumerate(a):
            a[i] = self.dot(j)
            a[i] /= a[i][-1]

        data = np.empty((a.shape[0], 4))

        data[:,0] = x_res = b[:,0] - a[:,0]
        data[:,1] = y_res = b[:,1] - a[:,1]
        data[:,2] = rms = np.sqrt(x_res**2 + y_res**2)
        total_rms = np.sqrt(np.mean(x_res**2 + y_res**2))
        x_rms = np.sqrt(np.mean(x_res**2))
        y_rms = np.sqrt(np.mean(y_res**2))

        data[:,3] = rms / total_rms

        df = pd.DataFrame(data,
                          columns=['x_residuals',
                                   'y_residuals',
                                   'rmse',
                                   'error_contribution'],
                          index=index)

        df.total_rms = total_rms
        df.x_rms = x_rms
        df.y_rms = y_rms

        return df

