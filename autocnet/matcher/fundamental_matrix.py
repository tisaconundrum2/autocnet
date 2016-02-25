from collections import deque

import numpy as np
import pandas as pd
import pysal as ps

from autocnet.matcher.outlier_detector import compute_fundamental_matrix
from autocnet.utils.utils import make_homogeneous


class FundamentalMatrix(np.ndarray):
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
    def __new__(cls, inputarr, x1, x2, mask=None):
        obj = np.asarray(inputarr).view(cls)

        if not isinstance(inputarr, np.ndarray):
            raise TypeError('The homography must be an ndarray')
        if not inputarr.shape[0] == 3 and not inputarr.shape[1] == 3:
            raise ValueError('The homography must be a 3x3 matrix.')

        obj.x1 = x1
        obj.x2 = x2
        obj.mask = mask
        obj._action_stack = deque(maxlen=10)
        obj._current_action_stack = 0
        # Seed the state package with the initial creation state
        state_package = {'arr': obj.copy(),
                         'mask': obj.mask.copy()}
        obj._action_stack.append(state_package)

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.x1 = getattr(obj, 'x1', None)
        self.x2 = getattr(obj, 'x2', None)
        self.mask = getattr(obj, 'mask', None)
        self._action_stack = getattr(obj, '_action_stack', None)
        self._current_action_stack = getattr(obj, '_current_action_stack', None)

    @property
    def determinant(self):
        if not getattr(self, '_determinant', None):
            self._determinant = np.linalg.det(self)
        return self._determinant

    @property
    def condition(self):
        if not getattr(self, '_condition', None):
            s = np.linalg.svd(self, compute_uv=False)
            self._condition = s[0] / s[1]
        return self._condition

    @property
    def error(self):
        if not hasattr(self, '_error'):
            self._error = self.compute_error()
        return self._error

    @property
    def describe_error(self):
        if not getattr(self, '_error', None):
            self._error = self.compute_error()
        return self.error.describe()

    def rollback(self, n=1):
        idx = self._current_action_stack - n
        if idx < 0:
            idx = 0
        self._current_action_stack = idx
        state = self._action_stack[idx]
        self[:] = state['arr']
        setattr(self, 'mask', state['mask'])
        # Reset attributes (could also cache)
        self._clean_attrs()

    def rollforward(self, n=1):
        idx = self._current_action_stack + n
        if idx > len(self._action_stack) - 1:
            idx = len(self._action_stack) - 1
        state = self._action_stack[idx]
        self[:] = state['arr']
        self.mask = state['mask']
        # Reset attributes (could also cache)
        self._clean_attrs()

    def refine(self, method=ps.esda.mapclassify.Fisher_Jenks, bin_id=0, **kwargs):
        """
        Refine the fundamental matrix by accepting some data classification
        method that accepts an ndarray and returns an object with a bins
        attribute, where bins are data breaks.  Using the bin_id, mask
        all values greater than the selected bin.  Then compute a
        new fundamental matrix.

        Parameters
        ----------
        method : object
                 A function that accepts and ndarray and returns an object
                 with a bins attribute
        bin_id : int
                 The index into the bins object.  Data classified > this
                 id is masked

        kwargs : dict
                 Keyword args supported by the data classifier

        Returns
        -------
        FundamentalMatrix : object
                            A fundamental matrix class object

        mask : series
               A bool mask with index attribute identifying the valid
               data in the new fundamental matrix.
        """
        # Perform the data classification
        fj = method(self.error.values.ravel(), **kwargs)
        bins = fj.bins
        # Mask the data that falls outside the provided bins
        mask = self.error['Reprojection Error'] <= bins[bin_id]
        new_x1 = self.x1.iloc[mask[mask==True].index]
        new_x2 = self.x2.iloc[mask[mask==True].index]
        fmatrix, new_mask = compute_fundamental_matrix(new_x1.values, new_x2.values)
        mask[mask==True] = new_mask

        # Update the current state
        self[:] = fmatrix
        self.mask[self.mask==True] = mask

        # Update the action stack
        state_package = {'arr': fmatrix.copy(),
                         'mask': self.mask.copy()}
        self._action_stack.append(state_package)
        self._current_action_stack = len(self._action_stack) - 1  # 0 based vs. 1 based
        self._clean_attrs()

    def _clean_attrs(self):
        for a in ['_error', '_determinant', '_condition']:
            try:
                delattr(self, a)
            except: pass

    def compute_error(self, x1=None, x2=None, index=None):
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

        if x1 is None:
            maskidx = self.mask[self.mask==True].index
            x1 = self.x1.iloc[maskidx].values
            index=maskidx
        if x2 is None:
            x2 = self.x2.iloc[maskidx].values
        err = np.zeros(x1.shape[0])

        # TODO: Vectorize the error computation
        for i, j in enumerate(x1):
            a = self[0,0] * j[0] + self[0,1] * j[1] + self[0,2]
            b = self[1,0] * j[0] + self[1,1] * j[1] + self[1,2]
            c = self[2,0] * j[0] + self[2,1] * j[1] + self[2,2]

            s2 = 1 / (a*a + b*b)
            d2 = x2[i][0] * a + x2[i][1] * b + c

            a = self[0,0] * x2[i][0] + self[0,1] * x2[i][1] + self[0,2]
            b = self[1,0] * x2[i][0] + self[1,1] * x2[i][1] + self[1,2]
            c = self[2,0] * x2[i][0] + self[2,1] * x2[i][1] + self[2,2]

            s1 = 1 / (a*a + b*b)
            d1 = j[0]*a + j[1]*b + c

            err[i] = max(d1*d1*s1, d2*d2*s2)

        error = pd.DataFrame(err, columns=['Reprojection Error'], index=index)

        return error

