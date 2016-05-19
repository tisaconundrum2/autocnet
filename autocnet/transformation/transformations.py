import abc
from collections import deque
import warnings

import numpy as np
import pandas as pd
import pysal as ps
from scipy import optimize

from autocnet.matcher.outlier_detector import compute_fundamental_matrix
from autocnet.utils.utils import make_homogeneous, normalize_vector, crossform
from autocnet.camera import camera


class TransformationMatrix(np.ndarray):
    """
    Abstract Base Class representing a 3x3 transformation matrix.
    This ABC subclasses numpy ndarrays.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __new__(cls, inputarr, x1, x2, mask, local_mask=None):
        if not isinstance(inputarr, np.ndarray):
            raise TypeError('The transformation must be an ndarray')

        obj = np.asarray(inputarr).view(cls)
        obj.x1 = x1
        obj.x2 = x2
        obj.mask = mask
        obj.local_mask = local_mask
        obj._action_stack = deque(maxlen=10)
        obj._current_action_stack = 0
        obj._observers = set()

        # Seed the state package with the initial creation state
        if mask is not None:
            state_package = {'arr': obj.copy(),
                             'mask': obj.mask.copy()}
        else:
            state_package = {'arr': obj.copy(),
                             'mask': None}
        obj._action_stack.append(state_package)

        return obj

    @abc.abstractmethod
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.x1 = getattr(obj, 'x1', None)
        self.x2 = getattr(obj, 'x2', None)
        self.mask = getattr(obj, 'mask', None)
        self._action_stack = getattr(obj, '_action_stack', None)
        self._current_action_stack = getattr(obj, '_current_action_stack', None)
        self._observers = getattr(obj, '_observers', None)

    @abc.abstractproperty
    def determinant(self):
        if not getattr(self, '_determinant', None):
            self._determinant = np.linalg.det(self)
        return self._determinant

    @abc.abstractproperty
    def condition(self):
        if not getattr(self, '_condition', None):
            s = np.linalg.svd(self, compute_uv=False)
            self._condition = s[0] / s[1]
        return self._condition

    @abc.abstractproperty
    def error(self):
        if not hasattr(self, '_error'):
            self._error = self.compute_error(self.x1,
                                             self.x2,
                                             self.mask)
        return self._error

    @abc.abstractproperty
    def describe_error(self):
        return self.error.describe()

    @abc.abstractmethod
    def rollback(self, n=1):
        """
        Roll backward in the object histroy, e.g. undo

        Parameters
        ----------

        n : int
            the number of steps to roll backwards
        """
        idx = self._current_action_stack - n
        if idx < 0:
            idx = 0
        self._current_action_stack = idx
        state = self._action_stack[idx]
        self[:] = state['arr']
        setattr(self, 'mask', state['mask'])
        # Reset attributes (could also cache)
        self._clean_attrs()
        self._notify_subscribers(self)

    @abc.abstractmethod
    def rollforward(self, n=1):
        """
        Roll forwards in the object history, e.g. do

        Parameters
        ----------

        n : int
            the number of steps to roll forwards
        """
        idx = self._current_action_stack + n
        if idx > len(self._action_stack) - 1:
            idx = len(self._action_stack) - 1
        self._current_action_stack = idx
        state = self._action_stack[idx]
        self[:] = state['arr']
        setattr(self, 'mask', state['mask'])
        # Reset attributes (could also cache)
        self._clean_attrs()
        self._notify_subscribers(self)

    @abc.abstractmethod
    def subscribe(self, func):
        """
        Subscribe some observer to the edge

        Parameters
        ----------

        func : object
               The callable that is to be executed on update
        """
        self._observers.add(func)

    @abc.abstractmethod
    def _notify_subscribers(self, *args, **kwargs):
        """
        The 'update' call to notify all subscribers of
        a change.
        """
        for update_func in self._observers:
            update_func(self, *args, **kwargs)

    @abc.abstractmethod
    def compute_error(self, x1, x2, index=None):
        pass

    @abc.abstractmethod
    def recompute_matrix(self):
        pass

    @abc.abstractmethod
    def refine(self):
        pass


class FundamentalMatrix(TransformationMatrix):
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

    @property
    def rank(self):
        """
        A valid fundamental matrix should be rank 2.
        Hartley & Zisserman p. 280
        """
        rank = np.linalg.matrix_rank(self)
        if rank != 2:
            warnings.warn('F rank not equal to 2.  This indicates a poor F matrix.')
        return rank

    def refine_with_mle(self, **kwargs):
        """
        Given a linear approximation of F, refine using Maximum Liklihood estimation
        as per Hartley and Zisseman p.285, algorithm 11.3.

        References
        ----------
        .. [Hartley2003]
        """

        p = camera.idealized_camera()
        p1 = camera.estimated_camera_from_f(self)

        correspondences1 = self.x1[self.local_mask]
        correspondences2 = self.x2[self.local_mask]

        if 'method' in kwargs.keys():
            method = kwargs.pop('method')
        else:
            method = 'trf'
        result = optimize.least_squares(camera.projection_error, p1.ravel(), args=(p,
                                                                             correspondences1.values,
                                                                             correspondences2.values),
                                        method=method,
                                        xtol=1e-6,
                                        ftol=1e-6,
                                        gtol=1e-6,
                                        **kwargs)

        if result[-1] > 4:
            warnings.warn('MLE failed to find an improved fundamental matrix.')

        # Scipy solvers are 1D, reshape to the correct form
        pgs = result[0].reshape(3,4)
        t = pgs[:, 3]
        M = pgs[:, 0:3]
        self[:] = crossform(t).dot(M)

    def refine(self, method=ps.esda.mapclassify.Fisher_Jenks, values=None, bin_id=0, **kwargs):
        """
        Refine the fundamental matrix by accepting some data classification
        method that accepts an ndarray and returns an object with a bins
        attribute, where bins are data breaks.  Using the bin_id, mask
        all values greater than the selected bin.  Then compute a
        new fundamental matrix.
        The matrix is "refined" based on the reprojection errors for
        each point.
        Parameters
        ----------
        method : object
                 A function that accepts and ndarray and returns an object
                 with a bins attribute
        values      : ndarray
                      (n,1) of values to use used for classification
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
        if values is None:
            values = self.error
        fj = method(values.ravel(), **kwargs)
        # Mask the data that falls outside the provided bins
        mask = values <= fj.yb[bin_id]
        new_x1 = self.x1.iloc[mask[mask == True].index]
        new_x2 = self.x2.iloc[mask[mask == True].index]
        fmatrix, new_mask = compute_fundamental_matrix(new_x1.values, new_x2.values)
        mask[mask == True] = new_mask

        # Update the current state
        self[:] = fmatrix
        self.mask = mask

        # Update the action stack
        try:
            state_package = {'arr': fmatrix.copy(),
                             'mask': self.mask.copy()}

            self._action_stack.append(state_package)
            self._current_action_stack = len(self._action_stack) - 1  # 0 based vs. 1 based
            self._clean_attrs()
            self._notify_subscribers(self)
        except:
            warnings.warn('Refinement outlier detection removed all observations.',
                          UserWarning)

    def _clean_attrs(self):
        for a in ['_error', '_determinant', '_condition']:
            try:
                delattr(self, a)
            except:
                pass

    @property
    def error(self):
        """
        Using the currently unmasked correspondences, compute the reprojection
        error.

        Returns
        -------
        : ndarray
          The current error

        See Also
        --------
        compute_error : The method called to compute element-wise error.
        """
        x = self.x1.loc[self.mask]
        x1 = self.x2.loc[self.mask]
        return self.compute_error(self.x1, self.x2)

    def compute_error(self, x, x1):
        """
        Given a set of matches and a known fundamental matrix,
        compute distance between all match points and the associated
        epipolar lines.

        Ideal error is defined by $x^{\intercal}Fx = 0$, where x
        where $x$ are all matchpoints in a given image and
        $x^{\intercal}F$ defines the standard form of the
        epipolar line in the second image.

        The distance between a point and the associated epipolar
        line is computed as: $d = \frac{\lvert ax_{0} + by_{0} + c \rvert}{\sqrt{a^{2} + b^{2}}}$.

        Parameters
        ----------

        x : dataframe
            n,3 dataframe of homogeneous coordinates

        x1 : dataframe
            n,3 dataframe of homogeneous coordinates with the same
            length as argument x

        Returns
        -------
        F_error : ndarray
                  n,1 vector of reprojection errors
        """

        # Normalize the vector
        l_norms = normalize_vector(x.dot(self.T))
        F_error = np.abs(np.sum(l_norms * x1, axis=1))

        return F_error

    def recompute_matrix(self):
        raise NotImplementedError


class Homography(TransformationMatrix):
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

    def compute_error(self, a, b, mask=None):
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

        if mask is not None:
            mask = mask
        else:
            mask = self.mask
        index = mask[mask == True].index

        a = a.iloc[index].values
        b = b.iloc[index].values

        if a.shape[1] == 2:
            a = make_homogeneous(a)
        if b.shape[1] == 2:
            b = make_homogeneous(b)

        # ToDo: Vectorize for performance
        for i, j in enumerate(a):
            a[i] = self.dot(j)
            a[i] /= a[i][-1]

        data = np.empty((a.shape[0], 4))

        data[:, 0] = x_res = b[:, 0] - a[:, 0]
        data[:, 1] = y_res = b[:, 1] - a[:, 1]
        data[:, 2] = rms = np.sqrt(x_res**2 + y_res**2)
        total_rms = np.sqrt(np.mean(x_res**2 + y_res**2))
        x_rms = np.sqrt(np.mean(x_res**2))
        y_rms = np.sqrt(np.mean(y_res**2))

        data[:, 3] = rms / total_rms

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

    def recompute_matrix(self):
        raise NotImplementedError

    def refine(self):
        raise NotImplementedError
