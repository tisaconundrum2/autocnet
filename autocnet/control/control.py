from time import gmtime, strftime

import pandas as pd


class CSeries(pd.Series):
    """
    A custom pandas series that can accept additional methods
    """
    @property
    def _constructor(self):
        return CustomSeries


class C(pd.DataFrame):
    """
    Control network.

    Parameters
    ----------

    Attributes
    ----------

    n : int
        Number of control points

    m : int
        Number of control measures

    creationdate : str
                   The date that this control network was created.
    """
    def __init__(self, *args,**kwargs):
        super(C, self).__init__(*args, **kwargs)
        self._creationdate = strftime("%Y-%m-%d %H:%M:%S", gmtime())

    @property
    def _constructor(self):
        return C

    _constructor_sliced = CSeries

    @property
    def n(self):
        if not getattr(self, '_n', None):
            self._n = 100
        return self._n

    @property
    def m(self):
        if not getattr(self, '_m', None):
            self._m = 500
        return self._m

    @property
    def creationdate(self):
        return self._creationdate

    @property
    def modifieddate(self):
        if not getattr(self, '_modifieddate', None):
            self._modifieddate = 'Not modified'
        return self._modifieddate

    @modifieddate.setter
    def update_modifieddate(self):
        self._modifieddate = strftime("%Y-%m-%d %H:%M:%S", gmtime())
