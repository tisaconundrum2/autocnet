import abc


class Observable(object):

    __metaclass__ = abc.ABCMeta

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
            update_func(*args, **kwargs)

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
        for a in self.attrs:
            setattr(self, a, state[a])
        # Reset attributes (could also cache)
        self._notify_subscribers(self)

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
        for a in self.attrs:
            setattr(self, a, state[a])

        # Reset attributes (could also cache)
        self._notify_subscribers(self)

    @abc.abstractmethod
    def _update_stack(self, state):
        self._action_stack.append(state)
        self._current_action_stack = len(self._action_stack) - 1
        self._notify_subscribers

    @abc.abstractmethod
    def _clean_attrs(self):
        pass
