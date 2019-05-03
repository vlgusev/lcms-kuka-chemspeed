import numpy as np
from .util import ensure_rng


def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        if isinstance(params,list):
            x = []
            for p in params:
                try:
                    assert set(p) == set(self.keys)
                except AssertionError:
                    raise ValueError(
                        "Parameters' keys ({}) do ".format(sorted(params)) +
                        "not match the expected set of keys ({}).".format(self.keys)
                    )
                x.append(np.asarray([p[key] for key in self.keys]))
        else: 
            try:
                assert set(params) == set(self.keys)
            except AssertionError:
                raise ValueError(
                    "Parameters' keys ({}) do ".format(sorted(params)) +
                    "not match the expected set of keys ({}).".format(self.keys)
                )
            x = np.asarray([params[key] for key in self.keys])
        return x

    def array_to_params(self, x):
        if isinstance(x,list):
            params = []
            for param in x:
                try:
                    assert len(param) == len(self.keys)
                except AssertionError:
                    raise ValueError(
                        "Size of array ({}) is different than the ".format(len(x)) +
                        "expected number of parameters ({}).".format(len(self.keys))
                    )
                params.append(dict(zip(self.keys, param)))
        else:
            try:
                    assert len(x) == len(self.keys)
            except AssertionError:
                raise ValueError(
                    "Size of array ({}) is different than the ".format(len(x)) +
                    "expected number of parameters ({}).".format(len(self.keys))
                )
            params = dict(zip(self.keys, x))
        return params

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique in continuous space'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def probe(self, params):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Notes
        -----
        If x has been previously seen returns a cached value of y.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)

        try:
            target = self._cache[_hashable(x)]
        except KeyError:
            params = dict(zip(self._keys, x))
            target = self.target_func(**params)
            self.register(x, target)
        return target

    def random_sample(self, constraints=[]):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        reject = True
        while reject:
            for col, (lower, upper) in enumerate(self._bounds):
                data.T[col] = self.random_state.uniform(lower, upper, size=1)
            reject = False
            for constraint in constraints:
                #if eval says reject, reject = true, break
                if constraint['fun'](data.ravel())<0:
                    reject = True
                    break
        return data.ravel()

    def max(self):
        """Get maximum target value found and corresponding parametes."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parametes."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]
                
class DiscreteSpace(TargetSpace):
    '''
    Holds the param-space coordinates (X) and target values (Y) in the discretized space. 
    This mirrors TargetSpace but supers methods to consider the floor value of discretized bins.
    That is, a prange (-5,5,.5) will register 1.3 as 1.0 in the cache but as 1.3 in the parameters list. 
    Allows for constant-time appends while ensuring no duplicates are added
    '''
    
    def __init__(self, target_func, prange, random_state=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            maximum, and step values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        
        self.random_state = ensure_rng(random_state)

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(prange)
        
        # Get associated pbounds for TargetSpace()
        self._pbounds = {item[0] :(item[1][:2]) for item in sorted(prange.items(), key=lambda x: x[0])}
        
        # Create an array with parameters steps
        self._steps = np.array(
            [item[1][-1] for item in sorted(prange.items(), key=lambda x: x[0])],
            dtype=np.float
            )
        
        # keep track of unique points we have seen so far
        self._discrete_cache = {}
        
        super(DiscreteSpace, self).__init__(target_func=target_func,
                                            pbounds=self._pbounds,
                                            random_state=random_state)
        
    @property
    def steps(self):
        return self._steps
    
    def _bin(self,x):
        # TODO: clean using modulo 
        binned  = np.empty((self.dim,1))
        for col, (lower, upper) in enumerate(self._bounds):
            binned[col] = np.floor((x[col]-lower)/self._steps[col])*self._steps[col]+lower
        return binned.ravel()
    
    def __contains__(self,x):
        return(_hashable(self._bin(x))) in self._discrete_cache
    
    def probe_discrete(self, params):    
        """
        Checks discrete cache for x and returns a cached value of y.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        x = self._as_array(params)

        try:
            target = self._discrete_cache[_hashable(x)]
        except KeyError:
            raise
        return target

    def register(self, params, target, verbose=False):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        """
        x = self._as_array(params)
        if x in self and verbose:
            print('Data point {} is not unique. \n(Discrete value {})'.format(x,self._bin(x)))
        # Insert data into unique dictionary
        self._discrete_cache[_hashable(self._bin(x))] = target
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])
        
class PartnerSpace(DiscreteSpace):
    '''
    Holds the param-space coordinates (X) in the discretized space while they have no values, but are cached. 
    This mirrors DiscreteSpace but ignores params and targets.
    Allows for constant-time appends while ensuring no duplicates are added
    '''
    
    def clear(self):
        self._discrete_cache = {}
        self._cache = {}
        
    def register(self, params, verbose=False):
        """
        Append a point and value of -1 to the partner cache.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim


        """
        x = self._as_array(params)
        if x in self and verbose:
            print('Data point {} is not unique in partner space. \n(Discrete value {})'.format(x,self._bin(x)))
        # Insert data into unique dictionary
        self._discrete_cache[_hashable(self._bin(x))] = -1
        self._cache[_hashable(x.ravel())] = -1
