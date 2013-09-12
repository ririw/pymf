import logging
import numpy as np
from chnmf import CHNMF


class HCHNMF_Rule(object):
    def factorize(self, nmf_producer):
        raise NotImplementedError

    def check_consistent(self, compute_w, compute_h):
        raise NotImplementedError

    def recover_error(self):
        raise NotImplementedError


class HCHNMF_vector_project_split(HCHNMF_Rule):
    def __init__(self, v, theta, left_rule=None, right_rule=None):
        super(HCHNMF_vector_project_split, self).__init__()
        self.v = v if np.linalg.norm(v) == 1 else v / np.linalg.norm(v)
        self.theta = theta
        self.left_rule = left_rule
        self.right_rule = right_rule

    def factorize(self, nmf_producer):
        assert isinstance(self.left_rule, HCHNMF_Rule)
        assert isinstance(self.right_rule, HCHNMF_Rule)
        self.left_rule.factorize(nmf_producer)
        self.right_rule.factorize(nmf_producer)

    def check_consistent(self, compute_w, compute_h):
        assert isinstance(self.left_rule, HCHNMF_Rule)
        assert isinstance(self.right_rule, HCHNMF_Rule)
        self.left_rule.check_consistent(compute_w, compute_h)
        self.right_rule.check_consistent(compute_w, compute_h)

    def recover_error(self):
        assert isinstance(self.left_rule, HCHNMF_Rule)
        assert isinstance(self.right_rule, HCHNMF_Rule)
        return self.left_rule.recover_error() + self.right_rule.recover_error()


class HCHNMF_leaf(HCHNMF_Rule):
    def __init__(self, points):
        super(HCHNMF_leaf, self).__init__()
        self.points = points
        self.factorization = None

    def factorize(self, nmf_producer):
        """ nmf_producer should return an object that fully factorizes the points """
        self.factorization = nmf_producer(self.points)
        assert (isinstance(self.factorization, CHNMF))

    def check_consistent(self, compute_w, compute_h):
        assert (isinstance(self.factorization, CHNMF))
        if not compute_w:
            assert (hasattr(self.factorization, 'W'))
        if not compute_h:
            assert (hasattr(self.factorization, 'H'))

    def recover_error(self):
        return self.factorization.ferr


class HCHNMF(object):
    def __init__(self,
                 data,
                 num_bases=4,
                 base_sel=3,
                 leaf_minimum=0.1,
                 leaf_count_kind='proportional',
                 projection_method='fastmap',
                 **kwargs):
        super(HCHNMF, self).__init__()
        self._logger = None
        # create logger
        self._logger = logging.getLogger("pymf")
        # add ch to logger
        if len(self._logger.handlers) < 1:
            # create console handler and set level to debug
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            # create formatter
            formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

            # add formatter to ch
            ch.setFormatter(formatter)
            self._logger.addHandler(ch)

        # set variables
        self.data = data
        self._num_bases = num_bases

        self._data_dimension, self._num_samples = self.data.shape

        assert (projection_method in set(['fastmap', 'pca']))
        if leaf_count_kind[0] == 'p':  # Proportional
            self._leaf_minimum_int = int(float(self._num_samples) * leaf_minimum)
        else:
            self._leaf_minimum_int = leaf_minimum
        self.projection_method = projection_method
        # base sel should never be larger than the actual data dimension
        self._base_sel = base_sel
        if base_sel > self.data.shape[0]:
            self._logger.warn("The base number of pairwise projections has been set to the number of data dimensions")
            self._base_sel = self.data.shape[0]
        self.rule_tree = None

    def _choose_rule_fastmap(self, data):
        """ Return a tuple-tree of rules that split our data """
        # Pick a random point
        n = data.shape[1]
        assert n > self._leaf_minimum_int * 2 + 1, "There must be at least %d items, but got %d" % (
            self._leaf_minimum_int * 2 + 1, n)

        t = data[:, np.random.randint(n)]
        # Pick the furthest point from it
        t_dist = [np.linalg.norm(d) for d in data.T - t]
        x = data[:, np.argmax(t_dist)]
        x_dist = [np.linalg.norm(d) for d in data.T - x]
        y = data[:, np.argmax(x_dist)]

        # unit vector xy
        projection_line = ((x - y) / (np.linalg.norm(x - y))).reshape((1, self._data_dimension))
        projection_distances = projection_line.dot(data).reshape((n,))
        sorted_projections = np.sort(projection_distances)
        c = np.zeros(n - 1)
        for i in range(1, n):  # We need preconditions to prevent breaks.
            u1 = np.mean(sorted_projections[:i])
            u2 = np.mean(sorted_projections[i:])
            c[i - 1] = np.sum((sorted_projections - u1) ** float(2)) + np.sum((sorted_projections - u2) ** float(2))
        # By only looking at the middle of this array, we ensure that we do not create
        # groups with less than self.leaf_minimum_int items
        rule_split = np.argmin(c[self._leaf_minimum_int:(-self._leaf_minimum_int)])+self._leaf_minimum_int
        # On the left we have all instances of data who's projection is less than or equal to the chosen split.
        rule_value = sorted_projections[rule_split]
        left_tree = data[:, projection_distances <= rule_value]
        right_tree = data[:, projection_distances > rule_value]
        print left_tree.shape, right_tree.shape
        rule_theta = (sorted_projections[rule_split] + sorted_projections[rule_split + 1]) / float(2)

        return HCHNMF_vector_project_split(x - y, rule_theta), left_tree, right_tree

    def _choose_rule_pca(self, data):
        """Our projections are onto the PCA vectors"""
        pass

    def _divide_space(self, choose_rule_fn, data):
        rule, left_points, right_points = choose_rule_fn(data)
        assert (isinstance(rule, HCHNMF_vector_project_split))

        left_full = left_points.shape[1] <= self._leaf_minimum_int * 2 + 1
        right_full = right_points.shape[1] <= \
                     self._leaf_minimum_int * 2 + 1
        rule.left_rule = HCHNMF_leaf(left_points) if left_full else self._divide_space(choose_rule_fn, left_points)
        rule.right_rule = HCHNMF_leaf(right_points) if right_full else self._divide_space(choose_rule_fn, right_points)
        return rule

    def factorize(self,
                  niter=1,
                  show_progress=False,
                  compute_w=True,
                  compute_h=True,
                  compute_err=True):
        """
        If compute_w or compute_h is false, then you must set rule_tree to a tree of rules where all of either the w or
        the h entries are set.
        """

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        if compute_h and compute_w:
            self.rule_tree = self._divide_space(self._choose_rule_fastmap, self.data)
        else:
            assert isinstance(self.rule_tree, HCHNMF_Rule)
            self.rule_tree.check_consistent(compute_w, compute_h)

        def nmf_factory(data):
            factorization = CHNMF(data, self._num_bases, self._base_sel)
            self._logger.error("Factorizing matrix of shape: %s" % str(data.shape))
            factorization.factorize(show_progress, compute_w, compute_h, compute_err, niter)
            return factorization

        self.rule_tree.factorize(nmf_factory)
        if compute_err:
            self.ferr = self.rule_tree.recover_error()

    def update_w(self):
        pass

    def update_h(self):
        pass


if __name__ == '__main__':
    x = np.random.random((10, 100))
    c = HCHNMF(x)
    c.factorize(niter=500)
    print c.ferr