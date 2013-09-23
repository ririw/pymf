import logging
import numpy as np
from chnmf import CHNMF
from pca import PCA


class HCHNMF_Rule(object):
    def factorize(self, nmf_producer):
        raise NotImplementedError

    def check_consistent(self, compute_w, compute_h):
        raise NotImplementedError

    def recover_error(self):
        raise NotImplementedError

    def update_w(self):
        raise NotImplementedError

    def update_h(self):
        raise NotImplementedError

    def apply_rule(self, data):
        raise NotImplementedError

    def update_data(self, data):
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

    def update_w(self):
        assert isinstance(self.left_rule, HCHNMF_Rule)
        assert isinstance(self.right_rule, HCHNMF_Rule)
        self.left_rule.update_w() + self.right_rule.update_h()

    def update_h(self):
        assert isinstance(self.left_rule, HCHNMF_Rule)
        assert isinstance(self.right_rule, HCHNMF_Rule)
        self.left_rule.update_h() + self.right_rule.update_h()

    def apply_rule(self, data):
        n_attr, n_data = data.shape
        projection_distances = self.v.dot(data).reshape((n_data,))
        left_data = data[:, projection_distances <= self.theta]
        right_data = data[:, projection_distances > self.theta]
        return self.left_rule.apply_rule(left_data), self.right_rule.apply_rule(right_data)

    def update_data(self, data):
        self._insert_data(self.apply_rule(data))

    def _insert_data(self, data):
        left, right = data
        self.left_rule._insert_data(left)
        self.right_rule._insert_data(right)


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

    def update_w(self):
        self.factorization.update_w()

    def update_h(self):
        self.factorization.update_h()

    def apply_rule(self, data):
        return data

    def update_data(self, data):
        self.factorization.data = data

    def _insert_data(self, data):
        self.factorization.data = data


class HCHNMF(object):
    def __init__(self,
                 data,
                 num_bases=4,
                 base_sel=3,
                 leaf_minimum=0.1,
                 leaf_count_kind='proportional', # Options: proportional, absolute
                 projection_method='pca',
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
        self._data = data
        self._num_bases = num_bases

        self._data_dimension, self._num_samples = self._data.shape

        if leaf_count_kind[0] == 'p':  # Proportional
            assert leaf_minimum < 0.5
            self._leaf_minimum_int = int(float(self._num_samples) * leaf_minimum)
        else:
            self._leaf_minimum_int = leaf_minimum
        if isinstance(projection_method, str):
            assert (projection_method in {'fastmap', 'pca'})
            self._projection_method = self._choose_rule_fastmap if projection_method == 'fastmap' \
                else self._choose_rule_pca
        else:
            assert callable(projection_method), \
                "The projection method you specify must by a function or the string 'fastmap' or 'pca'"
            self._projection_method = projection_method
            # base sel should never be larger than the actual data dimension
        self._base_sel = base_sel
        if base_sel > self._data.shape[0]:
            self._logger.warn("The base number of pairwise projections has been set to the number of data dimensions")
            self._base_sel = self._data.shape[0]
        self.rule_trees = None

    def getdata(self):
        return self._data

    def setdata(self, data):
        self._data = data
        if self.rule_trees is not None:
            self.rule_trees.update_data(data)

    data = property(getdata, setdata)

    def _choose_rule_vecproject(self, data, projection_vec):
        n = data.shape[1]
        projection_distances = projection_vec.dot(data).reshape((n,))
        sorted_projections = np.sort(projection_distances)
        c = np.zeros(n - 1)
        for i in range(1, n):  # We need preconditions to prevent breaks.
            u1 = np.mean(sorted_projections[:i])
            u2 = np.mean(sorted_projections[i:])
            c[i - 1] = np.sum((sorted_projections[:i] - u1) ** float(2)) + np.sum(
                (sorted_projections[i:] - u2) ** float(2))
            # By only looking at the middle of this array, we ensure that we do not create
        # groups with less than self.leaf_minimum_int items
        rule_split = np.argmin(c[self._leaf_minimum_int:(-self._leaf_minimum_int)]) + self._leaf_minimum_int
        # On the left we have all instances of data who's projection is less than or equal to the chosen split.
        #rule_value = sorted_projections[rule_split]
        rule_value = (sorted_projections[rule_split] + sorted_projections[rule_split + 1]) / float(2)
        left_tree = data[:, projection_distances <= rule_value]
        right_tree = data[:, projection_distances > rule_value]
        return HCHNMF_vector_project_split(projection_vec, rule_value), left_tree, right_tree

    def _choose_rule_fastmap(self, data, show_progress):
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
        return self._choose_rule_vecproject(data, projection_line)

    def _choose_rule_pca(self, data, show_progress):
        """Our projections are onto the PCA vectors"""
        pca = PCA(data, num_bases=1)
        pca.factorize(show_progress)
        primary_vec = pca.W.reshape(self._data_dimension)
        return self._choose_rule_vecproject(data, primary_vec)

    def _divide_space(self, choose_rule_fn, data, show_progress):
        rule, left_points, right_points = choose_rule_fn(data, show_progress)
        assert (isinstance(rule, HCHNMF_vector_project_split))

        left_full = left_points.shape[1] <= self._leaf_minimum_int * 2 + 1
        right_full = right_points.shape[1] <= \
                     self._leaf_minimum_int * 2 + 1
        if left_full:
            self._partitioned_items += left_points.shape[1]
        if right_full:
            self._partitioned_items += right_points.shape[1]
        if left_full or right_full:
            self._logger.info("%d%% of space partitioned" % (self._partitioned_items * 100 / self._num_samples))
        rule.left_rule = HCHNMF_leaf(left_points) if left_full \
            else self._divide_space(choose_rule_fn, left_points, show_progress)
        rule.right_rule = HCHNMF_leaf(right_points) if right_full \
            else self._divide_space(choose_rule_fn, right_points, show_progress)
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
            self._partitioned_items = 0
            self.rule_trees = self._divide_space(self._projection_method, self._data, show_progress)
        else:
            assert isinstance(self.rule_trees, HCHNMF_Rule)
            self.rule_trees.check_consistent(compute_w, compute_h)
            self.rule_trees.update_data(self._data)

        def nmf_factory(data):
            factorization = CHNMF(data, self._num_bases, self._base_sel)
            self._logger.info("Factorizing matrix of shape: %s" % str(data.shape))
            factorization.factorize(
                show_progress=show_progress,
                compute_w=compute_w,
                compute_h=compute_h,
                compute_err=compute_err,
                niter=niter)
            # Something helpfully resets the logging level >:|
            if show_progress:
                self._logger.setLevel(logging.INFO)
            else:
                self._logger.setLevel(logging.ERROR)
            return factorization

        self.rule_trees.factorize(nmf_factory)
        if compute_err:
            self.ferr = self.rule_trees.recover_error()

    def update_w(self):
        self.rule_trees.update_w()

    def update_h(self):
        self.rule_trees.update_h()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import itertools

    size = 100
    w1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size).T
    w2 = np.random.multivariate_normal([4, 0], [[0.5, 0], [0, 0.5]], size).T
    w3 = np.random.multivariate_normal([4, 4], [[0.25, 0], [0, 0.25]], size).T
    w4 = np.random.multivariate_normal([0, 4], [[0.25, 0], [0, 0.25]], size).T
    x = np.concatenate((w1, w2, w3, w4), axis=1)
    d1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 0.1]], size).T
    c = HCHNMF(x, leaf_minimum=70, leaf_count_kind='a', projection_method='fastmap')
    #c = HCHNMF(x, leaf_minimum=40, leaf_count_kind='a', projection_method='pca')
    c.factorize(niter=200, show_progress=True)
    print "Testing factorization of inserted data"
    c._data = x
    c.factorize(compute_h=False, show_progress=True)
    splits = c.rule_trees.apply_rule(x)
    groups = []
    colors = ['red', 'green', 'blue', 'orange', 'purple']

    def recover_hull_points(tree):
        if isinstance(tree, HCHNMF_leaf):
            assert (isinstance(tree.factorization, CHNMF))
            return tree.factorization.data[:, tree.factorization._hull_idx]
        else:
            assert (isinstance(tree, HCHNMF_vector_project_split))
            return np.concatenate((recover_hull_points(tree.left_rule), recover_hull_points(tree.right_rule)), axis=1)

    hull_points = recover_hull_points(c.rule_trees)

    def prepsplits(splits):
        if type(splits) != tuple:
            color = colors.pop()
            #print [(hull_points.shape, split.shape) for split in splits.T]
            #print [(split == hull_points.T).all(axis=1).any() for split in splits.T]

            non_hull_items = np.array([split for split in splits.T if not (split == hull_points.T).all(axis=1).any()])
            hull_items = np.array([split for split in splits.T if (split == hull_points.T).all(axis=1).any()])
            plt.scatter(non_hull_items[:, 0], non_hull_items[:, 1], c=color, marker='x')
            plt.scatter(hull_items[:, 0], hull_items[:, 1], c=color, marker='D')
        else:
            prepsplits(splits[0])
            prepsplits(splits[1])

    prepsplits(splits)
    plt.show()
