import logging
import numpy as np
from nmf import NMF


class HCHNMF_Rule(object):
    def factorize(self, nmf_producer):
        raise NotImplementedError


class HCHNMF_split(HCHNMF_Rule):
    def __init__(self, x, y, theta, left_rule=None, right_rule=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.left_rule = left_rule
        self.right_rule = right_rule

    def factorize(self, nmf_producer):
        self.left_rule.factorize(nmf_producer)
        self.right_rule.factorize(nmf_producer)


class HCHNMF_leaf(HCHNMF_Rule):
    def __init__(self, points):
        self.points = points
        self.factorization = None

    def factorize(self, nmf_producer):
        self.factorization = nmf_producer(self.points)
        self.factorization.factorize()


class HCHNMF(NMF):
    def __init__(self,
                 data,
                 num_bases=4,
                 base_sel=3,
                 leaf_minimum=0.1,
                 leaf_count_kind='proportional',
                 projection_method='fastmap',
                 **kwargs):
        NMF.__init__(self, data, num_bases=num_bases, **kwargs)
        assert (hasattr(self, '_logger'))
        assert (isinstance(self._logger, logging.Logger))
        assert (projection_method in set(['fastmap', 'pca']))
        if leaf_count_kind[0] == 'p':
            self._leaf_minimum_int = self.data.shape[1] * leaf_minimum
        else:
            self._leaf_minimum_int = leaf_minimum
        self.projection_method = projection_method
        # base sel should never be larger than the actual data dimension
        self._base_sel = base_sel
        if base_sel > self.data.shape[0]:
            self._logger.warn("The base number of pairwise projections has been set to the number of data dimensions")
            self._base_sel = self.data.shape[0]


    def _choose_rule_fastmap(self, data):
        """ Return a tuple-tree of rules that split our data """
        # Pick a random point
        n = data.shape[0]
        assert (n > 1)

        t = data[np.random.randint(n), :]
        # Pick the furthest point from it
        t_dist = [np.linalg.norm(d) for d in data - t]
        x = data[np.argmax(t_dist), :]
        x_dist = [np.linalg.norm(d) for d in data - x]
        y = data[np.argmax(x_dist), :]

        # unit vector xy
        projection_line = (x - y) / (np.linalg.norm(x - y))

        projection_distances = np.dot(data, projection_line)
        sorted_projections = np.sort(projection_distances)
        c = np.zeros(sorted_projections.shape)
        for i in range(1, n):  # We need preconditions to prevent breaks.
            u1 = np.mean(sorted_projections[:i])
            u2 = np.mean(sorted_projections[i:])
            c[i] = np.sum((sorted_projections - u1) ** float(2)) + np.sum((sorted_projections - u2) ** float(2))
        rule_split = np.argmin(c)

        # On the left we have all instances of data who's projection is less than or equal to the chosen split.
        left_tree = data[projection_distances <= sorted_projections[rule_split]]
        right_tree = data[projection_distances > sorted_projections[rule_split]]
        rule_theta = (sorted_projections[rule_split] + sorted_projections[rule_split + 1]) / float(2)

        return HCHNMF_split(x, y, rule_theta), left_tree, right_tree

    def _divide_space(self, choose_rule_fn, data):
        rule, left_points, right_points = choose_rule_fn(data)
        assert (isinstance(rule, HCHNMF_split))

        left_full = left_points.shape[0] <= self._leaf_minimum_int
        right_full = right_points.shape[0] <= self._leaf_minimum_int
        rule.left_rule = HCHNMF_leaf(left_points) if left_full else self._divide_space(choose_rule_fn, left_points)
        rule.right_rule = HCHNMF_leaf(right_points) if right_full else self._divide_space(choose_rule_fn, right_points)
        return rule

    def factorize(self, niter=100, show_progress=False,
                  compute_w=True, compute_h=True, compute_err=True):

        if show_progress:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.ERROR)

        if not hasattr(self, 'W') and compute_w:
            self.init_w()

        if not hasattr(self, 'H') and compute_h:
            self.init_h()

        if compute_err:
            self.ferr = np.zeros(niter)

        for i in xrange(niter):
            if compute_w:
                self.update_w()

            if compute_h:
                self.update_h()

            if compute_err:
                self.ferr[i] = self.frobenius_norm()
                self._logger.info('Iteration ' + str(i + 1) + '/' + str(niter) +
                                  ' FN:' + str(self.ferr[i]))
            else:
                self._logger.info('Iteration ' + str(i + 1) + '/' + str(niter))


            # check if the err is not changing anymore
            if i > 1 and compute_err:
                if self.converged(i):
                    # adjust the error measure
                    self.ferr = self.ferr[:i]
                    break

    def update_w(self):
        pass

    def update_h(self):
        pass