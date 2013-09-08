import logging
import numpy as np
from nmf import NMF


class LeafType(object):
    def count(self):
        raise NotImplemented
    def to_count(self, total):
        raise NotImplemented


class LeafProprortion(LeafType):
    def __init__(self, proportion):
        assert (proportion < 1.0)
        assert (proportion > 0.0)
        self._proportion = proportion

    def to_count(self, total):
        return self._proportion * total


class LeafCount(LeafType):
    def __init__(self, count):
        self._count = count

    def count(self):
        return self.count


    class HCHNMF(NMF):
        def __init__(self, data, num_bases=4, base_sel=3, leaf_minimum=LeafProprortion(0.1), **kwargs):
            NMF.__init__(data, num_bases=num_bases, **kwargs)
            assert(hasattr(self, '_logger'))
            assert(isinstance(self._logger, logging.Logger))

            assert (isinstance(leaf_minimum, LeafType))
            if isinstance(leaf_minimum, LeafProprortion):
                self._leaf_minimum_int = leaf_minimum.to_count(self.data.shape[1])
            else:
                self._leaf_minimum_int = leaf_minimum.count()
            # base sel should never be larger than the actual data dimension
            self._base_sel = base_sel
            if base_sel > self.data.shape[0]:
                self._logger.warn("The base number of pairwise projections has been set to the number of data dimensions")
                self._base_sel = self.data.shape[0]

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