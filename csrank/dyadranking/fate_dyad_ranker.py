from csrank.core.fate_network import FATENetwork
from csrank.dyadranking.dyad_ranker import DyadRanker
from csrank.numpy_util import scores_to_rankings


class FATEDyadRanker(FATENetwork, DyadRanker):
    def fit(self, Xo, Xc, Y, **kwargs):
        self._pre_fit()
        return self

    def predict_scores(self, Xo, Xc, **kwargs):
        return self.model_.predict([Xo, Xc], **kwargs)

    def predict(self, Xo, Xc, **kwargs):
        s = self.predict_scores(Xo, Xc, **kwargs)
        return scores_to_rankings(s)
