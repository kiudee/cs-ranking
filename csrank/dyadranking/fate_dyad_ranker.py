from csrank.dyadranking.contextual_ranking import ContextualRanker
from csrank.fate_network import FATENetwork
from csrank.util import scores_to_rankings


class FATEContextualRanker(FATENetwork, ContextualRanker):
    def fit(self, Xo, Xc, Y, **kwargs):
        pass

    def predict_scores(self, Xo, Xc, **kwargs):
        return self.model.predict([Xo, Xc], **kwargs)

    def predict(self, Xo, Xc, **kwargs):
        s = self.predict_scores(Xo, Xc, **kwargs)
        return scores_to_rankings(s)
