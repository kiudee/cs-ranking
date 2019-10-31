IMAGE_DATASET = 'image_dataset'
SUSHI = 'sushi'
SYNTHETIC_OR = 'synthetic_or'
SYNTHETIC_CHOICE = 'synthetic_choice'
SYNTHETIC_DC = 'synthetic_dc'
HYPER_VOLUME = "hyper_volume"
MNIST_CHOICE = "mnist_choice"
MNIST_DC = "mnist_dc"
LETOR_CHOICE = "letor_choice"
EXP_CHOICE = "exp_choice"

DEPTH = 'depth'
SENTENCE_ORDERING = "sentence_ordering"
LETOR_OR = "letor_or"
LETOR_DC = "letor_dc"
LETOR_RANKING_DC = "letor_ranking_dc"
SUSHI_DC = "sushi_dc"
EXP_DC = "exp_dc"

RANKSVM = 'ranksvm'
ERR = 'err'
CMPNET = "cmpnet"
RANKNET = 'ranknet'
FETA_RANKER = 'feta_ranker'
FATE_RANKER = "fate_ranker"
LISTNET = 'listnet'
FATELINEAR_RANKER = "fatelinear_ranker"
FETALINEAR_RANKER = "fetalinear_ranker"
RANDOM_RANKER = "random_ranker"

FETA_CHOICE = 'feta_choice'
FETALINEAR_CHOICE = "fetalinear_choice"
FATE_CHOICE = "fate_choice"
FATELINEAR_CHOICE = "fatelinear_choice"
RANKNET_CHOICE = "ranknet_choice"
CMPNET_CHOICE = "cmpnet_choice"
RANKSVM_CHOICE = "ranksvm_choice"
GLM_CHOICE = "glm_choice"
RANDOM_CHOICE = "random_choice"

TAG_GENOME_OR = 'tag_genome_or'
TAG_GENOME_DC = 'tag_genome_dc'

FETA_DC = 'feta_dc'
FATE_DC = "fate_dc"
FATELINEAR_DC = "fatelinear_dc"
FETALINEAR_DC = "fetalinear_dc"
RANDOM_DC = "random_dc"

RANKNET_DC = "ranknet_dc"
CMPNET_DC = "cmpnet_dc"
MNL = "multinomial_logit_model"
NLM = 'nested_logit_model'
GEV = 'generalized_extreme_value'
PCL = 'paired_combinatorial_logit'
RANKSVM_DC = 'ranksvm_dc'
MLM = 'mixed_logit_model'

DCFS = [FETA_DC, FATE_DC, FETALINEAR_DC, FATELINEAR_DC, RANKNET_DC, RANKSVM_DC, MNL, NLM, GEV, PCL, MLM, RANDOM_DC]
DCMS = ['FETA-Net', 'FATE-Net', 'FETA-Linear', 'FATE-Linear', 'RankNetDC', 'PairwiseSVM', 'LogitModel', 'NestedLogit',
        'GenNestedLogit', 'PairedLogit', 'MixedLogit', 'Random']

CHOICE_FUNCTIONS = [FETA_CHOICE, FATE_CHOICE, FETALINEAR_CHOICE, FATELINEAR_CHOICE, RANKNET_CHOICE, RANKSVM_CHOICE,
                    GLM_CHOICE, RANDOM_CHOICE]
CHOICE_MODELS = ['FETA-Net', 'FATE-Net', "FETA-Linear", "FATE-Linear", 'RankNet-Choice', 'PairwiseSVM',
                 'GeneralizedLinearModel', "AllPositive"]

OBJECT_RANKERS = [FETA_RANKER, FATE_RANKER, FETALINEAR_RANKER, FATELINEAR_RANKER, RANKNET, LISTNET, RANKSVM, ERR,
                  RANDOM_RANKER]
OR_MODELS = ['FETA-Net', 'FATE-Net', "FETA-Linear", "FATE-Linear", 'RankNet', "ListNet", 'RankSVM', 'ERR', "Random"]
