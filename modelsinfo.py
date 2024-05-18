from nets import *
from Q_predicts import *
from utils.modelinfo import ModelInfo, ModelInfoList

nessie_models = ModelInfoList([
    ModelInfo("NegativeBinomialNet", ["NegativeBinomial", "Negative_Binomial", "negativebinomial", "negative_binomial", "NB", "nb"], NegativeBinomialNet, Q_predict_NegativeBinomial),
    ModelInfo("PoissonNet",["Poisson", "posson", "PO", "po"], PoissonNet,Q_predict_Poisson),
    ModelInfo("MultivariateNormal2DNet", ["MultivariateNormal2D", "multivariatenormal2d", "Multivariate_Normal_2D", "multivariate_normal_2d"], MultivariateNormal2DNet, Q_predict_MultivariateNormal2D),
    
])