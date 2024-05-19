from nets import *
from Q_predicts import *
from utils.modelinfo import ModelInfo, ModelInfoList

nessie_models = ModelInfoList([
    ModelInfo("NegativeBinomial", ["Negative_Binomial", "negativebinomial", "negative_binomial", "NB", "nb"], NegativeBinomialNet, Q_predict_NegativeBinomial),
    ModelInfo("Poisson",["poisson", "PO", "po"], PoissonNet,Q_predict_Poisson),
    ModelInfo("Gumbel", ["gumbel", "GU", "gu"], GumbelNet, Q_predict_Gumbel),
    ModelInfo("LogNormal", ["lognormal", "Log_Normal", "log_normal", "LN", "ln"], LogNormalNet, Q_predict_LogNormal),
    ModelInfo("Normal", ["normal", "Gaussian", "gaussian", "NO", "no"], NormalNet, Q_predict_Normal),
    ModelInfo("Weibull", ["weibull", "WE", "we"], WeibullNet, Q_predict_Weibull),
    ModelInfo("MultivariateNormal2D", ["MultivariateNormal2D", "multivariatenormal2d", "Multivariate_Normal_2D", "multivariate_normal_2d"], MultivariateNormal2DNet, Q_predict_MultivariateNormal2D),
])