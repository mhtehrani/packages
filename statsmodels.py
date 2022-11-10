"""
statsmodels package

https://www.statsmodels.org/stable/api.html#statsmodels-api


"""


#LinearRegression
###############################################################################
if True:
    #Ordinary Least Squares
    #https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS
    from sklearn import datasets #for regression
    X, y = datasets.load_diabetes(return_X_y=True)
    
    
    import statsmodels.api as sm
    x_sm = sm.add_constant(X)
    model = sm.OLS(y, x_sm)
    results = model.fit()
    y_hat_sm = results.predict(x_sm)
    
    results.summary()
    results.rsquared
    results.rsquared_adj
    results.params
    results.fvalue #F-test critical value
    results.f_pvalue #F-test p-value
    results.llf #log-likelihood
    results.tvalues #t-statistic critical values (~coef/std err)
    results.pvalues #t-statistics p-values
    results.aic #AIC
    results.bic #BIC
    results.fittedvalues



#Link Functions
###############################################################################
if True:
    #https://www.statsmodels.org/stable/glm.html#links
    import statsmodels.api as sm
    
    
    #identity transform
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.identity.html#statsmodels.genmod.families.links.identity
    sm.families.links.identity()
    
    
    #power transform
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.Power.html#statsmodels.genmod.families.links.Power
    #Aliases of Power: inverse = Power(power=-1) sqrt = Power(power=.5) inverse_squared = Power(power=-2.) identity = Power(power=1.)
    sm.families.links.Power(power=1.0)
    
    
    #log transform
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.Log.html#statsmodels.genmod.families.links.Log
    sm.families.links.Log()
    
    
    #logit transform
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.Logit.html#statsmodels.genmod.families.links.Logit
    sm.families.links.Logit()
    
    
    #probit (standard normal CDF) transform
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.probit.html#statsmodels.genmod.families.links.probit
    sm.families.links.probit()
    
    
    #negative binomial link function
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.NegativeBinomial.html#statsmodels.genmod.families.links.NegativeBinomial
    #g(p) = log(p/(p + 1/alpha))
    sm.families.links.NegativeBinomial(alpha=1.0) #Permissible values of alpha are usually assumed to be in (.01, 2)
    
    
    #inverse transform (the same as Power(power=-1.) )
    #g(p) = 1/p
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.inverse_power.html#statsmodels.genmod.families.links.inverse_power
    sm.families.links.inverse_power()
    
    
    #inverse squared transform (the same as Power(power=-2.) )
    #g(p) = 1/(p**2)
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.links.inverse_squared.html#statsmodels.genmod.families.links.inverse_squared
    sm.families.links.inverse_squared()





#family of distributions
###############################################################################
if True:
    #https://www.statsmodels.org/stable/glm.html#families
    import statsmodels.api as sm
    
    #Binomial exponential family distribution
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Binomial.html#statsmodels.genmod.families.family.Binomial
    #available links: logit, probit, cauchy, log, loglog, and cloglog
    sm.families.Binomial(link=sm.families.links.logit()) #default: logit
    
    
    #Gamma exponential family distribution
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Gamma.html#statsmodels.genmod.families.family.Gamma
    #available links: log, identity, and inverse
    sm.families.Gamma(link=sm.families.links.inverse_power()) #default: inverse link
    
    
    #Gaussian exponential family distribution
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Gaussian.html#statsmodels.genmod.families.family.Gaussian
    #available links: log, identity, and inverse
    sm.families.Gaussian(sm.families.links.identity()) #default: identity
    
    
    #InverseGaussian exponential family
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.InverseGaussian.html#statsmodels.genmod.families.family.InverseGaussian
    #available links: inverse_squared, inverse, log, and identity
    sm.families.InverseGaussian(sm.families.links.inverse_squared()) #default: inverse_squared
    
    
    #Negative Binomial exponential family (corresponds to NB2)
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.NegativeBinomial.html#statsmodels.genmod.families.family.NegativeBinomial
    #available links: log, cloglog, identity, nbinom and power
    #Permissible values of alpha are usually assumed to be in (.01, 2)
    sm.families.NegativeBinomial(link=sm.families.links.log(), alpha=1.0) #default: log
     
    
    #Poisson exponential family
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Poisson.html#statsmodels.genmod.families.family.Poisson
    #available links: log, identity, and sqrt
    sm.families.Poisson(link=sm.families.links.log()) #default: log
    
    
    #Tweedie family
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.families.family.Tweedie.html#statsmodels.genmod.families.family.Tweedie
    #available links: log, Power and any aliases of power
    #var_power: variance power. The default is 1
    #eql: If True, the Extended Quasi-Likelihood is used, else the likelihood is used (however the latter is not implemented). If eql is True, var_power must be between 1 and 2.
    sm.families.Tweedie(link=sm.families.links.log(), var_power=1.0, eql=False) #default: log





#Generalized Linear Models
###############################################################################
if True:
    #Generalized Linear Models
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html#statsmodels.genmod.generalized_linear_model.GLM
    import statsmodels.api as sm
    data = sm.datasets.scotland.load()
    df = data.data
    df_x = data.exog
    df_y = data.endog
    df_x = sm.add_constant(df_x)
    
    #for selection of the distribution check this
    #https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression
    
    
    model = sm.GLM(y, #(endog) 1-D array of 0 and 1 for failure and sucess
                   X, #(exog) intercept is not included by default
                   
                   #one of the exponential family class
                   family=sm.families.Gamma(), #family class instance
                   
                   #the values that the entire model is offset
                   offset=None, #array_like | None
                   
                   #if the response variable is the summation within groups, then 
                   #the counts of each obs. should pass as exposure.
                   #In this case, under the Poisson assumption the distribution remains 
                   #the same but we have varying exposure given by the number of 
                   #individuals that are represented by one aggregated observation.
                   exposure=None, #array_like | None: Log(exposure) will be added to the linear prediction in the model. Exposure is only valid if the log link is used. If provided, it must be an array with the same length as y.
                   
                   #it's equivalent to the repeating records of data
                   freq_weights=None, #If None is selected or a blank value, then the algorithm will replace with an array of 1’s with length equal to the y. WARNING: Using weights is not verified yet for all possible options and results
                   
                   #if the response variable is the mean within groups, then the 
                   #counts of each obs. should pass as var_weights.
                   #In this case the variance will be related to the inverse of the 
                   #total exposure reflected by one combined observation.
                   var_weights=None, #If None is selected or a blank value, then the algorithm will replace with an array of 1’s with length equal to the endog. WARNING: Using weights is not verified yet for all possible options and results
                   
                   missing=None) #‘none’, ‘drop’, and ‘raise’. If ‘none’, no nan checking is done. If ‘drop’, any observations with nans are dropped. If ‘raise’, an error is raised
    
    
    gamma_model = sm.GLM(df_y, df_x, family=sm.families.Gamma())
    gamma_model = sm.GLM(df_y, df_x, family=sm.families.Gaussian())
    results = gamma_model.fit()
    
    #Methods that can be applied to the results of the fitted model
    #https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.html#statsmodels.genmod.generalized_linear_model.GLMResults
    results.summary()
    results.predict(df_x)
    
    results.params #betas
    results.bse #standard errors of the parameters
    results.pvalues #the p-values of the parameetrs
    results.tvalues #T-values
    results.mu #predicted values
    results.model.family.variance(results.mu) #the variance around the predicted values
    
    results.aic
    results.bic
    results.scale #phi
    results.df_model #model df = p-1
    results.deviance #(sum of deviance residuals)
    results.pearson_chi2 #pearson statistics (sum of pearson residuals)
    results.llf #Log-Likelihood value
    
    
    
    







#LogisticRegression
###############################################################################
if True:
    #Logistic Regression (aka logit, MaxEnt) classifier.
    #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    from sklearn.datasets import load_iris #for classification
    X_iris, true_labels = load_iris(return_X_y=True)
    
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        penalty='l2', #'none' (no penalty) | 'l2' (default) | 'l1' | 'elasticnet'
        fit_intercept=True,
        class_weight=None, #'none' (equal by default) | in the form of a dictionary {class_label: weight}
        random_state=None, 
        multi_class='multinomial', #'auto' | 'ovr' (one-vs-rest) | 'multinomial' (cross-entropy loss)
        l1_ratio=None) #0 < l1_ratio < 1  (Only if penalty='elasticnet'. l1_ratio=0 --> penalty='l2', while l1_ratio=1 --> penalty='l1')
    
    clf.fit(X_iris, true_labels)
    clf.predict(X_iris)
    pred_prob = clf.predict_proba(X_iris)
    pred_log_prob = clf.predict_log_proba(X_iris) #this is exactly the np.log(pred_prob)
    pred_labels = clf.predict(X_iris)
    clf.score(X_iris, true_labels) #classification accuracy: performs prediction and compare it with the true labels
    clf.coef_
    clf.intercept_
    
    if False: #how to reconstruct the probabilities (only  works with "ovr")
        import numpy as np
        coefficient = clf.coef_.transpose()
        logit = np.matmul(X_iris,coefficient) + clf.intercept_
        A = np.exp(logit)
        P = 1/(1+1/A) # or A/(1+A)
        Prob = P/P.sum(axis=1).reshape((-1, 1))
    
    if False: #how to reconstruct the probabilities (works with "ovr" and "multinomial")
        coefficient = clf.coef_.transpose()
        logit = np.matmul(X_iris,coefficient) + clf.intercept_
        from sklearn.utils.extmath import softmax
        probabbb = softmax(logit)
    
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(true_labels, pred_prob, pos_label=1)
    
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()





#Quantile-Quantile plot
###############################################################################
if True:
    #https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html#statsmodels.graphics.gofplots.qqplot
    
    import pandas as pd
    residuals = pd.DataFrame([-1.33425646e+00,  9.92728455e-01,  4.29447010e+00,  2.06894088e-01,
       -3.74187425e+00, -8.70704780e+00,  1.30406839e+00, -2.31712105e+00,
       -4.97165326e+00, -4.39362516e+00,  2.60022915e+00,  1.06554048e+00,
       -5.42553714e+00, -7.66000550e+00, -4.21755099e+00,  7.23167032e-01,
        8.34363159e-01,  4.16675868e-01, -9.70035938e-01, -7.82334441e-01,
        4.66473574e-01, -3.15517376e+00, -3.89217633e-01, -1.05016816e+00,
       -5.02673781e-01, -2.48961725e+00, -1.40162032e+00, -7.62768510e-01,
       -5.73579352e+00, -1.81679328e+00, -2.81012725e+00, -3.13483413e+00,
        1.28086051e+01, -3.97441737e+00, -4.50231421e+00, -3.83046908e-01,
        5.68147600e+00, -1.53774419e+00,  2.22092764e+00, -4.93404009e+00,
       -1.43825020e+00, -9.69250025e-01, -6.15352207e-01, -1.76822934e+00,
       -2.09839738e+00,  1.67224764e+00, -2.41097845e+00,  1.79811144e+00,
        2.77085272e+00, -1.96987982e+00,  2.82082868e+00,  4.53236445e-01,
        2.57416743e+00,  8.05680023e-01, -1.80140635e+00,  4.09013265e-01,
       -8.19120336e+00,  3.35615250e-01, -4.44742575e+00, -3.12546932e+00,
        2.53837536e+00, -5.33944466e-01, -4.98262443e+00,  6.12699814e-01,
       -3.40197791e+00,  3.49900129e+00, -1.84830258e+00, -5.13941095e+00,
       -2.85766218e+00, -1.36404383e+00, -4.45058964e+00, -3.34622506e+00,
       -1.92651072e+00, -5.78887515e+00, -3.31856966e+00,  3.38694485e-01,
        3.32919872e-03, -2.35061024e+00,  1.25256958e+00,  2.42382902e-01,
        2.33633141e+00,  2.95900470e+00, -2.06456021e+00, -3.81075257e+00,
       -2.33995884e+00,  3.22374625e+00,  2.61608489e+00,  2.29980751e+00,
        4.48780809e+00, -2.76021593e+00, -2.49823961e-01, -1.84280662e+00,
       -2.15805635e+00, -3.22990693e+00,  4.40364030e+00,  2.34368864e+00,
       -3.03303764e+00,  3.85156145e+00,  3.01027921e-01, -4.77748679e-01,
       -1.28736992e+00,  5.41298285e+00, -3.47466507e+00,  6.06946832e+00,
        3.09329291e+00,  4.42943491e+00,  5.12784007e+00,  4.64228433e-02,
       -4.44550281e+00,  3.49550323e+00,  7.40656931e-01, -4.14998998e+00,
        5.28409965e+00, -1.47437702e+00,  2.68378528e+00,  1.07097540e+01,
        3.55957532e+00, -9.63120352e-01,  2.21534049e+00, -5.66184244e-01,
       -4.25076659e+00, -1.06207479e+00, -2.78808924e+00, -1.58050615e+00,
        3.47655819e+00, -2.98370666e+00,  5.38330569e+00,  8.37572109e-01,
        3.31190534e+00, -1.31010894e-01, -2.16849016e-02,  5.54258445e+00,
        4.05106578e+00,  1.00812016e+00, -1.60678177e-01, -2.16851401e+00,
        2.36029455e+00,  8.83415885e-01,  1.40044121e+00, -2.65984854e+00,
       -4.59884292e-02, -4.91608414e+00, -1.50371455e-01, -7.08368682e-01,
       -3.04392681e+00, -1.41608904e+00, -2.64114306e+00,  1.62925129e+00,
       -4.00711508e+00, -2.42020627e+00, -5.59020654e-01, -1.55307971e+00,
       -3.56280754e-01, -6.00793770e+00,  4.43106155e+00, -8.46502147e-01,
       -3.08917584e-02,  8.82950327e-01,  2.35480245e+00,  4.44395019e+00,
        3.27453143e+00,  1.67156870e+00,  6.31825618e+00,  3.60928346e+00,
        5.26676506e+00,  2.16643953e+00, -5.28912983e-01,  1.75331115e+00,
        9.64588763e+00, -1.65968020e+00,  1.59736391e-01, -6.81161889e+00,
        3.30787603e+00, -3.48422539e+00, -6.05853384e+00, -5.96538622e+00,
       -3.67139906e+00, -9.25475485e-01, -5.33572256e+00,  3.92033604e+00,
       -6.68966312e+00, -6.31329507e+00, -2.03637507e-01,  2.99710403e+00,
       -2.58958425e+00,  2.07076948e+00, -4.67622516e+00, -6.12880476e+00,
        2.00505014e+00, -4.72245205e+00, -6.17245398e-01, -1.72600956e+00,
        4.27088266e+00,  1.38745512e-01,  6.19748935e-02, -5.15304599e-01,
       -5.06144608e+00,  4.93763846e+00,  4.15797640e-01,  1.56689963e+00,
       -1.02120385e+00,  2.70771221e+00, -1.71510365e+00, -1.19417036e+00,
       -9.52170859e+00, -2.49449445e+00, -5.25266524e-01,  2.39056253e+00,
        2.16511717e+00, -6.02996907e+00,  3.75021119e+00, -1.89188788e+00,
       -5.90102685e+00,  3.63578800e-01, -1.87548059e-01, -3.32497474e+00,
       -3.57662895e+00,  1.69984747e+00,  2.69202401e+00,  2.43008527e+00,
        8.31389988e+00,  4.60714711e+00,  9.77350556e-01, -2.06763843e+00,
        7.54932850e+00,  1.09549801e+00,  1.54199886e+01,  7.53824909e-01,
        1.54617714e+00,  5.19197593e-02,  6.78871709e+00, -1.31732072e+00,
        3.13570966e+00, -2.25759449e+00, -3.69801714e+00, -9.07985490e+00,
        1.31199666e+00,  5.25152770e+00, -6.76513746e+00,  1.47491829e+00,
        8.86123439e-01,  1.02279068e+01,  2.52230595e+00, -3.35658914e-01,
        1.20179903e+00, -1.91671077e-01, -1.15102017e+00, -3.32203300e+00,
        1.62890838e+00, -2.31639679e+00, -2.98980885e+00,  6.43925659e-01,
        1.87375469e+00, -4.15508172e+00, -2.00670109e+00, -3.06417837e+00,
        3.06043947e+00, -4.16523697e+00, -2.76733218e+00, -4.15076802e+00,
        2.11003293e+00,  4.13720910e+00, -2.78715160e-01,  9.17757955e-01,
       -6.41645120e+00, -3.77758242e+00, -3.14974786e+00, -8.89500541e-01,
        2.81796147e+00, -3.40932292e+00,  3.23271465e-01,  2.48364647e+00,
       -2.10978787e+00,  1.33757707e+00,  1.25869081e+00,  4.62375767e+00,
       -1.03879654e+00,  6.49642673e-01, -2.90830959e+00, -7.00208309e-01,
        9.06367289e-01, -2.37098171e-01,  8.47102939e+00,  4.79731302e+00,
       -1.42790072e+00, -6.23804615e+00, -1.13325100e+00,  2.65918579e+00,
       -6.97884901e-01, -1.40281551e+00,  3.49476581e+00, -5.09326243e+00,
        1.16415648e+00,  7.46340109e+00, -2.93802718e-01,  2.13070388e+00,
        6.70417848e+00, -1.87232103e-01, -1.80756798e+00, -4.09549156e+00,
       -2.32438109e+00, -7.18397313e+00,  3.08611715e+00])
    
    from statsmodels.graphics.gofplots import qqplot
    qqplot(residuals.iloc[:,0], line='r')
    
    
    #==========================================================================
    #MANUAL implementation (easier for use with non normal distribution)
    #ranking the residuals
    ranks = residuals.rank()
    #number of observations
    n = ranks.shape[0]
    #calculating the Sample Quantiles
    quantiles = (ranks-0.5)/n
    #calculating the Theoretical Quantiles
    import scipy.stats as stats
    normal_quantile = np.zeros(n)
    for i in range(n):
        normal_quantile[i] = stats.norm.ppf(quantiles.iloc[i,0], loc=0, scale=1)
    
    #fit a linear model for plotting
    import statsmodels.api as sm
    model = sm.OLS(residuals, sm.add_constant(normal_quantile))
    res = model.fit()
    
    import matplotlib.pyplot as plt
    plt.scatter(normal_quantile, residuals)
    plt.plot(normal_quantile, res.predict(sm.add_constant(normal_quantile)), c='r')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')





