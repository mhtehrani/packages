"""
sklearn package

https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling


"""
#Scalers | Transformers | Encoders
###############################################################################
if True:
    #StandardScaler
    ###############################################################################
    if True:
        #Z-Score Standardization Scaler
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
        data = [[8, 160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler() # create z-score standardization object
        scaler.fit(data) # it calculates the transformation parameteres for each column in data
        scaler.mean_ #the mean of each column
        scaler.var_ #the variance of each column
        scaler.transform(data) #applying the transformation on new data
    
    
    
    #MinMaxScaler
    ###############################################################################
    #Min Max Scaler
    if True:
        data = [[8, 160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler() #default is 0 to 1
        scaler = MinMaxScaler(feature_range=(-1, 1)) #or customized
        scaler.fit(data)
        scaler.data_max_
        scaler.data_min_
        scaler.transform(data)
    
    
    
    #MaxAbsScaler
    ###############################################################################
    #Max Absolute Scaler
    if True:
        data = [[8, -160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        #the maximum absolute value will be 1
        #it preserve the 0 entry in the data
        #it divide all the items by their column maximum
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        scaler.fit(data)
        scaler.transform(data)
    
    
    
    #RobustScaler
    ###############################################################################
    #Robust Scaler (recommended for data with outlier)
    if True:
        data = [[8, 160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        #it uses median and IQR (0.25 and 0.75)
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        scaler.fit(data)
        scaler.transform(data)
    
    
    
    #QuantileTransformer
    ###############################################################################
    #Quantile Transformer (Transform to Uniform or Normal)
    if True:
        data = [[8, 160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        #it uses quantiles and inverse CDF to transform
        from sklearn.preprocessing import QuantileTransformer
        transformer = QuantileTransformer(output_distribution='uniform')
        transformer = QuantileTransformer(output_distribution='normal')
        transformer.fit(data)
        transformer.transform(data)
    
    
    
    #PowerTransformer
    ###############################################################################
    #Power Transformer (Transform Normal)
    if True:
        data = [[8, 160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        from sklearn.preprocessing import PowerTransformer
        transformer = PowerTransformer(method='yeo-johnson')
        transformer = PowerTransformer(method='box-cox') #works with positive numebrs only
        transformer.fit(data)
        transformer.transform(data)
        transformer.lambdas_
    
    
    
    #FunctionTransformer
    ###############################################################################
    #Constructs a transformer from an arbitrary callable
    if True:
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer
        
        import numpy as np
        import pandas as pd
        data_c = pd.DataFrame([[8, 160, 18765,'a'],[11, 220, 11451,'c'], [1, 76, 16457,'b'], 
                               [3, 223, 23451,'a'], [7, 183, 17456,'b'], [20, 101, 10156, np.nan]]).values
        
        from sklearn.preprocessing import FunctionTransformer
        transformer = FunctionTransformer(np.log)
        transformer.fit(data_c)
        transformer.transform(data_c)
    
    
    
    #OrdinalEncoder
    ###############################################################################
    #Ordinal Encoder --> (0 to n_categories - 1) --> for input variables
    if True:
        import numpy as np
        import pandas as pd
        data_c = pd.DataFrame([[8, 160, 18765,'a'],[11, 220, 11451,'c'], [1, 76, 16457,'b'], 
                               [3, 223, 23451,'a'], [7, 183, 17456,'b'], [20, 101, 10156, np.nan]]).values
        
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(categories='auto', encoded_missing_value=np.nan) #or we should give the list of list for each column
        enc = OrdinalEncoder(categories=[['b','c','a']], handle_unknown='use_encoded_value', unknown_value=np.nan)
        enc.fit(data_c[:,-1].reshape([-1,1])) #it accepts 2D
        enc.transform(data_c[:,-1].reshape([-1,1]))
        enc.categories_
        enc.inverse_transform(enc.transform(data_c[:,-1].reshape([-1,1])))
    
    
    
    #LabelEncoder
    ###############################################################################
    #Label Encoder --> (0 to n_class - 1) --> for output variable
    if True:
        import numpy as np
        import pandas as pd
        data_c = pd.DataFrame([[8, 160, 18765,'a'],[11, 220, 11451,'c'], [1, 76, 16457,'b'], 
                               [3, 223, 23451,'a'], [7, 183, 17456,'b'], [20, 101, 10156, np.nan]]).values
        
        from sklearn.preprocessing import LabelEncoder
        enc = LabelEncoder()
        enc.fit(data_c[:,-1]) #it accepts 1D
        enc.transform(data_c[:,-1])
        enc.classes_
        enc.inverse_transform(enc.transform(data_c[:,-1]))
    
    
    
    #OneHotEncoder
    ###############################################################################
    #One-Hot (dummy) Encoder --> binary column for each category
    if True:
        import numpy as np
        import pandas as pd
        data_c = pd.DataFrame([[8, 160, 18765,'a'],[11, 220, 11451,'c'], [1, 76, 16457,'b'], 
                               [3, 223, 23451,'a'], [7, 183, 17456,'b'], [20, 101, 10156, np.nan]]).values
        
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder
        from sklearn.preprocessing import OneHotEncoder
        enc = OneHotEncoder(drop=None, #None : retain all | ‘first’ : drop the first category in each feature
                            categories='auto', #or we should give the list of list for each column
                            handle_unknown='ignore', 
                            sparse=False) #True: return matrix then need .toarray() to convert to array | False: return array
        
        enc.fit(data_c[:,-1].reshape([-1,1])) #it accepts 1D
        enc.transform(data_c[:,-1].reshape([-1,1])) #.toarray() if spars=True
        enc.categories_
        enc.inverse_transform(enc.transform(data_c[:,-1].reshape([-1,1]))) #create the labels from encoded
    
    
    
    #KBinsDiscretizer
    ###############################################################################
    #Bin continuous data into intervals.
    if True:
        #https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn.preprocessing.KBinsDiscretizer
        
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.preprocessing import KBinsDiscretizer
        est = KBinsDiscretizer(n_bins=3, #>=2: number of bins to produce
                               encode='onehot', # 'onehot' | 'onehot-dense' | 'ordinal'
                               strategy='quantile') #'uniform' | 'quantile' | 'kmeans': Strategy used to define the widths of the bins.
        
        est = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
        est.fit(X_iris)
        est.transform(X_iris)
    
    
    
    #label_binarize
    ###############################################################################
    #converting multi lable class to set of binary class types (one-vs-all)
    if True:
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        #(it can then be fed into one-vs-all classifier)
        # it changes the 1D labels from classes 0, 1, 2, ..., N to a new_label of size (:, N+1)
        from sklearn.preprocessing import label_binarize
        true_labels_one_vs_all = label_binarize(true_labels, classes=[0, 1, 2])
    
    
    
    #PolynomialFeatures
    ###############################################################################
    #creating polinomial of a variable upto a certain degree with all the interactions
    if True:
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        #for example: a degree 3 with all the interactions on x1 and x2 would result the following
        #x1    x2    x1^2    x1*x2    X2^2    x1^3    x1^2*x2    x1*x2^2    x2^3
        from sklearn.preprocessing import PolynomialFeatures
        transformer = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
        transformer.fit(X[:,0:2])
        X_p = transformer.transform(X[:,0:2])
    
    
    
    #SplineTransformer
    ###############################################################################
    #create a set of basis feature for each column (n_knots + degree - 1)
    #works kind of like feature extraction
    if True:
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        #it means that a combination of these basis are representing the ground truth 
        #and we can let the model decide on their factors
        from sklearn.preprocessing import SplineTransformer
        spline = SplineTransformer(degree=1, n_knots=2)
        spline.fit(X[:,0].reshape(-1,1))
        X_ = spline.transform(X[:,0].reshape(-1,1))
        spline.bsplines_
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(np.arange(X.shape[0]), X[:,0], color="darkorange")
        plt.scatter(np.arange(X.shape[0]), np.min(X[:,0])*X_[:,0]+np.max(X[:,0])*X_[:,1], color="navy")
        plt.show()
    
    
    
    #TransformedTargetRegressor *********Don't know what it does*********
    ###############################################################################
    #applying a transformer to the target variable in a regressor model
    if True:
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        #it can be defined using the func and the inverse of the func or a transformer object
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import QuantileTransformer
        tt = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)
        tt = TransformedTargetRegressor(regressor=LinearRegression(), transformer=QuantileTransformer(output_distribution='normal'))
        tt.fit(X, y)
        tt.predict(X) #the output are in original unit and there is no need to inverse
        tt.score(X, y)
    
    
    
    #ColumnTransformer
    ###############################################################################
    if True:
        #Applies transformers to columns of an array or pandas DataFrame
        #https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html
        data = [[8, 160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        #the make_column_transformer automatically fill the names
        from sklearn.compose import ColumnTransformer
        from sklearn.compose import make_column_transformer
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.preprocessing import MinMaxScaler
        #List of (name, transformer, columns)  tuples
        ct = ColumnTransformer([("QuantileTransformer", QuantileTransformer(output_distribution='normal'), [1]), 
                                ("MinMaxScaler", MinMaxScaler(), slice(0, 2))], 
                               remainder='drop') #'drop' | 'passthrough'
        ct = ColumnTransformer([("QuantileTransformer", QuantileTransformer(output_distribution='normal'), [1]), 
                                ("MinMaxScaler", 'passthrough', [2])], 
                               remainder='drop')
        ct = make_column_transformer((QuantileTransformer(output_distribution='normal'), [1]), 
                                (MinMaxScaler(), slice(0, 2))) #don't need []
        ct.fit(data)
        ct.transform(data)





#Imputers
###############################################################################
if True:
    import numpy as np
    import pandas as pd
    data_c = pd.DataFrame([[8, np.nan, 18765,'a'],[11, 220, 11451,'c'], [1, 76, 16457,'b'], 
                          [3, 223, np.nan,'a'], [7, 183, 17456,'b'], [20, 101, 10156, np.nan], 
                          [5, 213, 23451,'c'], [17, 123, np.nan,'b'], [10, 141, 8156, np.nan]]).values
    
    #SimpleImputer
    ###############################################################################
    if True:
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, #the missingness
                            strategy='mean', # 'mean' | 'median' | 'most_frequent' | 'constant'
                            fill_value=0, #the fill value - only be used with strategy='constant'
                            add_indicator=True) #add a column showing the location of missingness
        imp = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
        imp.fit(data_c[:,0:3])
        imp.transform(data_c[:,0:3])
        imp.statistics_
        imp.inverse_transform(imp.transform(data_c[:,0:3])) #works only with add_indicator=True
        
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent', add_indicator=True)
        imp.fit(data_c)
        imp.transform(data_c)
        imp.statistics_
        imp.inverse_transform(imp.transform(data_c)) #doesn't work with characters in the inverse
    
    
    
    #IterativeImputer
    ###############################################################################
    if True:
        #this doesn't support inverse transform (however, we can do it manually)
        #To perform multiple imputation, we need to define random seed and repeat m times for m imputation
        from sklearn.experimental import enable_iterative_imputer #it's needed to use the package in normal way
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge #we can use different regressor
        from sklearn.ensemble import RandomForestRegressor #similar to missForest in "R"
        from sklearn.neighbors import KNeighborsRegressor
        imp = IterativeImputer(estimator=BayesianRidge(),
                               add_indicator=True)#default
        
        imp.fit(data_c[:,0:3])
        imp.transform(data_c[:,0:3])
    
    
    
    #KNNImputer
    ###############################################################################
    if True:
        #this doesn't support inverse transform (however, we can do it manually)
        from sklearn.impute import KNNImputer
        imp = KNNImputer(missing_values=np.nan, n_neighbors=5, 
                         weights='uniform', #'uniform' | 'distance'
                         add_indicator=False)
        
        imp.fit(data_c[:,0:3]) #only works with numerical variables
        imp.transform(data_c[:,0:3])



    #MissingIndicator
    ###############################################################################
    if True:
        from sklearn.impute import MissingIndicator
        indicator = MissingIndicator()
        indicator.fit(data_c)
        indicator.transform(data_c)





#train_test_split | Pipeline | FeatureUnion
###############################################################################
if True:
    #train_test_split
    ###############################################################################
    if True:
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_iris, true_labels, 
                                                            test_size=0.33, #if float proportion of test size | if int the exact number of test samples
                                                            random_state=42)
    
    
    
    #Pipeline
    ###############################################################################
    #Sequentially apply a list of transforms and a final estimator
    if True:
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        #the difference between Pipeline and make_pipeline:
        #the make_pipeline automatically fill the names
        from sklearn.pipeline import Pipeline
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_iris, true_labels, random_state=0)
        #a list of (key, method) pairs #all the methods (except the last one should have transform() method)
        pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True))])
        pipe = make_pipeline(StandardScaler(), SVC(probability=False))
        pipe.set_params(svc__probability=True) #change the parameter of estimator | estimater__parameter | it's good for grid search CV
        pipe.fit(X_train, y_train) #fit the entire pipeline
        pipe.steps
        pipe.steps[0]
        pipe.named_steps.standardscaler #can aceess with the names
        pipe.named_steps.svc
        pipe['standardscaler']
        pipe.predict(X_train)
        pipe.predict_proba(X_train)
        pipe.predict_log_proba(X_train)
        pipe.score(X_train, y_train) #performance of the final model on training data
        pipe.score(X_test, y_test) #performance of the final model on test data
    
    
    
    #FeatureUnion
    ###############################################################################
    #Concatenates results of multiple independent transformer objects
    if True:
        data = [[8, 160, 18765],[11, 220, 11451], [1, 76, 16457], [3, 223, 23451], [7, 183, 17456], [20, 101, 10156]]
        
        #the make_union automatically fill the names
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import make_union
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.preprocessing import PowerTransformer
        #a list of (key, method) pairs #all the methods
        combined = FeatureUnion([('QuantileTransformer', QuantileTransformer(output_distribution='normal')), 
                                 ('PowerTransformer', PowerTransformer(method='box-cox'))])
        combined = make_union(QuantileTransformer(output_distribution='normal'), PowerTransformer(method='box-cox'))
        combined.fit(data)
        combined.transform(data) #it's just the concatenation of QuantileTransformer and PowerTransformer





#PCA
###############################################################################
#Principal component analysis
#The input data is centered but not scaled for each feature before applying the SVD
if True:
    from sklearn import datasets #for regression
    X, y = datasets.load_diabetes(return_X_y=True)
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    
    from sklearn.decomposition import PCA
    pca = PCA(#n_components=3, #if doesn't given: min(n_samples, n_features)
              svd_solver='full') # 'auto' | 'full'
    pca.fit(X)
    pca.transform(X)
    pca.components_ #Principal axes in feature space (right singular vectors) (eigenvectors) (n_components, n_features)
    pca.explained_variance_ #The amount of variance explained by each of the selected components
    pca.explained_variance_ratio_ #Percentage of variance explained by each of the selected components
    pca.singular_values_ #The singular values --> eignevalues = (singular values)^2 --> (S/sqrt(n-1))
    
    #eigenvector (principal axis or principal direction)
    V = np.transpose(pca.components_) #each column is one principal axis
    np.sum(np.power(V,2), axis=0) # to confirm the lengthes are equal 1
    L = np.diag(np.power(pca.singular_values_,2)) # eignevalues = (singular values)^2
    covariance_matrix = np.matmul(np.transpose(X), X) # covariance matrix
    covariance_matrix_ = np.matmul(V,np.matmul(L,np.transpose(V))) # covariance matrix (to confirm)
    
    proj_obs_0 = pca.transform(X[0:1,:])
    proj_obs_0_ = np.matmul(X[0,:],V) #confirmation of the projection

    #Eigenvalues
    import matplotlib.pyplot as plt
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.bar(list(range(1,pca.n_components_+1)), np.power(pca.singular_values_,2), width=0.8, color='blue', linewidth=2, 
            tick_label=['PC'+str(i) for i in range(1,pca.n_components_+1)])
    for i in range(1,pca.n_components_+1):
        plt.text(i, np.power(pca.singular_values_[i-1],2)+0.1, 
                 str(round(np.power(pca.singular_values_[i-1],2),2)), color = 'g', ha = 'center', va = 'center')
    
    plt.ylabel('Eigenvalues', fontsize=10, labelpad=5)
    plt.title('Scree plot', fontsize=20)
    plt.xlim([0.5,10.5])
    plt.show()
    
    
    #scree plot (percentage of variation that each PC accounts for)
    import matplotlib.pyplot as plt
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.bar(list(range(1,pca.n_components_+1)), 100*pca.explained_variance_ratio_, width=0.8, color='blue', linewidth=2, 
            tick_label=['PC'+str(i) for i in range(1,pca.n_components_+1)])
    for i in range(1,pca.n_components_+1):
        plt.text(i, 100*pca.explained_variance_ratio_[i-1]+1, 
                 str(np.round_(100*pca.explained_variance_ratio_[i-1],1)), color = 'g', ha = 'center', va = 'center')
    
    plt.ylabel('Explained variances (%)', fontsize=10, labelpad=5)
    plt.title('', fontsize=20)
    plt.xlim([0.5,10.5])
    plt.show()
    
    
    #Cumulative Explained variance (%)
    import matplotlib.pyplot as plt
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.bar(list(range(1,pca.n_components_+1)), 100*np.cumsum(pca.explained_variance_ratio_), width=0.8, color='blue', linewidth=2, 
            tick_label=['PC'+str(i) for i in range(1,pca.n_components_+1)])
    for i in range(1,pca.n_components_+1):
        plt.text(i, 100*np.cumsum(pca.explained_variance_ratio_)[i-1]+3, 
                 str(np.round_(100*np.cumsum(pca.explained_variance_ratio_)[i-1],1)), color = 'g', ha = 'center', va = 'center')
    
    plt.hlines(90, xmin=0.5, xmax=10.5, colors='red', alpha=0.8)
    plt.ylabel('Cumulative Explained variances (%)', fontsize=10, labelpad=5)
    plt.title('', fontsize=20)
    plt.show()
    
    #biplot (loading values which are the coefficients of variable on each principal direction)
    pc_i = 0
    pc_j = 1
    x_transformed = pca.transform(X)
    from sklearn.preprocessing import MinMaxScaler
    x_transformed_scaled = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x_transformed)
    import matplotlib.pyplot as plt
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    pci = x_transformed_scaled[:,pc_i]
    pcj = x_transformed_scaled[:,pc_j]
    plt.scatter(pci, pcj, c=y)
    pc_score_i = pca.components_[pc_i]
    pc_score_j = pca.components_[pc_j]
    for i in range(X.shape[1]):
        plt.arrow(0, 0, pc_score_i[i], pc_score_j[i], color='red', linewidth=2, head_width=0.04, overhang=1)
        plt.text(pc_score_i[i]+(0.1*pc_score_i[i])/abs(pc_score_i[i]), 
                 pc_score_j[i]+(0.1*pc_score_j[i])/abs(pc_score_j[i]), 
                 'var_'+str(i+1), color='red', ha='center', va='center', fontweight='bold')
    
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(pc_i+1))
    plt.ylabel("PC{}".format(pc_j+1))
    plt.grid()
    plt.show()





#permutation_importance
###############################################################################
if True:
    #Permutation importance for feature evaluation
    #Prefered over the impurity-based feature importance
    #https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance
    from sklearn.datasets import load_iris #for classification
    X_iris, true_labels = load_iris(return_X_y=True)
    
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X_iris, true_labels)
    
    from sklearn.preprocessing import label_binarize
    true_labels_one_vs_all = label_binarize(true_labels, classes=[0, 1, 2])
    
    from sklearn.inspection import permutation_importance
    result = permutation_importance(clf, #An estimator that has already been fitted
                                    X_iris, #Input
                                    true_labels_one_vs_all, #Output
                                    scoring='roc_auc', #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                    n_repeats=5, #Number of times to permute a feature
                                    random_state=None)
    result.importances_mean #Mean of feature importance over n_repeats
    result.importances_std #Standard deviation over n_repeats
    result.importances #Raw permutation importance scores
    
    #---------------------------------------
    from sklearn import datasets #for regression
    X, y = datasets.load_diabetes(return_X_y=True)
    
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression(fit_intercept=True) #create a linear regression object
    reg.fit(X, y)
    result = permutation_importance(reg, X, y, scoring='neg_mean_squared_error', n_repeats=5, random_state=None)
    result.importances_mean





#Feature selection
###############################################################################
if True:
    #VarianceThreshold
    if True:
        #Feature selector that removes all low-variance features. (good for unsupervised learning since it don't look at y)
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html#sklearn.feature_selection.VarianceThreshold
        X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
        #this is a bernuli trial and if we are looking for 80% of observation within one class,
        #we can use threshold=p*(1-p)=0.8*(1-0.8)
        
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.16) #Features with a training-set variance lower than this threshold will be removed
        selector.fit(X)
        selector.transform(X) #the first column is removed
        selector.variances_
    
    
    
    #scorings
    if True:
        #f_regression (continuous features in a regression problem)
        if True:
            #Univariate linear regression tests returning F-statistic and p-values.
            #Univariate method do not look at all the features collectively (ignore 
            #interactions between features). Instead, it looks at each feature 
            #separately and determines whether there is a significant relationship 
            #between that feature and the target.
            #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression
            #linear relationship between two datasets
            #The pearson correlation between each predictor and the target is computed
            #E[(X_i - X_i_bar) * (y - y_bar)] / (std(X_i) * std(y))
            #and then it is converted to an F score and then to a p-value (using Kowalski test)
            #https://towardsdatascience.com/mistakes-in-applying-univariate-feature-selection-methods-34c43ce8b93d
            from sklearn import datasets #for regression
            X, y = datasets.load_diabetes(return_X_y=True)
            
            from sklearn.feature_selection import f_regression
            f_statistic, p_values = f_regression(X, y)
        
        
        #chi2 (categorical features in a classification problem)
        if True:
            #Compute chi-squared stats between each non-negative feature and class.
            #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
            
            #chi-squre tests for independance of categorical features. The null 
            #hypothesis is that the two categorical features are independent. So 
            #high Chi-Square value indicates that the hypothesis of independence 
            #is incorrect. the higher the Chi-Square value the feature is more 
            #dependent on the response
            #https://towardsdatascience.com/mistakes-in-applying-univariate-feature-selection-methods-34c43ce8b93d
            from sklearn.datasets import load_iris #for classification
            X_iris, true_labels = load_iris(return_X_y=True)
            
            from sklearn.feature_selection import chi2
            chi2, p_values = chi2(X_iris, true_labels) #Note that the dataset do not have any categorical feature
        
        
        #f_classif (continuous features in a classification problem)
        if True:
            #Compute the ANOVA F-value for the provided sample.
            #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
            
            #In statistics, ANOVA is used to determine whether there is any statistically 
            #significant difference between the means of two or more groups. This is 
            #particularly useful in a classification problem where we want to know how well 
            #a continuous feature discriminates between multiple classes. ANOVA compares 
            #the variation between each group (classes) to the variation within each group.
            #https://towardsdatascience.com/mistakes-in-applying-univariate-feature-selection-methods-34c43ce8b93d
            from sklearn.datasets import load_iris #for classification
            X_iris, true_labels = load_iris(return_X_y=True)
            
            from sklearn.feature_selection import f_classif
            f_statistic, p_values = f_classif(X_iris, true_labels)
            
            
        #mutual_info_regression (continuous/categorical features in a regression problem)
        if True:
            #Estimate mutual information for a continuous target variable
            #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression
            
            #Mutual information (MI) between two continuous random variables is 
            #a non-negative value, which measures the dependency between the variables.
            #It is equal to zero if and only if two random variables are independent, 
            #and higher values mean higher dependency.
            from sklearn import datasets #for regression
            X, y = datasets.load_diabetes(return_X_y=True)
            
            from sklearn.feature_selection import mutual_info_regression
            mi = mutual_info_regression(X, y, 
                                        discrete_features='auto', #it's a bit different than categorical
                                        n_neighbors=3) #Number of neighbors to use for MI estimation for continuous variables. Higher values reduce variance of the estimation, but could introduce a bias.
            
            
        #mutual_info_classif (continuous/categorical features in a classification problem)
        if True:
            #Estimate mutual information for a discrete target variable.
            #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif
            from sklearn.datasets import load_iris #for classification
            X_iris, true_labels = load_iris(return_X_y=True)
            
            from sklearn.feature_selection import mutual_info_classif
            mi = mutual_info_classif(X_iris, true_labels, 
                                     discrete_features='auto', #it's a bit different than categorical
                                     n_neighbors=3) #Number of neighbors to use for MI estimation for continuous variables. Higher values reduce variance of the estimation, but could introduce a bias.
    
    
    
    #SelectKBest
    if True:
        #Select features according to the k highest scores.
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html#sklearn.feature_selection.SelectKBest
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        #chose any of the appropriate scoring from above
        from sklearn.feature_selection import SelectKBest, f_regression
        selector = SelectKBest(f_regression, k=8)
        selector.fit(X, y)
        selector.scores_ #scores
        selector.pvalues_ #associated p-values
        X_new = selector.transform(X)
    
    
    
    #SelectPercentile
    if True:
        #Select features according to a percentile of the highest scores.
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        #chose any of the appropriate scoring from above
        from sklearn.feature_selection import SelectPercentile, f_regression
        selector = SelectPercentile(f_regression, percentile=80)
        selector.fit(X, y)
        selector.scores_ #scores
        selector.pvalues_ #associated p-values
        X_new = selector.transform(X)
    
    
    
    #RFE
    if True:
        #Feature ranking with recursive feature elimination.
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.feature_selection import RFE
        from sklearn.svm import SVR
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=5, step=1) #step: int | float (# or % of features to remove at each step )
        selector.fit(X, y)
        selector.ranking_ #Selected (i.e., estimated best) features are assigned rank 1
        fitted_estimator = selector.estimator_ #The fitted estimator used to select features.
        X_new = selector.transform(X)
    
    
    
    #RFECV
    if True:
        #Recursive feature elimination with cross-validation to select features.
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.feature_selection import RFECV
        from sklearn.svm import SVR
        estimator = SVR(kernel="linear")
        selector = RFECV(estimator, min_features_to_select=2, step=1, cv=5)
        selector.fit(X, y)
        selector.ranking_ #Selected (i.e., estimated best) features are assigned rank 1
        a=selector.cv_results_ #mean | std | k-th cv results as split(k)
        selector.n_features_ #number of selected features with cross-validation
        fitted_estimator = selector.estimator_ #The fitted estimator used to select features.
        X_new = selector.transform(X)
    
    
    
    #SelectFromModel
    if True:
        #Meta-transformer for selecting features based on importance weights.
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.feature_selection import SelectFromModel
        from sklearn.svm import SVR
        estimator = SVR(kernel="linear")
        selector = SelectFromModel(estimator, #fitted or unfitted model
                                   prefit=False,
                                   max_features=8)
        selector.fit(X, y)
        fitted_estimator = selector.estimator_ #The fitted estimator used to select features.
        X_new = selector.transform(X)
    
    
    
    #SequentialFeatureSelector
    if True:
        #Transformer that performs Sequential Feature Selection.
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#sklearn.feature_selection.SequentialFeatureSelector
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.feature_selection import SequentialFeatureSelector
        from sklearn.svm import SVR
        estimator = SVR(kernel="linear")
        selector = SequentialFeatureSelector(estimator,
                                             n_features_to_select='auto', #'auto' based on tol | int | float 
                                             tol=None, #imporvement between two consecutive
                                             direction='backward', #'forward' | 'backward'
                                             scoring='r2', #model evaluation rules --> https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                             cv=5)
        selector.fit(X, y)
        selector.n_features_to_select_ #number of features that were selected
        X_new = selector.transform(X)
    
    
    


#ML Models
###############################################################################
if True:
    #LinearRegression
    ###############################################################################
    if True:
        #Ordinary least squares Linear Regression.
        #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        
        
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression(fit_intercept=True) #create a linear regression object
        reg.fit(X, y) #fit a linear regression model to X and y
        reg.score(X, y) #Return the coefficient of determination of the prediction
        reg.coef_ #return the coefficients for each variable
        reg.intercept_ #return the intercept value
        new_X = X.copy()
        y_hat = reg.predict(new_X) #Return prediction on new data
        
        import pickle
        # save the model to disk
        filename = 'finalized_model.sav'
        pickle.dump(reg, open(filename, 'wb'))
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        
        ###linear regression with statsmodels
        if True:
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
    
    
    
    #OneVsRestClassifier
    ###############################################################################
    if True:
        #One-vs-the-rest (OvR) multiclass strategy.
        #https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.linear_model import LogisticRegression
        estimator = LogisticRegression() #or any classification algorithm
        model = OneVsRestClassifier(estimator)
        model.fit(X_iris, true_labels)
        model.predict(X_iris)
        model.predict_proba(X_iris)
        model.score(X_iris, true_labels)
    
    
    
    #Support Vector Machines
    ###############################################################################
    #Support Vector Classification
    if True:
        #C-Support Vector Classification.
        #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf', # 'linear' | 'poly' | 'rbf' | 'sigmoid' | 'precomputed'
                  degree=3, # only used for kernel='poly'
                  gamma='scale', #'scale' | 'auto' Kernel coefficient for 'poly', 'rbf', 'sigmoid'
                  probability=True, # wether to calculate the probabilities (it uses 5 fold CV and may not be consistent with predict()) 
                  decision_function_shape='ovr') # 'ovr' | 'ovo'
        clf.fit(X_iris, true_labels) #fit a support vector classifier model to X and y
        y_hat = clf.predict(X_iris) #Return prediction on new data
        clf.predict_proba(X_iris) #may not be consistent with predict() | found with different methods
        clf.score(X_iris, true_labels)
        clf.support_vectors_ # get support vectors
        clf.support_ # get indices of support vectors
        clf.n_support_ # get number of support vectors for each class
    
    
    
    #Support Vector Regression
    if True:
        #Epsilon-Support Vector Regression.
        #https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.svm import SVR
        reg = SVR(kernel='poly', #'linear' | 'poly' | 'rbf' | 'sigmoid' | 'precomputed'
                  degree=6, #only used for kernel='poly'
                  coef0=10, #only used for kernel='poly' or 'sigmoid'
                  gamma='scale', #'scale' | 'auto' Kernel coefficient for 'poly', 'rbf', 'sigmoid'
                  C=7.0, #Regularization parameter; strength of the regularization is inversely proportional to C
                  epsilon=10.3) #specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
    
    
    
    #NearestNeighbors
    ###############################################################################
    #Finding the Nearest Neighbors
    if True:
        #Unsupervised learner for implementing neighbor searches.
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=2, #k
                                algorithm='auto', #'auto' | 'ball_tree' | 'kd_tree' | 'brute'
                                leaf_size=30, #used in 'ball_tree' and 'kd_tree'
                                metric='minkowski', #if use p=2 it would be euclidian
                                p=2)
        nbrs.fit(X_iris)
        distances, indices = nbrs.kneighbors(X_iris[0:2], 5, return_distance=True) #Find the K-neighbors of a point
    
    
    
    #Nearest Neighbors Classification
    if True:
        #Classifier implementing the k-nearest neighbors vote.
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=5, #k
                                   weights='uniform', #'uniform' | 'distance': Weight function used in prediction
                                   algorithm='auto', #'auto' | 'ball_tree' | 'kd_tree' | 'brute'
                                   leaf_size=30, #used in 'ball_tree' and 'kd_tree'
                                   metric='minkowski', #if use p=2 it would be euclidian
                                   p=2)
        clf.fit(X_iris, true_labels)
        clf.predict(X_iris)
        clf.predict_proba(X_iris)
        clf.score(X_iris, true_labels)
    
    
    
    #Radius Neighbors Classification
    if True:
        #Classifier implementing a vote among neighbors within a given radius.
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html#sklearn.neighbors.RadiusNeighborsClassifier
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        
        from sklearn.neighbors import RadiusNeighborsClassifier
        clf = RadiusNeighborsClassifier(radius=1, #Range of parameter space
                                        weights='uniform', #'uniform' | 'distance': Weight function used in prediction
                                        algorithm='auto', #'auto' | 'ball_tree' | 'kd_tree' | 'brute'
                                        leaf_size=30, #used in 'ball_tree' and 'kd_tree'
                                        outlier_label=None, #manual label | 'most_frequent' | None
                                        metric='minkowski', #if use p=2 it would be euclidian
                                        p=2)
        clf.fit(X_iris, true_labels)
        clf.predict(X_iris)
        clf.predict_proba(X_iris)
        clf.score(X_iris, true_labels)
    
    
    
    #Nearest Neighbors Regression
    if True:
        #Regression based on k-nearest neighbors.
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.neighbors import KNeighborsRegressor
        reg = KNeighborsRegressor(n_neighbors=5, #k
                                  weights='uniform', #'uniform' | 'distance': Weight function used in prediction
                                  algorithm='auto', #'auto' | 'ball_tree' | 'kd_tree' | 'brute'
                                  leaf_size=30, #used in 'ball_tree' and 'kd_tree'
                                  metric='minkowski', #if use p=2 it would be euclidian
                                  p=2)
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
    
    
    
    #Radius Neighbors Regression
    if True:
        #Regression based on neighbors within a fixed radius.
        #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.neighbors import RadiusNeighborsRegressor
        reg = RadiusNeighborsRegressor(radius=0.035, #Range of parameter space
                                       weights='uniform', #'uniform' | 'distance': Weight function used in prediction
                                       algorithm='auto', #'auto' | 'ball_tree' | 'kd_tree' | 'brute'
                                       leaf_size=30, #used in 'ball_tree' and 'kd_tree'
                                       metric='minkowski', #if use p=2 it would be euclidian
                                       p=2)
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
    
    
    
    #Decision Trees
    ###############################################################################
    #DecisionTreeClassifier
    if True:
        #A decision tree classifier. In case of multiple output (feed the y that has multiple columns)
        #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion='gini', #'gini' | 'entropy' | 'log_loss': measure the quality of a split
                                     max_depth=None, #maximum depth of the tree
                                     min_samples_split=2, #minimum number of samples required to split
                                     min_samples_leaf=1, #minimum number of samples required to be at a leaf node
                                     max_features=None, #int | float | None or {'auto' | 'sqrt' | 'log2'} : number of features to consider split
                                     min_impurity_decrease=0.0, 
                                     class_weight=None, #Weights associated with classes in the form {class_label: weight}
                                     ccp_alpha=0.0) #Complexity parameter used for Minimal Cost-Complexity Pruning
        clf.fit(X_iris, true_labels)
        clf.predict(X_iris)
        clf.predict_proba(X_iris)
        clf.score(X_iris, true_labels)
        clf.feature_importances_ #feature importances
        
        #plotting the decision tree
        from sklearn import tree
        tree.plot_tree(clf)
        import graphviz
        dot_data = tree.export_graphviz(clf, out_file=None, 
                                        filled=True, 
                                        rounded=True, 
                                        special_characters=True)
        graphviz.Source(dot_data)
        
        #printing the tree
        print(tree.export_text(clf))
    
    
    
    #DecisionTreeRegressor
    if True:
        #A decision tree regressor. In case of multiple output (feed the y that has multiple columns)
        #https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.tree import DecisionTreeRegressor
        reg = DecisionTreeRegressor(criterion='squared_error', #'squared_error' | 'friedman_mse' | 'absolute_error' | 'poisson': measure the quality of a split
                                    max_depth=None, #maximum depth of the tree
                                    min_samples_split=2, #minimum number of samples required to split
                                    min_samples_leaf=1, #minimum number of samples required to be at a leaf node
                                    max_features=None, #int | float | None or {'auto' | 'sqrt' | 'log2'} : number of features to consider split
                                    min_impurity_decrease=0.0, 
                                    ccp_alpha=0.0) #Complexity parameter used for Minimal Cost-Complexity Pruning
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
        reg.feature_importances_ #feature importances
        
        #plotting the decision tree
        from sklearn import tree
        tree.plot_tree(reg)
        import graphviz
        dot_data = tree.export_graphviz(reg, out_file=None, 
                                        filled=True, 
                                        rounded=True, 
                                        special_characters=True)
        graphviz.Source(dot_data)
        
        #printing the tree
        print(tree.export_text(reg))
    
    
    
    #Bagging
    ###############################################################################
    #BaggingClassifier
    if True:
        #A Bagging classifier.
        #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.ensemble import BaggingClassifier #by default it works base off of Decision Tree Classifier
        clf = BaggingClassifier(n_estimators=10, #number of base estimators
                                max_samples=1.0, #number of samples to draw from X to train each base estimator
                                max_features=1.0, #number of features to draw from X to train each base estimator
                                bootstrap=True, #Whether samples are drawn with replacement
                                bootstrap_features=False, #Whether features are drawn with replacement
                                oob_score=False, #Whether to use out-of-bag samples to estimate the generalization error
                                warm_start=False) #When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble
        clf.fit(X_iris, true_labels)
        clf.predict(X_iris)
        clf.predict_proba(X_iris)
        clf.score(X_iris, true_labels)
    
    
    
    #BaggingRegressor
    if True:
        #A Bagging regressor.
        #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.ensemble import BaggingRegressor #by default it works base off of Decision Tree Regressor
        reg = BaggingRegressor(n_estimators=10, #number of base estimators
                               max_samples=1.0, #number of samples to draw from X to train each base estimator
                               max_features=1.0, #number of features to draw from X to train each base estimator
                               bootstrap=True, #Whether samples are drawn with replacement
                               bootstrap_features=False, #Whether features are drawn with replacement
                               oob_score=False, #Whether to use out-of-bag samples to estimate the generalization error
                               warm_start=False) #When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new ensemble
        
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
    
    
    
    #Random Forest
    ###############################################################################
    #RandomForestClassifier
    if True:
        #A random forest classifier.
        #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, #number of trees in the forest
                                     criterion='gini', #'gini' | 'entropy' | 'log_loss'
                                     max_depth=None, #None | int: maximum depth of the tree
                                     min_samples_split=2, #minimum number of samples required to split an internal node:
                                     min_samples_leaf=1, #minimum number of samples required to be at a leaf node
                                     max_features='sqrt', #'sqrt' | 'log2' | None | int | float: number of features to consider when looking for the best split
                                     bootstrap=True, #Whether bootstrap samples
                                     oob_score=False, #Whether to use out-of-bag samples to estimate the generalization score
                                     warm_start=False, #When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest
                                     class_weight=None, #'balanced' | 'balanced_subsample' | dict | None: Weights associated with classes
                                     ccp_alpha=0.0) #Complexity parameter used for Minimal Cost-Complexity Pruning
        clf.fit(X_iris, true_labels)
        clf.predict(X_iris)
        clf.predict_proba(X_iris)
        clf.score(X_iris, true_labels)
        clf.feature_importances_ #impurity-based feature importances
    
    
    
    #RandomForestRegressor
    if True:
        #A random forest regressor.
        #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.ensemble import RandomForestRegressor
        reg = RandomForestRegressor(n_estimators=100, #number of trees in the forest
                                     criterion='squared_error', #'squared_error' | 'bsolute_error' | 'poisson'
                                     max_depth=None, #None | int: maximum depth of the tree
                                     min_samples_split=2, #minimum number of samples required to split an internal node:
                                     min_samples_leaf=1, #minimum number of samples required to be at a leaf node
                                     max_features=1.0, #'sqrt' | 'log2' | None | int | float: number of features to consider when looking for the best split
                                     bootstrap=True, #Whether bootstrap samples
                                     oob_score=False, #Whether to use out-of-bag samples to estimate the generalization score
                                     warm_start=False, #When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest
                                     ccp_alpha=0.0) #Complexity parameter used for Minimal Cost-Complexity Pruning
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
        reg.feature_importances_ #impurity-based feature importances
    
    
    
    #AdaBoost
    ###############################################################################
    #AdaBoostClassifier
    if True:
        #An AdaBoost classifier.
        #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from sklearn.ensemble import AdaBoostClassifier
        #There is a trade-off between the learning_rate and n_estimators parameters
        clf = AdaBoostClassifier(base_estimator=None, #object | None (DecisionTreeClassifier by default))
                                 n_estimators=50, #maximum number of estimators at which boosting is terminated
                                 learning_rate=1.0) #Weight applied to each classifier at each boosting iteration
        clf.fit(X_iris, true_labels)
        clf.predict(X_iris)
        clf.predict_proba(X_iris)
        clf.score(X_iris, true_labels)
        clf.feature_importances_ #impurity-based feature importances
    
    
        
    #AdaBoostRegressor
    if True:
        #An AdaBoost regressor.
        #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.ensemble import AdaBoostRegressor
        #There is a trade-off between the learning_rate and n_estimators parameters
        reg = AdaBoostRegressor(base_estimator=None, #object | None (DecisionTreeRegressor by default))
                                 n_estimators=50, #maximum number of estimators at which boosting is terminated
                                 loss='linear', #'linear' | 'square' | 'exponential': loss function to use when updating the weights after each boosting iteration
                                 learning_rate=1.0) #Weight applied to each classifier at each boosting iteration
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
        reg.feature_importances_ #impurity-based feature importances
    
    
    
    #XGBoost
    ###############################################################################
    #XGBClassifier
    if True:
        #Implementation of the scikit-learn API for XGBoost classification.
        #https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
        from sklearn.datasets import load_iris #for classification
        X_iris, true_labels = load_iris(return_X_y=True)
        
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=100, #Number of boosting rounds
                            max_depth=None, #Maximum tree depth for base learners
                            max_leaves=None, #0 or None (no limit) | int: Maximum number of leaves
                            learning_rate=0.3, #eta
                            booster='gbtree', #'gbtree' | 'gblinear' | 'dart'
                            gamma=0, #Minimum loss reduction to make a split at each node
                            reg_alpha=0, #L1 regularization term on weights
                            reg_lambda=1, #L2 regularization term on weights
                            missing=np.nan, #Value in the data which needs to be present as a missing value
                            importance_type=None, #'gain' | 'weight' | 'cover' | 'total_gain' | 'total_cover'
                            eval_metric=None) #Metric used for monitoring the training result and early stopping - changes the objective function (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
        
        clf.get_params(deep=True)
        clf.fit(X_iris, true_labels)
        clf.predict(X_iris)
        clf.predict_proba(X_iris)
        clf.score(X_iris, true_labels)
        clf.feature_importances_ #impurity-based feature importances
        clf.save_model("model.json")
        clf.load_model("model.json")
    
    
        
    #XGBRegressor
    if True:
        #Implementation of the scikit-learn API for XGBoost regression.
        #https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from xgboost import XGBRegressor
        reg = XGBRegressor(n_estimators=100, #Number of boosting rounds
                           max_depth=None, #Maximum tree depth for base learners
                           max_leaves=None, #0 or None (no limit) | int: Maximum number of leaves
                           learning_rate=0.3, #eta
                           booster='gbtree', #'gbtree' | 'gblinear' | 'dart'
                           gamma=0, #Minimum loss reduction to make a split at each node
                           reg_alpha=0, #L1 regularization term on weights
                           reg_lambda=1, #L2 regularization term on weights
                           missing=np.nan, #Value in the data which needs to be present as a missing value
                           importance_type=None, #'gain' | 'weight' | 'cover' | 'total_gain' | 'total_cover'
                           eval_metric=None) #Metric used for monitoring the training result and early stopping - changes the objective function (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
        
        reg.get_params(deep=True)
        reg.fit(X, y)
        y_hat = reg.predict(X)
        reg.score(X, y) #R2
        reg.feature_importances_ #impurity-based feature importances
        reg.save_model("model.json")
        reg.load_model("model.json")





#Evaluation Metrics
###############################################################################
if True:
    #Regression Metrics
    ###############################################################################
    #sample data
    if True:
        import numpy as np
        y_true = np.array([151.,  75., 141., 206., 135.,  97., 138.,  63., 110., 310., 101.,
                69., 179., 185., 118., 171., 166., 144.,  97., 168.,  68.,  49.,
                68., 245., 184., 202., 137.,  85., 131., 283., 129.,  59., 341.,
                87.,  65., 102., 265., 276., 252.,  90., 100.,  55.,  61.,  92.,
               259.,  53., 190., 142.,  75., 142., 155., 225.,  59., 104., 182.,
               128.,  52.,  37., 170., 170.,  61., 144.,  52., 128.,  71., 163.,
               150.,  97., 160., 178.,  48., 270., 202., 111.,  85.,  42., 170.,
               200., 252., 113., 143.,  51.,  52., 210.,  65., 141.,  55., 134.,
                42., 111.,  98., 164.,  48.,  96.,  90., 162., 150., 279.,  92.,
                83., 128., 102., 302., 198.,  95.,  53., 134., 144., 232.,  81.,
               104.,  59., 246., 297., 258., 229., 275., 281., 179., 200., 200.,
               173., 180.,  84., 121., 161.,  99., 109., 115., 268., 274., 158.,
               107.,  83., 103., 272.,  85., 280., 336., 281., 118., 317., 235.,
                60., 174., 259., 178., 128.,  96., 126., 288.,  88., 292.,  71.,
               197., 186.,  25.,  84.,  96., 195.,  53., 217., 172., 131., 214.,
                59.,  70., 220., 268., 152.,  47.,  74., 295., 101., 151., 127.,
               237., 225.,  81., 151., 107.,  64., 138., 185., 265., 101., 137.,
               143., 141.,  79., 292., 178.,  91., 116.,  86., 122.,  72., 129.,
               142.,  90., 158.,  39., 196., 222., 277.,  99., 196., 202., 155.,
                77., 191.,  70.,  73.,  49.,  65., 263., 248., 296., 214., 185.,
                78.,  93., 252., 150.,  77., 208.,  77., 108., 160.,  53., 220.,
               154., 259.,  90., 246., 124.,  67.,  72., 257., 262., 275., 177.,
                71.,  47., 187., 125.,  78.,  51., 258., 215., 303., 243.,  91.,
               150., 310., 153., 346.,  63.,  89.,  50.,  39., 103., 308., 116.,
               145.,  74.,  45., 115., 264.,  87., 202., 127., 182., 241.,  66.,
                94., 283.,  64., 102., 200., 265.,  94., 230., 181., 156., 233.,
                60., 219.,  80.,  68., 332., 248.,  84., 200.,  55.,  85.,  89.,
                31., 129.,  83., 275.,  65., 198., 236., 253., 124.,  44., 172.,
               114., 142., 109., 180., 144., 163., 147.,  97., 220., 190., 109.,
               191., 122., 230., 242., 248., 249., 192., 131., 237.,  78., 135.,
               244., 199., 270., 164.,  72.,  96., 306.,  91., 214.,  95., 216.,
               263., 178., 113., 200., 139., 139.,  88., 148.,  88., 243.,  71.,
                77., 109., 272.,  60.,  54., 221.,  90., 311., 281., 182., 321.,
                58., 262., 206., 233., 242., 123., 167.,  63., 197.,  71., 168.,
               140., 217., 121., 235., 245.,  40.,  52., 104., 132.,  88.,  69.,
               219.,  72., 201., 110.,  51., 277.,  63., 118.,  69., 273., 258.,
                43., 198., 242., 232., 175.,  93., 168., 275., 293., 281.,  72.,
               140., 189., 181., 209., 136., 261., 113., 131., 174., 257.,  55.,
                84.,  42., 146., 212., 233.,  91., 111., 152., 120.,  67., 310.,
                94., 183.,  66., 173.,  72.,  49.,  64.,  48., 178., 104., 132.,
               220.,  57.])
        
        y_pred = np.array([206.11667725,  68.07103297, 176.88279035, 166.91445843,
               128.46225834, 106.35191443,  73.89134662, 118.85423042,
               158.80889721, 213.58462442,  97.07481511,  95.10108423,
               115.06915952, 164.67656842, 103.07814257, 177.17487964,
               211.7570922 , 182.84134823, 148.00326937, 124.01754066,
               120.33362197,  85.80068961, 113.1134589 , 252.45225837,
               165.48779206, 147.71997564,  97.12871541, 179.09358468,
               129.05345958, 184.7811403 , 158.71516713,  69.47575778,
               261.50385365, 112.82234716,  78.37318279,  87.66360785,
               207.92114668, 157.87641942, 240.84708073, 136.93257456,
               153.48044608,  74.15426666, 145.62742227,  77.82978811,
               221.07832768, 125.21957584, 142.6029986 , 109.49562511,
                73.14181818, 189.87117754, 157.9350104 , 169.55699526,
               134.1851441 , 157.72539008, 139.11104979,  72.73116856,
               207.82676612,  80.11171342, 104.08335958, 134.57871054,
               114.23552012, 180.67628279,  61.12935368,  98.72404613,
               113.79577026, 189.95771575, 148.98351571, 124.34152283,
               114.8395504 , 121.99957578,  73.91017087, 236.71054289,
               142.31126791, 124.51672384, 150.84073896, 127.75230658,
               191.16896496,  77.05671154, 166.82164929,  91.00591229,
               174.75156797, 122.83451589,  63.27231315, 151.99867317,
                53.72959077, 166.0050229 ,  42.6491333 , 153.04229493,
                80.54701716, 106.90148495,  79.93968011, 187.1672654 ,
               192.5989033 ,  61.07398313, 107.4076912 , 125.04307496,
               207.72402726, 214.21248827, 123.47464895, 139.16439034,
               168.21372017, 106.92902558, 150.64748328, 157.92364009,
               152.75958287, 116.22381927,  73.03167734, 155.67052006,
               230.1417777 , 143.49797317,  38.09587272, 121.8593267 ,
               152.79404663, 207.99702587, 291.23106133, 189.17571129,
               214.02877593, 235.18106509, 165.38480498, 151.2469168 ,
               156.57659557, 200.44066818, 219.35193167, 174.78830391,
               169.23118221, 187.87537099,  57.49340026, 108.54836058,
                92.68731024, 210.87347343, 245.47097701,  69.84285129,
               113.03485904,  68.42650654, 141.69639374, 239.46240737,
                58.37858726, 235.47123197, 254.92309543, 253.30708899,
               155.51063293, 230.55961445, 170.44330954, 117.9953395 ,
               178.55406527, 240.07119308, 190.33892524, 228.66470581,
               114.24456339, 178.36552308, 209.091817  , 144.85615197,
               200.65926745, 121.34295733, 150.50993019, 199.01879825,
               146.27926469, 124.02163345,  85.25913019, 235.16173729,
                82.1730808 , 231.29474031, 144.36940116, 197.04628448,
               146.99841953,  77.18813284,  59.37368356, 262.68557988,
               225.12900796, 220.20301952,  46.59651844,  88.10194612,
               221.77450036,  97.25199783, 164.48838425, 119.90096817,
               157.80220788, 223.08012207,  99.59081773, 165.84386951,
               179.47680741,  89.83353846, 171.82590335, 158.36419935,
               201.48185539, 186.39194958, 197.47424761,  66.57371647,
               154.59985312, 116.18319159, 195.91755793, 128.04834496,
                91.20395862, 140.57223765, 155.22669143, 169.70326581,
                98.7573858 , 190.14568824, 142.51704894, 177.27157771,
                95.30812216,  69.06191507, 164.16391317, 198.0659024 ,
               178.25996632, 228.58539684, 160.67104137, 212.28734795,
               222.4833913 , 172.85421282, 125.27946793, 174.72103207,
               152.38094643,  98.58135665,  99.73771331, 262.29507095,
               223.74033222, 221.33976142, 133.61470602, 145.42828204,
                53.04569008, 141.82052358, 153.68617582, 125.22290891,
                77.25168449, 230.26180811,  78.9090807 , 105.2051755 ,
               117.99622779,  99.06233889, 166.55796947, 159.34137227,
               158.27448255, 143.05684078, 231.55890118, 176.64724258,
               187.23580712,  65.39099908, 190.66218796, 179.75181691,
               234.9080532 , 119.15669025,  85.63551834, 100.8597527 ,
               140.41937377, 101.83524022, 120.66560385,  83.0664276 ,
               234.58488012, 245.15862773, 263.26954282, 274.87127261,
               180.67257769, 203.05642297, 254.21625849, 118.44300922,
               268.45369506, 104.83843473, 115.86820464, 140.45857194,
                58.46948192, 129.83145265, 263.78607272,  45.00934573,
               123.28890007, 131.0856888 ,  34.89181681, 138.35467112,
               244.30103923,  89.95923929, 192.07096194, 164.33017386,
               147.74779723, 191.89092557, 176.44360299, 158.3490221 ,
               189.19166962, 116.58117777, 111.449754  , 117.45232726,
               165.79598354,  97.80405886, 139.54451791,  84.17319946,
               159.93677518, 202.39971737,  80.48131518, 146.64558568,
                79.05314048, 191.33777472, 220.67516721, 203.75017281,
                92.86459928, 179.15576252,  81.79874055, 152.8290929 ,
                76.80052219,  97.79590831, 106.8371012 , 123.83461591,
               218.13908293, 126.01937664, 206.7587966 , 230.5767944 ,
               122.05921633, 135.67824405, 126.37042532, 148.49374458,
                88.07147107, 138.95823614, 203.8691938 , 172.55288732,
               122.95701477, 213.92310163, 174.89158814, 110.07294222,
               198.36584973, 173.25229067, 162.64748776, 193.31578983,
               191.53493643, 284.13932209, 279.31133207, 216.00823829,
               210.08668656, 216.21612991, 157.01450004, 224.06431372,
               189.06103154, 103.56515315, 178.70270016, 111.81862434,
               291.00196609, 182.64651752,  79.33315426,  86.33029851,
               249.1510082 , 174.51537682, 122.10291074, 146.2718871 ,
               170.65483847, 183.497196  , 163.36806262, 157.03297709,
               144.42614949, 125.30053093, 177.50251197, 104.57681546,
               132.17560518,  95.06210623, 249.89755705,  86.23824126,
                61.99847009, 156.81295053, 192.32218372, 133.85525804,
                93.67249793, 202.49572354,  52.54148927, 174.82799914,
               196.91468873, 118.06336979, 235.29941812, 165.09438096,
               160.41761959, 162.37786753, 254.05587268, 257.23492156,
               197.5039462 , 184.06877122,  58.62131994, 194.39216636,
               110.775815  , 142.20991224, 128.82520996, 180.13082199,
               211.26488624, 169.59494046, 164.33851796, 136.23374077,
               174.51001028,  74.67587343, 246.29432383, 114.14494406,
               111.54552901, 140.0224376 , 109.99895704,  91.37283987,
               163.01540596,  75.16804478, 254.06119047,  53.47338214,
                98.48397565, 100.66315554, 258.58683032, 170.67256752,
                61.91771186, 182.31148421, 171.26948629, 189.19505093,
               187.18494664,  87.12170524, 148.37964317, 251.35815403,
               199.69656904, 283.63576862,  50.85911237, 172.14766276,
               204.05976093, 174.16540137, 157.93182911, 150.50028158,
               232.97445368, 121.5814873 , 164.54245461, 172.67625919,
               226.7768891 , 149.46832104,  99.13924946,  80.43418456,
               140.16148637, 191.90710484, 199.28001608, 153.63277325,
               171.80344337, 112.11054883, 162.60002916, 129.84290324,
               258.03100468, 100.70810916, 115.87608197, 122.53559675,
               218.1797988 ,  60.94350929, 131.09296884, 119.48376601,
                52.60911672, 193.01756549, 101.05581371, 121.22668124,
               211.85894518,  53.44727472])
    
    #common metrics
    if True:
        #coefficient of determination
        from sklearn.metrics import r2_score
        r2_score(y_true, y_pred)
        
        #mean squared error | root mean squared error
        from sklearn.metrics import mean_squared_error
        mean_squared_error(y_true, y_pred)
        mean_squared_error(y_true, y_pred, squared=False)
        
        #mean squared logarithmic error | root mean squared logarithmic error
        from sklearn.metrics import mean_squared_log_error
        mean_squared_log_error(y_true, y_pred)
        mean_squared_log_error(y_true, y_pred, squared=False)
        
        #mean absolute percentage error
        from sklearn.metrics import mean_absolute_percentage_error
        mean_absolute_percentage_error(y_true, y_pred)
        
        #max error
        from sklearn.metrics import max_error
        max_error(y_true, y_pred)
    
    
    
    #Classification Metrics (works with binary and multi_class)
    ###############################################################################
    #sample data
    if True:
        y_true = [0, 2, 1, 3, 0, 3, 2, 1, 0, 0, 1, 1, 3, 3, 2, 2, 0, 0, 3, 3]
        y_pred = [0, 2, 2, 3, 0, 1, 2, 1, 0, 0, 1, 2, 3, 3, 2, 2, 2, 0, 3, 2]
    
    #common metrics
    if True:
        from sklearn.metrics import accuracy_score #classification score
        accuracy_score(y_true, y_pred) #return the ratio of correctly classified
        accuracy_score(y_true, y_pred, normalize=False) #return the number of correctly classified
        
        
        #confusion matrix
        from sklearn.metrics import confusion_matrix
        confusion_matrix(y_true, y_pred)
        
        
        
        #Precision = tp / (tp + fp)
        #https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826#:~:text=True%20Positive%20(TP)%3A%20It,the%20positive%20class%20as%20positive.
        from sklearn.metrics import precision_score
        #works with binary only
        precision_score(y_true, y_pred, pos_label=1)
        
        #by counting the total true positives, false negatives and false positives
        #total tp = sum(all the diag elements)
        #total fn = total fp = sum(all the off-diag elements)
        precision_score(y_true, y_pred, average='micro')
        
        #Calculate metrics for each label, and find their unweighted mean. This 
        #does not take label imbalance into account
        precision_score(y_true, y_pred, average='macro')
        
        #Calculate metrics for each label, and find weighted average by 
        #(the number of true instances for each label).
        #This alters ‘macro’ to account for label imbalance
        precision_score(y_true, y_pred, average='weighted')
        
        
        
        #Recall = tp / (tp + fn)
        from sklearn.metrics import recall_score
        #works with binary only
        recall_score(y_true, y_pred, pos_label=1)
        
        #by counting the total true positives, false negatives and false positives
        #total tp = sum(all the diag elements)
        #total fn = total fp = sum(all the off-diag elements)
        recall_score(y_true, y_pred, average='micro')
        
        #Calculate metrics for each label, and find their unweighted mean. This 
        #does not take label imbalance into account
        recall_score(y_true, y_pred, average='macro')
        
        #Calculate metrics for each label, and find weighted average by 
        #(the number of true instances for each label).
        #This alters ‘macro’ to account for label imbalance
        recall_score(y_true, y_pred, average='weighted')
        
        
        
        #F1-Score = 2 * (precision * recall) / (precision + recall)
        from sklearn.metrics import f1_score
        #works with binary only
        f1_score(y_true, y_pred, pos_label=1)
        
        #by counting the total true positives, false negatives and false positives
        #total tp = sum(all the diag elements)
        #total fn = total fp = sum(all the off-diag elements)
        f1_score(y_true, y_pred, average='micro')
        
        #Calculate metrics for each label, and find their unweighted mean. This 
        #does not take label imbalance into account
        f1_score(y_true, y_pred, average='macro')
        
        #Calculate metrics for each label, and find weighted average by 
        #(the number of true instances for each label).
        #This alters ‘macro’ to account for label imbalance
        f1_score(y_true, y_pred, average='weighted')
        
        
        
        #F-beta-Score (optimal value at 1 and its worst value at 0)
        from sklearn.metrics import fbeta_score
        #The beta parameter determines the weight of recall in the combined score. 
        #beta < 1 lends more weight to precision, while beta > 1 favors recall 
        #(beta -> 0 considers only precision, beta -> +inf only recall)
        #works with binary only
        fbeta_score(y_true, y_pred, pos_label=1)
        
        #by counting the total true positives, false negatives and false positives
        #total tp = sum(all the diag elements)
        #total fn = total fp = sum(all the off-diag elements)
        fbeta_score(y_true, y_pred, average='micro')
        
        #Calculate metrics for each label, and find their unweighted mean. This 
        #does not take label imbalance into account
        fbeta_score(y_true, y_pred, average='macro')
        
        #Calculate metrics for each label, and find weighted average by 
        #(the number of true instances for each label).
        #This alters ‘macro’ to account for label imbalance
        fbeta_score(y_true, y_pred, average='weighted')
    
    
    
    #Classification Metrics (Binary Only)
    ###############################################################################
    #sample data
    if True:
        y_true = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        y_pred = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
               0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
        prob = np.array([0.01818917, 0.02824932, 0.01454129, 0.02369606, 0.01460758,
               0.02952693, 0.01310771, 0.02358782, 0.02014746, 0.03084765,
               0.02348006, 0.02458653, 0.02542159, 0.00802249, 0.01178319,
               0.0132213 , 0.01187461, 0.01843194, 0.04338363, 0.01587505,
               0.05319087, 0.01824517, 0.00399024, 0.04764113, 0.04815543,
               0.04839763, 0.03035673, 0.02505102, 0.02262874, 0.0287647 ,
               0.03569169, 0.03500401, 0.01162027, 0.01092747, 0.03125401,
               0.01531127, 0.0210141 , 0.01312693, 0.01413853, 0.02587445,
               0.01335831, 0.0376937 , 0.01097907, 0.02751953, 0.03975048,
               0.02609948, 0.01967995, 0.01664675, 0.02140038, 0.02130393,
               0.99758551, 0.99333152, 0.99855038, 0.98392334, 0.99710826,
               0.99203574, 0.99482494, 0.85204883, 0.99692938, 0.95683568,
               0.94387017, 0.98351594, 0.9907704 , 0.99614372, 0.92496376,
               0.99435094, 0.98902201, 0.98334543, 0.99774983, 0.97572401,
               0.00513496, 0.98275973, 0.9988118 , 0.99651304, 0.99258175,
               0.99453433, 0.99861209, 0.00119005, 0.99336251, 0.93808467,
               0.97053507, 0.96263388, 0.97443978, 0.00126506, 0.98675869,
               0.98764893, 0.99721523, 0.9969878 , 0.97193826, 0.97932759,
               0.99048599, 0.99448032, 0.98203345, 0.87800671, 0.98465175,
               0.97933598, 0.98201456, 0.99104569, 0.75570166, 0.98007919])
    
    #common metrics
    if True:
        #TN | FP | FN | TP (works with binary only)
        from sklearn.metrics import confusion_matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        
        #Receiver operating characteristic (ROC)
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, prob, pos_label=1) #returns false/true positive rates
        
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
        
        
        #Area under the ROC curve
        from sklearn.metrics import auc
        area = auc(fpr, tpr) #it works with any x, y pairs
    




#Cross-validation
###############################################################################
if True:
    #cross-validation iterators
    if True:
        #the iterators create the set of train and test sets
        
        #KFold
        if True:
            #K-Folds cross-validator
            #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold
            from sklearn import datasets #for regression
            X, y = datasets.load_diabetes(return_X_y=True)
            
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=5, shuffle=True, random_state=11) #without shuffling by default
            kf.get_n_splits() #get the number of split
            kf.split(X) #return an iterator for the indices of the train and test split
            for train_index, test_index in kf.split(X):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
            
        #RepeatedKFold
        if True:
            #Repeated K-Fold cross validator.
            #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold
            from sklearn import datasets #for regression
            X, y = datasets.load_diabetes(return_X_y=True)
            
            from sklearn.model_selection import RepeatedKFold
            rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=11)
            rkf.get_n_splits() #get the number of split
            rkf.split(X) #return an iterator for the indices of the train and test split
            for train_index, test_index in rkf.split(X):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
            
        #StratifiedKFold
        if True:
            #Stratified K-Folds cross-validator (preserving the percentage of samples for each class)
            #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
            from sklearn.datasets import load_iris #for classification
            X_iris, true_labels = load_iris(return_X_y=True)
            
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=5, shuffle=False, random_state=11) #without shuffling by default
            skf.get_n_splits() #get the number of split
            skf.split(X_iris, true_labels) #return an iterator for the indices of the train and test split
            for train_index, test_index in skf.split(X_iris, true_labels):
                print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = X_iris[train_index], X_iris[test_index]
                y_train, y_test = true_labels[train_index], true_labels[test_index]
        
    
    
    #cross_validate
    if True:
        #Evaluate metric(s) by cross-validation and also record fit/score times.
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.tree import DecisionTreeRegressor
        estimator = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
        
        from sklearn.model_selection import RepeatedKFold
        rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=11)
        
        from sklearn.model_selection import cross_validate
        scores = cross_validate(estimator, X, y, scoring=('r2', 'neg_mean_squared_error'), #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                return_train_score=True,
                                cv=rkf) #int | cross-validation iterator: the strategy or the number of cross-validation splitting
        
        print("%0.2f mean squared error with a standard deviation of %0.2f" % (scores['train_neg_mean_squared_error'].mean(), scores['train_neg_mean_squared_error'].std()))
        print("%0.2f mean R2 with a standard deviation of %0.2f" % (scores['train_r2'].mean(), scores['train_r2'].std()))
        
        print("%0.2f mean squared error with a standard deviation of %0.2f" % (scores['test_neg_mean_squared_error'].mean(), scores['test_neg_mean_squared_error'].std()))
        print("%0.2f mean R2 with a standard deviation of %0.2f" % (scores['test_r2'].mean(), scores['test_r2'].std()))
    
    
    
    #cross_val_score (works for one score only compared to "cross_validate")
    if True:
        #Evaluate a score by cross-validation.
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.tree import DecisionTreeRegressor
        reg = DecisionTreeRegressor(criterion='squared_error')
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(reg, #estimator
                                 X, y, 
                                 scoring='neg_mean_squared_error', #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                                 cv=5) #int | cross-validation iterator: the strategy or the number of cross-validation splitting
        
        print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    
    
    
    #
    if True:
        #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
        from sklearn import datasets #for regression
        X, y = datasets.load_diabetes(return_X_y=True)
        
        from sklearn.tree import DecisionTreeRegressor
        
        from sklearn.model_selection import GridSearchCV
        
        search_result = GridSearchCV(estimator, 
                                     param_grid,
                                     scoring=None, 
                                     n_jobs=None, 
                                     refit=True, 
                                     cv=None, 
                                     verbose=0, 
                                     pre_dispatch='2*n_jobs', 
                                     error_score=nan, 
                                     return_train_score=False)











