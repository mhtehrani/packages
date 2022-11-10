"""
pandas package
"""

# importing the package
import pandas as pd

#checking the panda version
print(pd.__version__)



###############################################################################
###############################################################################
###############################################################################
if True:
    #Series()
    if True:
        # Creating and/or initializing a Series: the shape would be (n,)
        # empty series
        series = pd.Series() #default dtype is object
        # empty series with fixed datatype
        series = pd.Series(dtype='float64')
        # filled series with an integer
        series = pd.Series(5)
        # filled series with a list
        series = pd.Series([1, 2, 3])
        # filled series with numpy array
        import numpy as np
        arr = np.array([1,2,3,4,5,6,7,8,9])
        series = pd.Series(arr, dtype=np.float32) #arr must be 1 dimensional
        # filled series with a list of list
        series = pd.Series([[1, 2], [3, 4]])
        series = pd.Series([1, 2.2]) # upcasting"
        # Assigning labels to the Series indicies
        series = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        series = pd.Series([1, 2, 3], index=['a', 8, 0.3])
        # Creating labeled series using dictionary
        series = pd.Series({'a':1, 'b':2, 'c':3})
        series = pd.Series({'b':2, 'a':1, 'c':3})
    
    
    
    #retrieving from Series (exactly like a python list)
    if True:
        series = pd.Series({'b':2, 'a':1, 'c':3, 'd':12, 'e':124, 'z':0})
        series['a']
        series[1]
        series[:2]
        series[2:4] #slicing
        series[-2] #the second to the last element
        series[-2:] #the last two elements
        s = pd.Series([[1, 2], [3, 4]])
        s[1][0]
    
    
    
    #dtypes | astype()
    if True:
        #Series dtypes
        series = pd.Series({'b':2, 'a':1, 'c':3, 'd':12, 'e':124, 'z':0})
        series.dtypes
        series = pd.Series({'b':2, 'a':1, 'c':3, 'd':12, 'e':124, 'z':0, 'ab':'we'})
        series.dtypes #dtype('O') is object element
        
        #casting dtype
        series = pd.Series({'b':2, 'a':1, 'c':3, 'd':12, 'e':124, 'z':0, 'x':-3, 'y':-34, 'v':-98, 'g':-98})
        series.dtypes
        series = series.astype('float32')
        series.dtypes
    
    
    
    #shape
    if True:
        #series shape (length)
        series.shape
        series.shape[0]
        len(series)
    
    
    
    #series statistics
    if True:
        series.mean()
        series.sum()
        series.std()
        series.var()
        series.median()
        series.max()
        series.min()
        series.quantile(0.5)
        series.quantile([0, 0.25, 0.5, 0.75, 1])
        series.skew()
        series.mode()
        series.cumsum()
        series.mad()
        series.rank() #max to min
        
        #index of the min and max values
        series.argmin()
        series.argmax()
    
    
    
    #other methods
    if True:
        series.unique()
        series.nunique() #number of unique values
        series.to_numpy()
        series.values
        series.tolist() #convert a series to list (only works with Series)
        np.array(series)
        series.rename('a', inplace=True) #renaming the series
        series.rename(lambda x: 2*x) #x is the existing index
        series.name = 'Jafar'
        series.name
        series.reset_index() #resetting the index (by default put the old index as a new column and return dataframe)
        series.reset_index(drop=False) #return dataframe
        series.reset_index(drop=True) #return series
    
    
    
    #map()
    if True:
        #Map values of Series according to an input mapping or function.
        s = pd.Series(['cat', 'dog', 'deer', 'rabbit','cat', 'dog', 'deer'])
        s.map({'cat': 'kitten', 'dog': 'puppy'}) #returns NaN for unidentified items
        s.map('I am a {}'.format)
        s2 = pd.Series([1,2,3,4], index=['cat', 'deer', 'dog', 'rabbit'])
        s3 = pd.Series([1,4], index=['cat', 'rabbit'])
        s.map(s2) #can pass another series with index equal to the series values
        s.map(s3) #returns NaN for unidentified items
    
    
    
    #replace()
    if True:
        s = pd.Series(['Diam. 12.3 in. (64cm)','12.3 in. (64cm)'])
        s.replace(to_replace='Diam. ', value='', regex=True)
        s.replace(to_replace=['Diam. ','in.', '\(', 'cm\)'], value=['','','',''], regex=True)
    
    
    
    #str.split()
    if True:
        s = pd.Series(['3 large red cats','4 huge blue buses','12 small green dogs'])
        s.str.split(pat=' ', expand=False, regex=None) 
        s.str.split(pat=' ', expand=True, regex=None) #return df
        s.str.split(pat=' ', n=1, expand=False, regex=None)
        s.str.split(pat=' ', n=1, expand=True, regex=None)
        s.str.split(pat=' ', n=2, expand=True, regex=None)
    
    
    
    #plotting
    if True:
        series.plot.hist()
        series.plot.box()
    
    
    
    #saving | loading
    if True:
        series.to_csv('out.csv', index=True)
        
        series_without_index = pd.read_csv('out.csv', index_col=False)
        series_with_index = pd.read_csv('out.csv', index_col=0)





###############################################################################
###############################################################################
###############################################################################
if True:
    #DataFrame()
    if True:
        # create a data frame: the shape would be (n_r, n_c)
        # the first outter list indicates the list of rows, the inner list indicates the columns
        df = pd.DataFrame()
        df = pd.DataFrame([5, 6])
        df = pd.DataFrame([[5, 6]])
        df = pd.DataFrame([[5, 6], [1, 3]], index=['r1', 'r2'], columns=['c1', 'c2'])
        df = pd.DataFrame({'c1': [1, 2], 'c2': [3, 4]}, index=['r1', 'r2']) #from a dictionary
        df = pd.DataFrame([[5, 6], [1.2, 3]]) #upcasting
        df = pd.DataFrame({'name':['a','b','c','d','e','f','g','h'], 
                           'age':['23','34','45','56','67','78','89','90'], 
                           'income':['$128,012.3','$198,023.4','$298,034.5','$398,045.6','$68,056.7','$208,067.8','$918,078.9','$98,090.1'], 
                           'signed_up_datetime':['2021-02-11 14:28:32','2021-04-13 11:18:38','2019-12-01 11:18:32','2020-05-11 14:28:32','2001-11-11 06:28:32','2020-05-11 23:28:32','2011-10-11 16:28:32','2008-12-11 11:28:32'],
                           'sex':['M','M','F','M','F','F','M','F'],
                           'maried':['True','True','False','True','False','False','False','True'],
                           'parent_age_diff':['1 days 06:05:01.00003','-15 days 06:05:01.00003','41 days 23:05:01.00003','10 days 06:05:01.00003','21 days 03:05:03.00003','11 days 06:25:01.20003','0 days 01:05:01.00003','10 days 16:05:01.00003'],
                           'quantity':['550','1025','856','1000','856','200','550','856']})
    
    
    
    #astype()
    if True:
        # dataframe data types AND upcasting data type
        df.dtypes
        
        #Note that "astype()" only works good with clean data
        df[['age','quantity']] = df[['age','quantity']].astype('int64') #multiple columns
        df['sex'] = df['sex'].astype('category')
        
        # get rid of $ and ,
        df['income'] = df['income'].replace(to_replace=['\$', '\,'], value=['',''], regex=True)
        df['income'] = df['income'].astype('float64')
        
        # df['maried'] = df['maried'].astype('bool')
        # astype method does not work properly in this case because it gives True  
        #when the value exist and False otherwise
        import numpy as np
        df["maried"] = np.where(df["maried"] == 'True', True, False)
        
        df['signed_up_datetime'] = df['signed_up_datetime'].astype('datetime64')
        #if the data were not clean as in this case, you can use "to_datetime" customizable method
        df['signed_up_datetime'] = pd.to_datetime(df['signed_up_datetime'], format='%Y-%m-%d %H:%M:%S')
        
        df['parent_age_diff'] = pd.to_timedelta(df['parent_age_diff'])
        
        #checking the updated df
        df.dtypes
    
    
    
    #apply()
    if True:
        def fun_gender(a):
            if a=='M':
                return 'Male'
            else:
                return 'Female'
            
        def fun_name(n):
            return n+'_'+n
        
        
        df = pd.DataFrame({'name':['a','b','c','d','e','f','g','h'], 
                           'sex':['M','M','F','M','F','F','M','F'],
                           'quantity':['550','1025','856','1000','856','200','550','856']})
        
        #applies a function to all the elements
        df['sex'].apply(fun_gender)
        df['name'].apply(fun_name)
    
    
    
    #copy()
    if True:
        # Creating a copy
        de_copy = df.copy()
    
    
    
    #head() | info() | tail() | dtypes | shape
    if True:
        #general information of a df
        df.head()
        df.info()
        df.tail()
        df.dtypes
        df.shape
    
    
    
    #describe()
    if True:
        # Count, mean, std, min, 25th percentile, 50th percentile (median), 75th percentile, max
        df.describe()
        df.describe(percentiles=[.5]) # return for specific percentiles
        df.describe(percentiles=[.1])
        df.describe(percentiles=[.2,.3,.4,.8])
    
    
    
    #index | columns
    if True:
        #getting the row labels of a dataframe
        df.index
        df.index[3:5]
        #getting the column labels of a dataframe
        df.columns
        df.columns[2:4]
    
    
    
    #Retrieval
    if True:
        df = pd.DataFrame({'c1': [1,2,3,2,5,3], 
                           'c2': [3,4,5,6,7,5], 
                           'c3': [5,6,7,8,9,7], 
                           'c4': [6,7,8,9,0,8]}, 
                          index=['r0','r1','r2','r3','r4','r5'])
        
        # using column labels
        df['c1'] # as Series
        df[['c1']] # as data frame
        df[['c2', 'c3']]
        
        # using direct indexing for rows only
        df[0:2]
        df['r2':'r3']
        df[['c2', 'c3']][0:2]
        df[[False, True, True, False, True, False]]
        
        # using loc (by employing row index and column NAMEs)
        df.loc['r2'] #as a series
        df.loc[['r2']] #as dataframe
        bool_list = [False, True, True, False, True, False]
        df.loc[bool_list]
        df.loc['r1', 'c2']
        df.loc[['r1', 'r3'], 'c2'] #as a series
        df.loc[['r1', 'r3'], ['c2']] #as dataframe
        df.loc[['r1', 'r3'], ['c1','c2']] #prefered way
        df.loc[:, 'c2']
        df.loc[bool_list, ['c1','c2']]
        
        # using iloc for rows and columns
        df.iloc[1] #as a series
        df.iloc[[1]] #as a data frame
        df.iloc[[0, 2]]
        df.iloc[0:3,1:3] #using slicing
        df.iloc[:,1:3]
        df.iloc[1:4,:]
        df.iloc[1:4,1] #as a series
        df.iloc[1:4,[1]] #as a data frame
        df.iloc[1:4,1:2] #as a data frame
        bool_list = [False, True, True, False, True, False]
        df.iloc[bool_list]
    
    
    
    #value_counts()
    if True:
        # frequency counts
        df = pd.DataFrame({'c1': [1,2,3,2,5,3,3], 
                           'c2': [3,4,5,4,7,5,5], 
                           'c3': [5,6,7,6,9,7,7], 
                           'c4': [6,7,8,7,0,8,8]}, 
                          index=['r0','r1','r2','r3','r4','r5','r6'])
        df.value_counts() #sorted in descending order (return multi index and their frequencies)
        df.value_counts().index[0] # the most frequent record
        df.value_counts().index[-1] # the least frequent record
        df.value_counts()[df.value_counts().index[0]] # the value count of the most frequent record
        df.value_counts().index[3] # the 4th most frequent record
        df.value_counts()[df.value_counts().index[3]] # the value count of the 4th most frequent record
        max(df.value_counts())
        min(df.value_counts())
        df.value_counts(normalize=True)
        df.value_counts(ascending=True) #default is False
        df.value_counts(normalize=True, ascending=True)
        df['c1'].value_counts(bins=3)
        df['c1'].value_counts().loc[lambda x : x%2==1]
        df['c1'].value_counts().index[0] # the most frequent value
    
    
    
    #corr()
    if True:
        #returning correlation coefficient matrix
        df = pd.DataFrame({'c1': [1,2,3,2,5,3], 
                           'c2': [3,4,5,6,7,5], 
                           'c3': [5,6,7,8,9,7], 
                           'c4': [6,7,8,9,0,8]}, 
                          index=['r0','r1','r2','r3','r4','r5'])
        
        df.corr('pearson') #default #linear correlation
        df.corr('kendall') #rank correlation coefficient (represent degree of concordance)
        df.corr('spearman') #good for nonlinear relation
    
    
    
    #round()
    if True:
        #round the values of a dataframe to a specific decimal
        df = pd.DataFrame({'c1': [1,2,3,2,5,3], 
                           'c2': [3,4,5,6,7,5], 
                           'c3': [5,6,7,8,9,7], 
                           'c4': [6,7,8,9,0,8]}, 
                          index=['r0','r1','r2','r3','r4','r5'])
        
        df.corr('pearson')
        df.corr('pearson').round(2)
    
    
    
    #rename() | reset_index()
    if True:
        #dataframe manipulation
        df.rename(columns={df.columns[0]: "c11", df.columns[1]: "c21"}, inplace=True)
        
        df.rename(columns={'c11': "c1", 'c21': "c2"}, inplace=False)
        # resetting the indices in the data frame
        df.reset_index(drop=False, inplace=False)
        df.reset_index(drop=True, inplace=False)
    
    
    
    #where()
    if True:
        #replace the values where the condition is NOT satisfied
        df = pd.DataFrame({'c1': [1,2,3,2,5,3,3], 
                           'c2': [3,4,5,4,7,5,5], 
                           'c3': [5,6,7,6,9,7,7], 
                           'c4': [6,7,8,7,0,6,8]}, 
                          index=['r0','r1','r2','r3','r4','r5','r6'])
        df.where(df['c1']>4, 0) # the first is the condition and the second argument is the value when the condtion is False
        df.where(df['c1']>4, False) # the first is the condition and the second argument is the value when the condtion is False
        df.where(df['c1']>4, np.nan) # the first is the condition and the second argument is the value when the condtion is False
        df.where((df['c1']==3) & (df['c4']==8), np.nan)#and
        df.where((df['c1']==3) | (df['c4']==7), np.nan)#or
        df.where((df['c1']==3) ^ (df['c4']==8), np.nan)#xor
    
    
    
    #equals()
    if True:
        #used to check if to object (data frame, series, cell, ...) are the same
        df = pd.DataFrame({'c1': [1,2,3,2,5,3,3], 
                           'c2': [3,4,5,4,7,5,5], 
                           'c3': [5,6,7,6,9,7,7], 
                           'c4': [6,7,8,7,0,6,8]}, 
                          index=['r0','r1','r2','r3','r4','r5','r6'])
        
        df1 = df.copy()
        df.equals(df1) #the two objects should have exactly the same size
    
    
    
    #to_numpy() | values | array()
    if True:
        #converting a df to numpy array
        df.to_numpy()
        df.values
        np.array(df)
    
    
    
    #combining or expanding dataframes
    if True:
        '''
        Concat: gives the flexibility to join based on the axis (all rows or all columns)
        
        Append: is the specific case (axis=0, join='outer') of concat (being deprecated use concat)
        
        Join: is based on the indexes (set by set_index), an specific column match, and how ['left','right','inner','couter']
        
        Merge: is based on any particular column each of the two dataframes, 
        this columns are variables on like 'left_on', 'right_on', 'on'''
        #append()        **deprecated **don't use (use concat() instead
        if True:
            # Appending a series to a dataframe
            df = pd.DataFrame({'c1': [1, 2], 'c2': [3, 4]}, index=['r1', 'r2'])
            ser = pd.Series([0, 0], name='r3')
            df.append(ser) #create two new columns since the names do not match
            df.append(ser, ignore_index=True) #ignore_index would reset to default numeric indicis
            ser.rename({0:'c1',1:'c2'}, inplace=True) #should change the index to column labels to append below the df
            df.append(ser, ignore_index=True)
            
            # Appending a dataframe to another dataframe
            df = pd.DataFrame({'c1': [1, 2], 'c2': [3, 4]}, index=['r1', 'r2'])
            df2 = pd.DataFrame([[0,0],[9,9]])
            df.append(df2)
            df.append(df2, ignore_index=True)
            df2.rename(columns={0: "c1", 1: "c2"}, inplace=True)#it only works when the columns have no names
            df.append(df2, ignore_index=True)
        
        
        #concat()
        if True:
            # Combining data frames by rows or columns
            df1 = pd.DataFrame({'c1':[1,2], 'c2':[3,4]}, index=['r1','r2'])
            df2 = pd.DataFrame({'c1':[5,6], 'c2':[7,8]}, index=['r1','r2'])
            df3 = pd.DataFrame({'c1':[5,6], 'c2':[7,8]})
            df4 = pd.DataFrame({'c3':[5,6], 'c4':[7,8]})
            pd.concat([df1, df2], axis=1)
            pd.concat([df2, df1, df3]) # default axis=0
            pd.concat([df1, df3], axis=1) #row names do not match
            pd.concat([df1, df4]) #row and column names do not match
            pd.concat([df1, df4], ignore_index=True)
        
        
        #join()
        if True:
            #joining two dataframes
            #by default join works on index and we need to modify the index using .set_index
            df1 = pd.DataFrame({'name':['john doe', 'al smith', 'sam black', 'john doe'], 
                                'pos':['1B', 'C', 'P', '2B'], 
                                'year':[2000, 2004, 2008, 2003]})
            df2 = pd.DataFrame({'name':['jack lee', 'john doe', 'al smith', 'jafar'], 
                                'year':[2012, 2000, 2004,2022], 
                                'rbi':[80, 100, 12, 1]})
            
            df1.join(df2, lsuffix='_left', rsuffix='_right') #join based on the row indicies
            df1.join(df2.set_index('name'), on='name', lsuffix='_left', rsuffix='_right', how='left') #use column "name" as the index
            df1.join(df2.set_index('name'), lsuffix='_left', rsuffix='_right', how='left') # on foce the specific columns to be the same eventhou the index is not the same
            df1.join(df2.set_index('name'), on='name', lsuffix='_left', rsuffix='_right', how='right')
            df1.join(df2.set_index('name'), on='name', lsuffix='_left', rsuffix='_right', how='inner')
            df1.join(df2.set_index('name'), on='name', lsuffix='_left', rsuffix='_right', how='outer')
        
        
        #merge()
        if True:
            # Merging two data frames (results in the intersection of two data frames)
            df1 = pd.DataFrame({'name':['john doe', 'al smith', 'sam black', 'john doe'], 
                                'pos':['1B', 'C', 'P', '2B'], 
                                'year':[2000, 2004, 2008, 2003]})
            df2 = pd.DataFrame({'name':['john doe', 'al smith', 'jack lee'], 
                                'year':[2000, 2004, 2012], 
                                'rbi':[80, 100, 12]})
            pd.merge(df1, df2)
    
    
    
    #shift()
    if True:
        #Shift the dataframe values by not shifting the index 
        #by desired number of periods with an optional time freq.
        df = pd.DataFrame({'name':['a','b','c','d','e','f','g','h'], 
                           'year':[2022,2022,1999,2008,2020,2022,2008,1999],
                           'age':[23,34,45,56,67,78,89,90], 
                           'income':[128012.3,198023.4,298034.5,398045.6,68056.7,208067.8,918078.9,98090.1], 
                           'sex':['M','M','F','M','F','F','M','F'],
                           'maried':[True,True,False,True,False,False,False,True],
                           'quantity':[550,1025,856,1000,856,200,550,856]}, 
                          index=pd.date_range("2020-01-01", "2020-01-08"))
        
        df.shift(periods=1)
        df.shift(periods=-1)
        df.shift(periods=1, fill_value=0)
        
        #the follwoing two changes the index without changing the original data
        df.shift(periods=3, freq="D") #only works with index defined as time series
        df.shift(periods=3, freq="infer") #only works with index defined as time series
    
    
    
    #drop()
    if True:
        # Dropping rows or columns
        df = pd.DataFrame({'c1': [1,2,3,2,5,3], 
                           'c2': [3,4,5,6,7,5], 
                           'c3': [5,6,7,8,9,7], 
                           'c4': [6,7,8,9,0,8]}, 
                          index=['r0','r1','r2','r3','r4','r5'])
        
        # option 1: use "labels" for both rows and columns in conjunction with "axis"
        df.drop(labels='r1') # default axis=0
        df.drop(labels='r1', axis=0)
        df.drop(labels=['r1','r4'], axis=0)
        df.drop(labels=['c1', 'c3'], axis=1)
        
        # option 2: use "index" for row and "columns" for columns
        df.drop(index='r2')
        df.drop(index=['r2','r1'])
        df.drop(columns='c2')
        df.drop(columns=['c2','c3'])
        df.drop(index='r2', columns='c2')
        df.drop(index=df.index[2:4])
        df.drop(columns=df.columns[1:3])
        #the most straight forward
        df.drop(index=df.index[2:4], columns=df.columns[1:3]) #BEST Approach
        
        # option 3: use "iloc"
        df.drop(df.iloc[:, 1:3], inplace=False, axis=1) #not prefered
        df.drop(df.iloc[1:3, :].index, inplace=False, axis=0) #not prefered
    
    
    
    #isnull()
    if True:
        #return False if it's not missing and True if the value is missing
        df = pd.DataFrame({'name':['a',np.nan,'b','c','b','b','c','c',np.nan,'a'], 
                           'sex':['M','M',np.nan,'F','M','F','F','M','F',np.nan],
                           'marrital':['S','M',np.nan,'M','S',np.nan,'S','S','M','S'],
                           'quantity':[550,1025,856,1000,856,200,550,856,np.nan,np.nan]})
        
        df.isnull()
    
    
    
    #duplicated() | drop_duplicates()
    if True:
        #return the duplicated rows
        df.duplicated()
        # dropping the duplicates
        df.drop_duplicates()
        df.drop_duplicates(subset=['c1']) #check the subset columns only for matching
    
    
    
    #dropna()
    if True:
        df = pd.DataFrame({'name':['a',np.nan,'b','c','b','b','c','c',np.nan,'a'], 
                           'sex':['M','M',np.nan,'F','M','F','F','M','F',np.nan],
                           'marrital':['S','M',np.nan,'M','S',np.nan,'S','S','M','S'],
                           'quantity':[550,1025,856,1000,856,200,550,856,np.nan,np.nan]})
        
        # dropping the data with missing values
        df = df.dropna(inplace=False) #default
        df.dropna(inplace=True)
    
    
    
    #replace()
    if True:
        #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.replace.html
        df = pd.DataFrame({'A': ['bat', 'foo', 'bait'],
                       'B': ['abc', 'bar', 'foz']})
        df.replace(to_replace='^ba.$', value='new', regex=True)
        df.replace(to_replace='^fo\w', value='fnew', regex=True)
        df.replace(to_replace=['^ba.$', '^fo\w'], value=['new', 'old'], regex=True)
    
    
    
    #statistics
    if True:
        df = pd.DataFrame({'c1': [1,2,3,2,5,3], 
                           'c2': [3,4,5,6,7,5], 
                           'c3': [5,6,7,8,9,7], 
                           'c4': [6,7,8,9,0,8]}, 
                          index=['r0','r1','r2','r3','r4','r5'])
        
        #statistics on rows or columns
        df.sum()#default axis is 0
        df.sum(axis=0)
        df.sum(axis=1)
        df.mean()#default axis is 0
        df.mean(axis=0)
        df.mean(axis=1)
        df.std()#default axis is 0
        df.std(axis=0)
        df.std(axis=1)
        df.var()
        df.median()
        df.max()
        df.max(axis=1)
        df.min()
        df.quantile(0.5)
        df.quantile([0, 0.25, 0.5, 0.75, 1])
        df.skew()
        df.mode()
        df.cumsum()
        df.mad()
        df.rank()
        df.rank(axis=0)
        df.rank(axis=1)
        df.nunique() #number of unique values
        df.nunique(axis=0)
        df.nunique(axis=1)
        
        df.sort_index(axis = 0) #it sorts based on the index only NOT the actual data
    
    
    
    #idxmin() | idxmax()
    if True:
        #index of the min and max values
        df.idxmin()
        df.idxmin(axis=0)
        df.idxmin(axis=1)
        df.idxmax()
        df.idxmax(axis=0)
        df.idxmax(axis=1)
        df[df['c1'].index==df.idxmax()[0]]
        df[df['c4'].index==df.idxmax()[3]]
    
    
    
    #rename()
    if True:
        #renaming the columns and rows
        df.rename(columns={'old_name1': "new_name1", 'old_name2': "new_name2"}, inplace=True)
        df.rename(columns={'c1':'a','c3':'c','c4':'d'})
        df.rename(index={'r1':'a','r3':'c'})
    
    
    
    #unique()
    if True:
        df['c1'].unique()
    
    
    
    #sort_values()
    if True:
        # Sorting data frame
        df = pd.DataFrame({'c1': [1,2,3,2,5,3], 
                           'c2': [3,4,5,6,7,5], 
                           'c3': [5,6,7,8,9,7], 
                           'c4': [6,7,8,9,0,8]}, 
                          index=['r0','r1','r2','r3','r4','r5'])
        
        df.sort_values('c1')
        df.sort_values('c1', ascending=False)
        df.sort_values(['c2', 'c1'])
        df.sort_values(['c4', 'c1'], ascending=[True, False])
        df.sort_values('c4', inplace=True)
    
    
    
    #multiply()
    if True:
        # Quantitative weighted featears
        df = pd.DataFrame({'T1': [0.1, 150.], 
                           'T2': [0.25, 240.], 
                           'T3': [0.16, 100.]})
        df.multiply(2)
        df.multiply([1,0.5,1]) #default axis is 1
        df.multiply([1000, 1], axis=0)
    
    
    
    #fillna()
    if True:
        #fill the missing values
        df = pd.DataFrame({'A':['Ali', 'Zli', 'Cli', 'Dli','Blio'], 
                           'B':[500,np.nan,300,400,400], 
                           'C':[11,12,np.nan,14,15]})
        
        df['B'].fillna(1200)
        df.fillna(1200)
        #fill the missing value with the most frequent of each column using apply
        df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))
    
    
    
    #get_dummies()
    if True:
        # Creating indicator features
        df = pd.DataFrame({'name':['a',np.nan,'b','c','b','b','c','c',np.nan,'a'], 
                           'sex':['M','M',np.nan,'F','M','F','F','M','F',np.nan],
                           'marrital':['S','M',np.nan,'M','S',np.nan,'S','S','M','S'],
                           'quantity':[550,1025,856,1000,856,200,550,856,np.nan,np.nan]})
        
        pd.get_dummies(df)
        #dummy_na: assign a NA column for the missed values
        pd.get_dummies(df, dummy_na=True)
        #drop_first: remove one column which end of k-1 columns for k class
        pd.get_dummies(df, prefix_sep='_', drop_first=True)
        #columns: create dummies for the specified columns only
        pd.get_dummies(df, prefix_sep='_of_', dummy_na=False, columns=['sex','marrital'], drop_first=False)
    
    
    
    #filtering
    if True:
        # Filtering data frame: return boolean
        import numpy as np
        df = pd.DataFrame({'A':['Ali', 'Bli', 'Cli', 'Dli','Blio'], 
                           'B':[500,200,300,400,400], 
                           'C':[11,12,np.nan,14,15],
                           'D':[True,True,False,True,False]})
        
        filter_ = df['A'] == 'Bli'
        filter_ = df['B'] < 400
        filter_ = df['A'] != 'Dli'
        filter_ = df['A'].str.startswith('B')
        filter_ = df['A'].str.endswith('o')
        filter_ = ~df['A'].str.contains('C') # ~ negates the filter condition
        filter_ = df['A'].isin(['Ali', 'Dli'])
        filter_ = df['B'].isin([200, 400]) # check to see if the values are in the given list
        filter_ = df['C'].isna()
        filter_ = df['C'].notna()
        #Filtering data frame
        filter_ = df[df['B'] < 400]
        filter_ = df[~(df['B'] < 400)] # ~ negates the filter condition
        filter_ = df[df['A'].str.startswith('B')]
        
        df[(df['D']==True) & (df['B']==400)]#and
        df[(df['D']==True) | (df['B']==400)]#or
        df[(df['D']==True) ^ (df['B']==400)]#xor
    
    
    
    #groupby()
    if True:
        # Grouping data
        #https://machinelearningknowledge.ai/pandas-tutorial-groupby-where-and-filter/
        df = pd.DataFrame({'name':['a','b','c','d','e','f','g','h'], 
                           'year':[2022,2022,1999,2008,2020,2022,2008,1999],
                           'age':[23,34,45,56,67,78,89,90], 
                           'income':[128012.3,198023.4,298034.5,398045.6,68056.7,208067.8,918078.9,98090.1], 
                           'sex':['M','M','F','M','F','F','M','F'],
                           'maried':[True,True,False,True,False,False,False,True],
                           'quantity':[550,1025,856,1000,856,200,550,856]})
        
        # Grouping data by single column
        groups = df.groupby('year')
        for name, group in groups: #name is the key for the groups
            print('Year:', name)
            print(group, end='\n\n')
        #end
        groups.get_group(2022)
        groups.sum()
        groups.aggregate(sum) #benefit: we can pass any function such as frozenset that it's not part of pandas
        groups.agg(sum)
        groups.name.count() #count the members in each group for the column "name"
        groups['name'].count()
        groups[['age']].sum() #as DataFrame
        groups['age'].sum() #as series
        groups['age'].mean()
        groups.mean()['age']
        for name, group in groups: #name is the key for the groups
            print('Year:', name)
            print(group['age'].mean(), end='\n\n')
        #end
        # Filtering the group
        groups.filter(lambda x: x.name < 2015) #name is the key for the groups
        df[df['year']<2015]
        
        # Grouping data by multiple columns
        groups2 = df.groupby(['year', 'sex'])
        for name, group in groups2:
            print('Year, sex:', name)
            print(group, end='\n\n')
        #end
        groups2.filter(lambda x: x.name[0] < 2015) #name is the columns that are passed to groupby
        groups2.filter(lambda x: x.name[1] == 'M')
        groups2.filter(lambda x: x.name[1] != 'M')
        groups2.filter(lambda x: (x.name[1] == 'M') & (x.name[0] < 2015))
        groups2.filter(lambda x: (x.name[1] == 'M') | (x.name[0] < 2015))
        groups2.filter(lambda x: (x.name[1] == 'M') ^ (x.name[0] < 2015))
        groups2.name.count()
    
    
    
    #melt()
    if True:
        #Dataframe reshaping: converting a wide dataframe to a long dataframe
        #It converts the df to three parts of 'identifiers', 'variables', and 'values'
        pd.melt(df, id_vars=None, #Column(s) to use as identifier variables
                value_vars=None, #Column(s) to unpivot. If not specified, uses all columns that are not set as id_vars
                var_name=None, #Name to use for the ‘variable’ column
                value_name='value') #Name to use for the ‘value’ column
        
        df = pd.DataFrame({'name':['a','b','c','d','e','f','g','h'], 
                           'year':[2022,2022,1999,2008,2020,2022,2008,1999],
                           'age':[23,34,45,56,67,78,89,90], 
                           'income':[128012.3,198023.4,298034.5,398045.6,68056.7,208067.8,918078.9,98090.1], 
                           'sex':['M','M','F','M','F','F','M','F'],
                           'maried':[True,True,False,True,False,False,False,True],
                           'quantity':[550,1025,856,1000,856,200,550,856]})
        
        pd.melt(df)
        pd.melt(df, id_vars=['name'])
        pd.melt(df, id_vars=['name','year','age'])
        pd.melt(df, id_vars=['name','year','age'], value_vars=['sex'])
    
    
    
    #pivot()
    if True:
        #Dataframe reshaping: converting a long dataframe to a wide dataframe (opposite of melt)
        df.pivot(index='index_column', #Column to use to make new frame’s index
                 columns='column_names') #Column to use to make new frame’s columns
        
        df = pd.DataFrame({'name':['a','b','c','d','e','f','g','h'], 
                           'year':[2022,2022,1999,2008,2020,2022,2008,1999],
                           'age':[23,34,45,56,67,78,89,90], 
                           'income':[128012.3,198023.4,298034.5,398045.6,68056.7,208067.8,918078.9,98090.1], 
                           'sex':['M','M','F','M','F','F','M','F'],
                           'maried':[True,True,False,True,False,False,False,True],
                           'quantity':[550,1025,856,1000,856,200,550,856]})
        
        df_long = pd.melt(df, id_vars=['name'])
        df_long.pivot(index='name', columns='variable')
    
    
    
    #explod()
    if True:
        #used to transform each element of a list-like to a row
        df = pd.DataFrame({'A': [[0, 1, 2], 'foo', [], [3, 4]],
                           'B': 1, 
                           'C': [['a', 'b', 'c'], np.nan, [], ['d', 'e']]})
        df.explode('A') #for each item in a list create a new row
        df.explode('C')
        df.explode(['A','C']) #columns must have matching element counts
    
    
    
    #aggregate() | agg()
    if True:
        #Aggregate using one or more operations over the specified axis.
        df.aggregate(func=None, axis=0)
        
        df = pd.DataFrame({'name':['a','b','c','d','e','f','g','h'], 
                           'year':[2022,2022,1999,2008,2020,2022,2008,1999],
                           'age':[23,34,45,56,67,78,89,90], 
                           'income':[128012.3,198023.4,298034.5,398045.6,68056.7,208067.8,918078.9,98090.1], 
                           'sex':['M','M','F','M','F','F','M','F'],
                           'maried':[True,True,False,True,False,False,False,True],
                           'quantity':[550,1025,856,1000,856,200,550,856]})
        
        df.aggregate(func=['sum', 'min','mean'])
        df['sex'].aggregate(func=['sum'])
        df.aggregate({'sex' : ['sum'], 'income' : ['min', 'max', 'mean'], 'age':['mean']})
        df.agg(row_index_1=('sex', 'sum'), row_index_2=('income', 'min'), row_index_3=('age', np.mean))
        df.aggregate(row_index_1_sum=('sex', 'sum'), row_index_2_min=('income', 'min'), row_index_3_mean=('age', np.mean))
        df.aggregate(frozenset, axis=0)
    
    
    
    #read_csv()
    if True:
        df = pd.read_csv('data.csv')
        df = pd.read_csv('data.csv', index_col=0)
        df = pd.read_csv('D:\Jafar.csv', index_col=0)
    
    
    
    #read_excel()
    if True:
        df = pd.read_excel('data.xlsx')
        df = pd.read_excel('data.xlsx', sheet_name=1)
        df = pd.read_excel('data.xlsx', sheet_name='MIL')
        df_dict = pd.read_excel('data.xlsx', sheet_name=[0, 1]) # Sheets 0 and 1
        df_dict = pd.read_excel('data.xlsx', sheet_name=None) # All Sheets
    
    
    
    #read_json
    if True:
        df1 = pd.read_json('data.json')
        df2 = pd.read_json('data.json', orient='index')
    
    
    
    #to_csv()
    if True:
        # Saving data frame as csv file
        df.to_csv('data.csv')
        df.to_csv('data.csv', index=False)
    
    
    
    #to_excel()
    if True:
        # Saving data frame as xlsx file
        with pd.ExcelWriter('data.xlsx') as writer:
            df.to_excel(writer, index=False, sheet_name='NYY')
            df.to_excel(writer, index=False, sheet_name='BOS')
    
    
    
    #to_json()
    if True:
        # Saving data frame as json file
        df.to_json('data.json')
        df.to_json('data.json', orient='index')



