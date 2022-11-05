"""
numpy package
"""
# importing the package
import numpy as np

#represent a number in a different base
np.base_repr(120, base=7)

# Initializing a NumPy array
arr = np.array([-1, 2, 5], dtype=np.float32)
arr = np.array([[-1, 2, 5]], dtype=np.float32)

# Checking the size of an array
arr.shape
arr.shape[0]


# Print the representation of the array, which shows the array and data type
print(repr(arr))
print('Array b: {}'.format(repr(arr)))


# creating a copy without having reference
arr2 = arr.copy()


#converting an array to list
list_ = arr.tolist()


# Casting the property of an array
arr = arr.astype(np.int32)
arr = arr.astype(np.int64)
arr = arr.astype(np.float32)
arr = arr.astype(np.float64)

# Printing the data type of an array
print(arr.dtype)


# Appending a value or an array to another array (arr and arr2 should exactly have 
#the same number of dimensions in the dimension being concatenated)
arr = np.append(arr, arr2, axis=1)
arr = np.append(arr, arr2, axis=0)


# Using np.nan as placeholder
np.array([np.nan, 1, 2], dtype=np.float32)


# Infinity
np.array([np.inf, 3], dtype=np.float32)


# Creating ranged array (# of elements) or (start, end, step) excluding end (it's an iterator)
arr = np.arange(5, dtype=np.int32)
arr = np.arange(5.1)
arr = np.arange(-1, 4)
arr = np.arange(-1.5, 4, 2)


# To specify the number of elements in the returned array
arr = np.linspace(5, 11, num=4) # default includes end point
arr = np.linspace(5, 11, num=4, endpoint=False)
arr = np.linspace(5, 11, num=4, dtype=np.int32)


# Reshaping data
reshaped_arr = np.reshape(arr, (1, 4)) # convert into 2-D array
reshaped_arr = np.reshape(arr, (-1, 4)) # -1 make the reshape flexible to find the optimal size (works for one dimension only)
reshaped_arr = np.reshape(arr, (-1, 2, 2)) # convert into 3-D array
arr.reshape(1, 4) # reshaping

# Flattening an array reshapes it into a 1D array
flattened = arr.flatten()


# Transposing an array
transposed = np.transpose(arr)
transposed = np.transpose(reshaped_arr, axes=(1, 2, 0)) # it should be consistent with the number of dimentions in arr

#diagonalized an array or list
np.diag(np.array([1,2,3]))
np.diag([1,2,3])

# Creating zeros and ones arrays
arr_new = np.zeros(4)
arr_new = np.ones((2, 3))
arr_new = np.ones((2, 3), dtype=np.int32)


# Creating an array of 0's or 1's with the same shape as another array
arr_new = np.zeros_like(arr)
arr_new = np.ones_like(arr)
arr_new = np.ones_like(arr, dtype=np.int32)

#filling entire array with something (it works in place)
arr_new.fill(np.nan)




# math
arr_new = arr + 1	#Add 1 to each element values
arr_new = arr - 1.2	#Subtract each element values by 1.2
arr_new = arr * 2	#Double each element values
arr_new = arr / 2	#Halve each element values
arr_new = arr // 2	#quotient of each value by 2
arr_new = arr % 2	#remainder division each value
arr_new = arr**2	#Square each element values
arr_new = arr**0.5	#Square root each element values
arr_new = np.exp(arr)	#Raised e t othe power of each element
arr_new = np.exp2(arr)	#Raised 2 to power of each element
arr_new = np.power(2, arr)	#Raised 2 to power of each element
arr_new = np.log(arr2)	#Natural logarithm of each element
arr_new = np.log2(arr2)	#Base 2 logarithm of each element
arr_new = np.log10(arr2)	#Base 10 logarithm of each element
arr_new = np.power(3, arr)	#Raise 3 to power of each number in arr
arr_new = np.power(arr, 3)	#Raise each number in arr to power 3
arr_new = np.power(arr2, arr)	#Raise arr2 to power of each number in arr

arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[5,6],[7,8]])
np.multiply(arr1, arr2) #perform Hadamard (elementwise) product
arr1 * arr2 #perform Hadamard (elementwise) product

np.matmul(arr1, arr2) #Perform matrix multiplication
np.dot(arr1, arr2) #Perform matrix multiplication
arr1 @ arr2 #Perform matrix multiplication



np.ceil(15.1)
np.floor(15.1)

# Generate pseudo-random integers
random_arr = np.random.randint(5) # default is from 0 inclusive
random_arr = np.random.randint(5, high=7) #excluding high
random_arr = np.random.randint(-3, high=14, size=(3, 3))


# Controling the outputs of the pseudo-random functions
np.random.seed(0)
#or we can create an object and call each distribution as a method
random_state = np.random.RandomState(0)

# Shuffling an array (rows for matrix)
vec = np.arange(1,10)
np.random.shuffle(vec) #works in place
matrix = np.random.randint(-3, high=14, size=(5, 5))
np.random.shuffle(matrix) #works in place (shuffle the rows)


# Drawing samples from uniform probability distributions
arr_new = np.random.uniform() #default is between 0 and 1
arr_new = np.random.uniform(low=-1.5, high=2.2)
arr_new = np.random.uniform(size=3)
arr_new = np.random.uniform(low=-3.4, high=5.9, size=(2, 2))
#or
random_state.uniform()

# Drawing samples from normal (Gaussian) probability distributions
arr_new = np.random.normal()
arr_new = np.random.normal(loc=1.5, scale=3.5)
arr_new = np.random.normal(loc=-2.4, scale=4.0, size=(2, 2))


# Drawing samples from a custom distributions
colors = ['red', 'blue', 'green'] # distribution
arr_new = np.random.choice(colors)
arr_new = np.random.choice(colors, size=2)
arr_new = np.random.choice(colors, size=(2, 2), p=[0.8, 0.19, 0.01]) #weighted each element in color by p


# One dimensional Slicing
arr = np.random.randint(-3, high=14, size=(5, 6))
arr_new = arr[:]	#Returning all elements
arr_new = arr[1:]	#Returning all elements starting at the 2nd element
arr_new = arr[2:4]	#Returning all elements starting at the 3rd element and ending at 4th element
arr_new = arr[:-2]	#Returning all elements except the last two element
arr_new = arr[-2:]	#Returning all elements starting from the 2nd from the end element


# Multidimentional slicing
arr_new = arr[:]
arr_new = arr[1:]
arr_new = arr[:, -1] # last column
arr_new = arr[:, -1:]
arr_new = arr[:, 1:]
arr_new = arr[0:1, 1:]
arr_new = arr[0, 1:]


# Returning the index of the max or min element in the flattened version
np.argmin(arr)
np.argmax(arr)


# Returning the index of the max or min element by looping through the element of identified axis
np.argmin(arr, axis=0) #Returning the index of the min element for each column
np.argmin(arr, axis=1) #Returning the index of the min element for each row
np.argmax(arr, axis=0) #Returning the index of the max element for each column
np.argmax(arr, axis=1) #Returning the index of the max element for each row


# Returning a boolean array that meat the filtering operations
arr == 3
arr > 0
arr != 1
~(arr != 1)
arr == 1
np.isnan(arr) # Returning locations of np.nan


arr[arr == 3]
arr[arr > 0]
arr[arr < 6]
arr[(arr > 0) & (arr < 6)] #and
arr[(arr > 0) | (arr < 6)] #or
arr[(arr > 0) ^ (arr < 6)] #xor



# Returning the indices of the True values or where the condition is met in the form of a tuple
x_ind, y_ind = np.where(arr != 0)
x_ind, y_ind = np.where(arr > 0)


# Returning a new array where the true and false values are replaced with something
np.where(np_filter, value_if_true, value_if_false)
np.where(arr > 0, 1000, -10)


# Filtering
arr_new = np.any(arr > 0)
arr_new = np.all(arr > 0)


# Axis wise filtering by looping through the element of identified axis
arr_new = np.any(arr > 0, axis=0) #for each column going over rows to check
arr_new = np.any(arr > 0, axis=1) #for each row going over columns to check
arr_new = np.all(arr > 0, axis=1) #for each row going over columns to check


# Returning the min and max
arr.min()
arr.max()
arr.min(axis=0) #for each column going over rows to check
arr.min(axis=1) #for each row going over columns to check
arr.max(axis=-1) # -1 means the last dimension
arr.max(axis=1) #for each row going over columns to check


# Returning the mean, variance, median, and summation
np.mean(arr)#overall
np.mean(arr, axis=0) #for each column going over rows to check
np.var(arr)
np.median(arr)
np.median(arr, axis=-1)
np.sum(arr)
np.sum(arr, axis=0)
np.sum(arr, axis=1)


# Returning cumulative summation
np.cumsum(arr)
np.cumsum(arr, axis=0) #for each column going over rows to check
np.cumsum(arr, axis=1)


# Concatenating list of arrays (the dimension being concatenated should match)
arr1 = np.random.randint(-3, high=14, size=(3, 4))
arr2 = np.random.randint(-3, high=14, size=(3, 4))
np.concatenate([arr1, arr2]) # default axis is 0
np.concatenate([arr1, arr2], axis=0)#adding rows
np.concatenate([arr1, arr2], axis=1)#adding columns
np.concatenate([arr2, arr1], axis=1)
#stacking two columns since xx is a one dimensional array it can't be concatenated
xx = np.random.randint(-3, high=14, size=3)
np.concatenate([arr1, np.column_stack([xx, xx**2])], axis=1)

# Saving numpy array
np.save('arr.npy', arr)
np.save('arr', arr)
np.save('D:\jafar',arr)


# Loading numpy array
load_arr = np.load('arr.npy')
load_arr = np.load('D:\jafar.npy')


#####linear algebra
#computes the determinant of an array
np.linalg.det([[1 , 2], [2, 1]])
#inverse of a matrix
A = np.array([[1 , 2], [3, 1]])
Ai = np.linalg.inv(A)
#computes the eigenvalues and right eigenvectors of a square array
vals, vecs = np.linalg.eig(A)
A_original = np.matmul(np.matmul(vecs, np.diag(vals)), np.linalg.inv(vecs))

#computes the singular values and left and right singular vectors of a rectangular array X = USV^T
X = np.array([[12, 4, 7], [10, 7, -3], [8, 2, 1], [16, 9, 2]])
U, s, V_T = np.linalg.svd(X)
S = np.zeros_like(X, dtype=np.float64)
S[:np.min(S.shape),:np.min(S.shape)] = np.diag(s)
X_original = np.matmul(np.matmul(U, S), V_T)




