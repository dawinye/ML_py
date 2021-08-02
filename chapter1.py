import numpy as np
from scipy import sparse

# Create a vector as a row
vector_row = np.array([1, 2, 3])
# Create a vector as a column
vector_column = np.array([[1],
[2],
[3]])

# Create a matrix
matrix = np.array([[1, 2],
 [1, 2],
 [1, 2]])

 # Create a matrix
matrix = np.array([[0, 0],
 [0, 1],
 [3, 0]])
# Create compressed sparse row (CSR) matrix
matrix_sparse = sparse.csr_matrix(matrix)

#print(matrix_sparse)
 #(1, 1) 1
 #(2, 0) 3

# Select everything up to and including the third element
#vector[:3]

# Select everything after the third element
#vector[3:]

# Select the first two rows and all columns of a matrix
#matrix[:2,:]

# Select all rows and the second column
#matrix[:,1:2]

# View number of rows and columns
#matrix.shape

# View number of elements (rows * columns)
#matrix.size

# View number of dimensions
#matrix.ndim

# Create matrix
matrix = np.array([[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]])

# Create function that adds 100 to something
add_100 = lambda i: i + 100
# Create vectorized function
vectorized_add_100 = np.vectorize(add_100)
# Apply function to all elements in matrix
vectorized_add_100(matrix)

# array([[101, 102, 103],
#  [104, 105, 106],
#  [107, 108, 109]])

# Return maximum element
np.max(matrix)

# Return minimum element
np.min(matrix)

# Find maximum element in each column
np.max(matrix, axis=0)
array([7, 8, 9])
# Find maximum element in each row
np.max(matrix, axis=1)

# Return mean
np.mean(matrix)

# Return variance
np.var(matrix)

# Return standard deviation
np.std(matrix)

# Find the mean value in each column
np.mean(matrix, axis=0)

# Create 4x3 matrix
matrix = np.array([[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9],
 [10, 11, 12]])
# Reshape matrix into 2x6 matrix
matrix.reshape(2, 6)

matrix.reshape(1, -1)

# Create matrix
matrix = np.array([[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]])
# Transpose matrix
matrix.T

# Tranpose row vector
np.array([[1, 2, 3, 4, 5, 6]]).T


# Create matrix
matrix = np.array([[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]])
# Flatten matrix
matrix.flatten()

# Create matrix
matrix = np.array([[1, 1, 1],
 [1, 1, 10],
 [1, 1, 15]])
# Return matrix rank
np.linalg.matrix_rank(matrix)

# Create matrix
matrix = np.array([[1, 2, 3],
 [2, 4, 6],
 [3, 8, 9]])
# Return determinant of matrix
np.linalg.det(matrix)

# Create matrix
matrix = np.array([[1, 2, 3],
 [2, 4, 6],
 [3, 8, 9]])
# Return diagonal elements
matrix.diagonal()

# Return diagonal one above the main diagonal
matrix.diagonal(offset=1)

# Return diagonal one below the main diagonal
matrix.diagonal(offset=-1)

# Create matrix
matrix = np.array([[1, 2, 3],
 [2, 4, 6],
 [3, 8, 9]])
# Return trace
matrix.trace()

# Create matrix
matrix = np.array([[1, -1, 3],
 [1, 1, 6],
 [3, 8, 9]])
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)


# Create two vectors
vector_a = np.array([1,2,3])
vector_b = np.array([4,5,6])
# Calculate dot product
np.dot(vector_a, vector_b)

# Calculate dot product
#vector_a @ vector_b

# Create matrix
matrix_a = np.array([[1, 1, 1],
 [1, 1, 1],
 [1, 1, 2]])
# Create matrix
matrix_b = np.array([[1, 3, 1],
 [1, 3, 1],
 [1, 3, 8]])
# Add two matrices
np.add(matrix_a, matrix_b)

# Subtract two matrices
np.subtract(matrix_a, matrix_b)

# Create matrix
matrix_a = np.array([[1, 1],
 [1, 2]])
# Create matrix
matrix_b = np.array([[1, 3],
 [1, 2]])
# Multiply two matrices
np.dot(matrix_a, matrix_b)


# Create matrix
matrix = np.array([[1, 4],
 [2, 5]])
# Calculate inverse of matrix
np.linalg.inv(matrix)

# Set seed
np.random.seed(0)
# Generate three random floats between 0.0 and 1.0
np.random.random(3)
# Set seed


# Generate three random integers between 0 and 10
np.random.randint(0, 11, 3)

# Draw three numbers from a normal distribution with mean 0.0
# and standard deviation of 1.0
np.random.normal(0.0, 1.0, 3)
#array([-1.42232584, 1.52006949, -0.29139398])
# Draw three numbers from a logistic distribution with mean 0.0 and scale of 1.0
np.random.logistic(0.0, 1.0, 3)
#array([-0.98118713, -0.08939902, 1.46416405])
# Draw three numbers greater than or equal to 1.0 and less than 2.0
np.random.uniform(1.0, 2.0, 3)
#array([ 1.47997717, 1.3927848 , 1.83607876])
