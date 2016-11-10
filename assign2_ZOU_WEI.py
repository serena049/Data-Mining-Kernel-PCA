#### Using Python 2.7
#### HW2
#### By Wei Zou



import numpy as np 
import matplotlib.pyplot as plt
import itertools as it
from collections import Counter

##############################################################
##### Read in the data 
data = np.genfromtxt('Assign2.txt',delimiter=',') 
n = np.shape(data)[0] # number of rows
d = np.shape(data)[1] # number of columns
# print n
# print d


'''
##### Linear Kernal
# Step 1: Calculate the Kernal Matrix 
K1 = np.dot(data, np.transpose(data))


# Step 2: Center the Kernal Matrix
I = np.identity(n)
ONE = np.ones((n,n))
K11 = np.dot((I-ONE/n),K1)
K1 = np.dot(K11,(I-ONE/n))


# Step 3 & 4: Compute eigenvalues and eigenvectors
w1, v1= np.linalg.eigh(K1)
v1 = np.transpose(v1)
w1[:]=w1[::-1] # sort desc
#print "eigenvalues for linear PCA are ", w1
#w1 = w1[0:n]
v1[:]=v1[::-1] # sort desc
#v1 = v1[0:n,:]


# Step 5: Compute variance for each component
lamb1 = w1/n

# Step 6: Ensure that transpose(u)*u = 1
c1 = np.zeros((n,n))

for i in xrange(n):
	c1[i] = v1[i]/np.sqrt(w1[i])


# Step 7: Fraction of total variance
F1 = np.zeros(n)
for i in xrange(n):
	F1[i] = sum(lamb1[0:i+1])/np.sum(lamb1)

# Step 8: choose dimensionality
alpha = 0.95
for i in xrange(n):
	if (F1[i]>=alpha):
		break

r = i+1
print "r is ", r

# Step 9: Reduced basis
C1 = c1[0:r]

# Step 10: Reduced dimensionality data (Project the data onto the first two kernel PCs)

A1 = np.dot(C1[0],K1)
A2 = np.dot(C1[1],K1)

print "data projection on C11 is ", A1
print "data projection on C12 is ", A2

# Scatter Plot of the projected points
plt.scatter(A1,A2)
plt.title('Part I, Projections from Linear Kernel PCA')
plt.xlabel('Project_C11')
plt.ylabel('Project_C12')
plt.show()


##### Regular PCA 
# Mean Vector
mu = np.sum(data,0)/n

# Total Variance Vector
Z = data-mu   # calculate the centralized data 
sigma1 = np.cov(np.transpose(Z)) # Covaraince matrix 

#print np.shape(sigma1)

# Calcuate the eigenvalues and vectors 
u,v = np.linalg.eigh(sigma1)
v = np.transpose(v)
#print "eigenvalues for regular PCA are ", u
u1 = v[d-1]
u2 = v[d-2]


# Project original data onto the new coordinates

data_u1 = np.dot(data,u1)
data_u2 = np.dot(data,u2)


print "data projection on u1 is ", data_u1
print "data projection on u2 is ", data_u2

# Scatter Plot of the projected points
plt.scatter(data_u1,data_u2)
plt.title('Part I, Projections from Regular PCA')
plt.xlabel('Project_C1')
plt.ylabel('Project_C2')
plt.show()


##### Comparison btw Kernal PCA and regular PCA 

print "Linear Kernal PCA and Regular PCA are equivalent, however, the plots are symmetric to each other,
since we didn't do centralization for data when we apply the regular PCA projection"

'''
##### Gaussian Kernal	
sigma2 = input('Enter the variance of the guassian kernal: ')

# Step 1: Calculate the Kernal Matrix 
K2 = np.zeros((n,n))
# print np.shape(K2)
for i in xrange(n):
	for j in xrange(n):
		K2[i,j] = np.exp(-(np.linalg.norm(data[i]-data[j])**2/(2*sigma2)))


#K2= np.exp(-np.dot(data,np.transpose(data))/(2*sigma2))
# print np.shape(K2)

# Step 2: Center the Kernal Matrix
I = np.identity(n)
ONE = np.ones((n,n))
K2 = np.dot((I-ONE/n),K2)
K2 = np.dot(K2,(I-ONE/n))
# print np.shape(K1)

# Step 3 & 4: Compute eigenvalues and eigenvectors
w2, v2= np.linalg.eigh(K2)
v2 = np.transpose(v2)
print v2
w2[:]=w2[::-1] # sort desc
v2[:]=v2[::-1] # sort desc


# Step 5: Compute variance for each component
lamb2 = w2/n

# Step 6: Ensure that transpose(u)*u = 1
c2 = np.zeros((n,n))

for i in xrange(n):
	c2[i] = v2[i]/np.sqrt(w2[i])

# Step 7: Fraction of total variance
F2 = np.zeros(n)
for i in xrange(n):
	F2[i] = sum(lamb2[0:i+1])/np.sum(lamb2)

# Step 8: choose dimensionality
alpha = 0.95
for i in xrange(n):
	if (F2[i]>=alpha):
		break

r2 = i+1
print "r is ", r2

# Step 9: Reduced basis
C2 = c2[0:r2]


# Step 10: Reduced dimensionality data (Project the data onto the first two kernel PCs)
A21 = np.dot(C2[0],K2)
A22 = np.dot(C2[1],K2)

print "data projection on C21 is ", A21
print "data projection on C22 is ", A22

# Scatter Plot of the projected points
plt.scatter(A21,A22)
plt.title('Part I, Gauss Kernal Projections')
plt.xlabel('Project_C21')
plt.ylabel('Project_C22')
plt.show()

print "For me, I think larger variance makes more sense as we have projected points with larger variances explained by the first two PCA axis shown on the plots"

'''
##################################################################### Part 2

# generate the corners of the hypercube
D = input('Enter the dimension of the hypercube: ') # try 10, 100, 1000


def compute_angle(point1,point2):
	#since the unit length must be sqrt(D)
	return np.dot(point1,point2)/(np.linalg.norm(point1)*np.linalg.norm(point2))

N = 100000 # randomly select 10000 pairs

# generate the pairs and calcuate the angle btw each pair
results = np.zeros(N)
i=0
while(i < N):
	points_pre = np.random.rand(2,D) # generate 2*D array
	points_pre[points_pre<=0.5] = -1
	points_pre[points_pre>0.5] = 1
	results[i] = compute_angle(points_pre[0],points_pre[1])
	i = i+1



print 'min is '
print min(results)
print 'max is '
print max(results)
print 'range is '
print max(results)-min(results)
print 'mean is '
print np.mean(results)
print 'variance is '
print np.var(results)


plt.hist(results, bins=50, normed=True)
plt.show()



'''

