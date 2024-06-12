import numpy as np
#mac.lib
# using numpy for passing data in list
a = np.array([1, 2, 3, 4, 5])
print(a)
print(type(a))
print(len(a))

# using numpy for passing data in tuple 
a = np.array((1,2,3,4))
print("The numpy using tuple is ",a)
print(type(a))

#Creating 0-d array
b = np.array(42)
print(b)
print(type(b))

#Creating 1-d array
one_d = np.array([1,2,3,4,5,6])
print(one_d)
print(type(one_d))
print(len(one_d))

#creating 2-d array 
two_d = np.array([[1,2,3],[4,5,6]])
print(two_d)
print(type(two_d))

#creating 3-d arrays
three_d = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(three_d)
print(type(three_d))

#checking the number of dimensions of the array
r = np.array(40)
q = np.array([1,2,3])
w = np.array([[1,2,3],[4,5,6]])
d = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(r.ndim)
print(q.ndim)
print(w.ndim)
print(d.ndim)

#Create an array with 5 dimensions and verify that it has 5 dimensions:
arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print("The number of dimensions are",arr.ndim)

#Access Array Elements
access = ([1,2,3,4,5])
print(access[1])


#Get third and fourth elements from the following array and add them.
ex = ([1,2,3,4])
print(ex[2])
print(ex[3])
add = ex[2] + ex[3]
print("Adding both the elements ",add)

f1 = ([[1,2,3],[4,5,6]])
