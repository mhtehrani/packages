"""
math package
"""
import math


#calculate tan inverse (returns in radian)
math.atan(1.18)

#calculate cos inverse (returns in radian)
math.acos(0.707)

#the constant pi
math.pi

#convert radian to degrees
angle = math.degrees(math.pi)

#number of combinations
#number of ways to choose r items from n items without repetition and without order: 
#(1,2) is the same as (2,1)
#n! / (r! * (n - r)!)
math.comb(10,2) #10 chose 2
math.comb(4,2)

#number of permutation
#number of ways to choose r items from n items without repetition and with order: 
#(1,2) is NOT the same as (2,1)
#n! / (n - r)!
math.perm(10,2)