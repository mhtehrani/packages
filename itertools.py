"""
itertools package
"""

#defining chain iterable
from itertools import chain
for i in chain(range(6), range(10, 16)):
    print(i)



#group similar connected elements together
#This tool creates a complex iterator over an iteratable object and returns a key 
#and the associated group object, which is a collection of similar items 
from itertools import groupby
S = [1,2,2,2,3,1,1]
for k, g in groupby(S):
    print(k)
    print(list(g))
    print()

for k, g in groupby('AAAABBBCCDAABBB'):
    print(k)
    print(list(g))
    print()



#Combinations are the way of selecting the objects or numbers from a group of objects 
# or collection, in such a way that the order of the objects does not matter.
#This tool returns the length subsequences of elements from the input iterable 
#(Only one of (1,2) and (2,1) would be among the results)
#n choose r
#order does NOT matter
from itertools import combinations
list(combinations(['1','2','3'],2))
#[('1','2'), ('1','3'), ('2','3')]
list(combinations('12345',2))
#[('1','2'), ('1','3'), ('1','4'), ('1','5'), ('2','3'), ('2','4'), ('2','5'), 
#('3','4'), ('3','5'), ('4','5')]



#This tool returns length subsequences of elements from the input iterable allowing 
#individual elements to be repeated more than once.
#order does NOT matter
from itertools import combinations_with_replacement
list(combinations_with_replacement(['1','2','3'],2))
#[('1', '1'), ('1', '2'), ('1', '3'), ('2', '2'), ('2', '3'), ('3', '3')]
list(combinations_with_replacement('12345',2))
#[('1','1'), ('1','2'), ('1','3'), ('1','4'), ('1','5'), ('2','2'), ('2','3'), ('2','4'), 
#('2','5'), ('3','3'), ('3','4'), ('3','5'), ('4','4'), ('4','5'), ('5','5')]



#A permutation is an act of arranging the objects or numbers "in order". 
#This tool returns successive length permutations of elements in an iterable 
#(both (1,2) and (2,1) would be among the results)
#order does matter
from itertools import permutations
list(permutations(['1','2','3']))
#[('1','2','3'), ('1','3','2'), ('2','1','3'), ('2','3','1'), ('3','1','2'), ('3','2','1')]
list(permutations(['1','2','3'],2))
#[('1', '2'), ('1', '3'), ('2', '1'), ('2', '3'), ('3', '1'), ('3', '2')]
list(permutations('12345', 2))
#[('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]



#Cartesian product(set of all possible ordered pairs)
#similar to "permutations" with replacement
from itertools import product
list(product(['1','2','3'], repeat=2))
#[('1','1'), ('1','2'), ('1','3'), ('2','1'), ('2','2'), ('2','3'), ('3','1'), ('3','2'), ('3','3')]
list(product([1,2,3], repeat=1))
list(product([1,2,3], repeat=2))
list(product([1,2,3], repeat=3))
list(product([1,1,3], repeat=3))


