#convert a decimal number (base 10) to decimal (base 10)
format(15, "d")
#convert a decimal number (base 10) to Octal (base 8)
format(15, "o")
oct(15)[2:]
#convert a decimal number (base 10) to binary (base 2)
format(15, "b")
bin(15)[2:]
#convert a decimal number (base 10) to hexadecimal (base 16)
format(15, "x")
hex(15)[2:]
format(15, "X") # capitalized
# convert a string in base 2 to decimal (base 10)
int(format(15, "b"), base=2)


#comparing booleans (return boolean)
True & True
False & True
False & False
#comparing booleans (return 0 or 1)
True * True
False * True
False * False


######################## Bit Manipulation
# a number in a binary mode is presented by "0b..." for a positive number and 
#"-0b..." for a negative number
0b1000
-0b1000

# AND "&" operator (return 1 if both values are 1) ****Not the same as "and" operator****
#0b00110010
#&
#0b10011101
#=
#0b00010000  which is 16 in decimal
0b00110010 & 0b10011101


# OR "|" operator (return 1 if either values are 1) ****Not the same as "or" operator****
#0b00110010
#|
#0b10011101
#=
#0b10111111  which is 191 in decimal
0b00110010 | 0b10011101


# XOR "^" operator (return 1 if only one of the values are 1 but not both)
#0b00110010
#^
#0b10011101
#=
#0b10101111  which is 175 in decimal
0b00110010 ^ 0b10011101


#shift to the left "<<", it's equivalent to adding zeros to the begining, each 
#shift means multiply by 2 which makes sense
0b00110010 # which is 50
0b00110010 << 1 # which is 100
0b001100100
0b00110010 << 2 # which is 200
0b0011001000

#shift to the right ">>", it's equivalent to removing digits from the begining, 
#each shift means divide by 2 which makes sense
0b00110010 # which is 50
0b00110010 >> 1 # which is 25
0b0011001
0b00110010 >> 2 # which is 12
0b001100


#adding in binary is the same as addition in decimal
#when we hit the limit we carry over to the next digit
#0b00110010
# +
#0b10011101
# =
0b11001111 # 207
0b00110010 + 0b10011101 #which is equivalent to 50+157 = 207





#read from standard input
a = input()

#convert a string to int
a = int('11')

#apply the same funtion to all the elements in an iterable object (returns a map 
#object that needs to be converted)
a = ['1', '2', '3', '4']
b = map(int, a)
c = list(b)


#create a copy insteda of copying the pointer to memory location
a_copy = a.copy()


#all the variables are stored as a dictionary in locals()
#list of variables
variables = locals()
#add a new variable
variables['new_var'] = 'value of the new variable'



#all the variables defined in the main function are globals and can be accessed 
#inside functions. If a variable with the same name as a global variable is defined 
#inside a function, then we will have two variables (one local and one global)
#to define a global variable inside a function we need to do the following
def myfunc():
    global x
    x = 12

print(x)
myfunc()
print(x)
#we can have access to the global variables inside the functions. However, if we 
#need to modify them we need to do:
x = "awesome"
def myfunc():
    global x
    x = "fantastic"

print(x)
myfunc()
print("Python is " + x)



# Lambda is a single expression anonymous function often used as an inline function. 
#In simple words, it is a function that has only one line in its body. It proves 
#very handy in functional and GUI programming.
sum12 = lambda a, b, c: a + b + c
sum12(1, 2, 3)
jafar = lambda a, b, c: a*b+c
jafar(2,2,1)


#An assert statement consists of the assert keyword, the expression or condition to test, 
#and an optional message. If statement is true, it do nothing, if it's False it hault 
#and return the message if exists
number = 40
assert number > 0
assert number > 0, f"number greater than 0 expected, got: {number}"
assert number == 0, f"number equal to 0 expected, got: {number}"


#sorting a list
a = ['1', '2', '8', '3', '4']
a.sort() #sort in place
a.sort(reverse=False) #default
a.sort(reverse=True)
aa = [[1,'z'],[2,'z'],[3,'b'],[3,'x'],[3,'a']]
aa.sort(key=lambda x: x[1]) #sort by the second item in the sublist
aa.sort(key=lambda k: (k[0], k[1]), reverse=True) #sort by the first and second items in the sublist
aa.sort(key=lambda j: (-j[0], j[1])) #sort by the first and second items in the sublist (the "-" for the numerical sublist changes from ASC to DESC)
aa.sort(key=lambda m: (-m[0], m[1]), reverse=True) #sort by the first and second items in the sublist (the "-" for the numerical sublist changes from DESC to ASC)



#sorting a list or set
a = ['1', '2', '3', '4', '0']
sorted(a)
sorted(a, reverse=False)
sorted(a, reverse=True)
b = {21:'a', 1:'z', 2:'y', 3:'x'}
sorted(b)#returning the sorted keys
aa = [[1,'z'],[2,'z'],[3,'b'],[3,'x'],[3,'a']]
sorted(aa, key = lambda x: x[1])#sort by the second item in the sublist

#summation of over a set or list of numbers (it should be iterable)
sum([1,2])
sum(set([1,2,2]))

#rounding to n=3 decimal points
round(12.123456,3)


#creating a tuple of complex number: complex(real, imaginary)
z = complex(2, -3)
z = complex(1)
z = complex()
z = complex('5-9j')
#This tool returns the phase of complex number z (also known as the argument of z).
import cmath
cmath.phase(complex(z))


#calculate the quotient
quotient  = 100//3
#calculate the remainder
remainder = 100%3
#calculate both the quotient  and the remainder
quotient, remainder = divmod(100, 3)
#calculate a^b
pow(5,2)
#calculate a^b modulus (mod) m, which results in the remainder of (a^b)%m
m=3
pow(5, 2, m)



#check if all the elements of a given iterable (List, Dictionary, Tuple, set, etc) 
#are True else it returns False
#note, "0" counts as False and "1" counts as True and [] counts as True
l = [4, 5, 1]
print(all(l))
l = [0, 0, 0]
print(all(l))
l = [1, 1, 1]
print(all(l))
l = [1, 0, 6, 7, False]
print(all(l))
l = []
print(all(l))
l = [4, 5, -1]
all([x>0 for x in l])

#check if any (at least one) of the elements of a given iterable (List, Dictionary, 
#Tuple, set, etc) are True else it returns False.
#note, "0" counts as False and "1" counts as True and [] counts as False
l = [4, 5, 1]
print(any(l))
l = [0, 0, 0]
print(any(l))
l = [1, 1, 1]
print(any(l))
l = [1, 0, 6, 7, False]
print(any(l))
l = []
print(any(l))
l = [4, 5, -1]
any([x>0 for x in l])



#This function returns a list of tuples. The tuple contains the elements from each 
#of the argument sequences or iterables. If the argument sequences are of unequal 
#lengths, then the returned list is truncated to the length of the shortest argument sequence.
list(zip([1,2,3,4,5,6],'Hacker'))
list(zip([1,2,3,4,5,6],'Hacker', [1,2,3,4,5,6]))
X = [[1, 2, 3], [6, 5, 4], [7, 8, 9]]
list(zip(X))
list(zip(*X)) # * unzip the list.



#eval performs the operations passed as string argument
#it can also be used to work with Python keywords or defined functions and variables. 
#These would normally be stored as strings.
number = 9
eval('number * number + 1 - 0.5')
eval('9 + 5')
type(eval('len'))



#cumulatively compute a final value of the items in an iterable. reduce() performs the following steps:
# 1. Apply a function (or callable) to the first two items in an iterable and generate a partial result.
# 2. Use that partial result, together with the third item in the iterable, to generate another partial result.
# 3. Repeat the process until the iterable is exhausted and then return a single cumulative value.
from functools import reduce
def my_add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")
    return result

numbers = [0, 1, 2, 3, 4]
reduce(my_add, numbers)#reduce(function, iterable)




######################## Tuple
#A tuple is a collection of objects which ordered and immutable (it means you cannot 
#update or change the values) tuple is hashable
#creating tuples
A = ()
A = (50,) #a tuple with single value (comma must be there)
A1 = ('physics', 'chemistry', 1997, 2000)
A2 = "a", "b", "c", "d"
#convert to tuple
A = tuple('ABCDEFGGGH')
#retrieving the elements
A[0]
A[2:5]
#concatenate tuples
A1+A2
#Repetition
A*3
#membership
3 in (1, 2, 3)
#list comprehension
[x for x in A]
#count the element in the tuple
A.count('G')
#return the index
A.index('C')




######################## List
#convert to list
A = list('ABCDEFGGGH')
#remove an element from a list
A.remove('C')
#add one element to a list
A.append('Z')
# adding all the elements of an iterable (list, tuple, string etc.) to the end of the list
A.extend(list('JAFAR'))
#remove the last element and return the val
A.pop()
#list comprehension
L = [1,2,3]
M = [5,6,7]
[[x, y] for x in L for y in M] #nested loop
[[x, y] for (x,y) in zip(L,M)] #single joint loop
[x*2 for x in L]
a = [1,2,3,4,5]
[1 if i >= 2 else 0 for i in a]



######################## dictionary
#create empty dictionary
dic = {}
#Directly assigning key and values to a dictionary
dic = {1:100, 2:200, 3:300, 4:400}
#remove an element with a specific key
dic.pop(2)
#delete the last entered element from the dictionary
dic.popitem()
#retriving the value associated with a key
dic[4]
#add a key value to the dictionary
dic[5] = 500
#the list of keys
list(dic.keys())
#list of values
list(dic.values())
#extending the key value pairs
dic2 = {'a': 321, 'g':14567, 'z':876}
dic.update(dic2)
#merging two dictionary into a new one (if there are similar keys, one would be ignored)
dic1 = {1:100, 2:200, 3:300, 4:400}
dic2 = {'a': 321, 'g':14567, 'z':876, 1:100, 2:32}
dic_merged = {**dic1, **dic2}
#merging using union operator
dic1 | dic2





######################## set
#A set is an unordered collection of elements without duplicate entries.
#When printed, iterated or converted into a sequence, its elements will appear in an arbitrary order.

#create empty set
A = set()
#Directly assigning values to a set
A = {1, 2}
#Creating a set from a list
A = set(['a', 'b'])
#adding single element to a set
A.add('c')
A.add((11,34))
#adding iterable object
A.update([1,2,3,4])
A.update({1, 7, 8})
A.update({1, 6}, [5, 13])
#remove a single value from a set (do nothing if the value does not exists)
A.discard(1)
A.discard(10)
#remove a single value from a set (raise an exception if the value does not exists)
A.remove(13)
A.remove(13)
#randomly remove an element from a set
A.pop()
#common operations
a = {2, 4, 5, 9}
b = {2, 4, 11, 12}
c = {2,4}
a.union(b) # Values which exist in a or b (a.union(b) == b.union(a))
a | b #union operator (works the same as union())
a.intersection(b) # Values which exist in a and b (a.intersection(b) == b.intersection(a))
a & b #and operator (works the same as intersection())
a.difference(b) # Values which exist in a but not in b (a.difference(b) != b.difference(a))
a.symmetric_difference(b) #all the element in a or b minus the intersection (a.union(b) - a.intersection(b))
a ^ b #^ operator (works the same as symmetric_difference())
c.issubset(b) #check if c is a subset of b
a.isdisjoint({1,3}) #check if a is disjoint with {1,3} (has no element in common)
b.issuperset(c) #check if a is a superset of b (opposite of subset)







#Creating immutable (unchangable) set
l = ["Geeks", "for", "Geeks"]
frozenset(l)

Student = {"name": "Ankit", "age": 21, "sex": "Male",
           "college": "MNNIT Allahabad", "address": "Allahabad"}
frozenset(Student)




######################## Text
#convert to upper case
'ali'.upper()
#convert to lower case
'ALI'.lower()
#check if all the charachters are uppercase or lower case
'ali'.isupper()
'ali'.islower()
#capitalized the first character
'8ali'.capitalize()
'ali'.capitalize()
#swapping all lower case to upper and vise versa
"Hello My Name Is PETER".swapcase()
#joining characters using the seperator
''.join(['A','l','i'])
'_'.join(['A','l','i'])
#seperating the words of a text to a list
txt = "welcome to the jungle"
txt.split()
txt = "welcome_to_the_jungle"
txt.split('_')
#seperating the lines of a text to a list
text = '''Monday
Tuesday
Wednesday
Thursday
Friday
Saturday
Sunday'''
text.splitlines()
#seperating the characters of a text to a list of all the charachters
txt = "welcome to the jungle"
list(txt)
#seperating the characters of a text to a set of all the unique charachters
txt = "welcome to the jungle"
set(txt)
list(set(txt))
#degree sign symbol
u'\N{DEGREE SIGN}'
#return true if all the characters are numbers
'123'.isdigit()
'12.3'.isdigit()
'12s'.isdigit()
'hello'.isdigit()
#return true if all the characters are alphabet
'1q23'.isalpha()
'hello'.isalpha()
'123'.isalpha()
#return true if all the characters are alphabet or numbers
'123'.isalnum()
'12.3'.isalnum()
'12s'.isalnum()
'hello'.isalnum()
#searching for a word (only returns the index of the beginning of the first occurance 
#and return "-1" if not found)
txt = "Hello, welcomeeeee to my world. welcome to this world."
txt.find("welcome")
txt.find("welcome", 20, 50) #search between the given index
txt.find("world")
txt.find("wd")
#replace a specific character
txt.replace('wel', 'new')





######################## PRINT
#convert to string
str(12)
#print on the screen
print('sample','text', sep='', end='')
print('sample','text', sep=' ', end='')
#left aligned string of length width
print('HackerRank'.ljust(20,'-'))
#centered string of length width
print('HackerRank'.center(20,'-'))
#right aligned string of length width
print('HackerRank'.rjust(20,'-'))
#astric command to print the elements of an iterator instead of the entire object (each * removes one bracket)
print(*[1,2,3,4])
print(*[[1,2,3,4]])
print(*[*[1,2,3,4]])
print(*set([1,2,3,4]))
#formated print
print('%.2f' % 12)
print('%.4f' % -12.12499)
print('%.2f-%.3fi' %(12, 34))
val = 'Geeks'
print(f"{val}for{val} is a portal for {val}.")
name = 'Tushar'
age = 23
print(f"Hello, My name is {name} and I'm {age} years old.")
print('The market is {}.'.format('closed'))
print('The market is {} but {}.'.format('closed', 'stable'))
import datetime
today = datetime.datetime.today()
print(f"{today:%B %d, %Y}")

