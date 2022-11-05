"""
collections package
"""

#Counter is a container that stores elements as dictionary keys, and their counts 
#are stored as dictionary values.
from collections import Counter
myList = [1,1,2,3,4,5,3,2,3,4,2,1,2,3]
fre = Counter(myList)
print(fre)
fre[1]
fre.keys()
fre.values()



#create an ordered dictionary that remember the order of items added
from collections import OrderedDict
myList = [11,1,2,13,4,15,3,12,3,4,21,19,29,39]
dic = OrderedDict()
for n in myList:
    dic[n] = 1

list(dic.keys())[5]
dic.values()



#A deque is a double-ended queue. It can be used to add or remove elements from both ends
from collections import deque
d = deque() # creating an empty deque object
d.append(1) #adding to the right
d.appendleft(2) #adding to the left
d.clear() #clearing all the values
d.extend('123') #adding an iterable object to the right
d.extend('7896')
d.extendleft('234') #adding an iterable object to the left (note the final results is 4,3,2,...)
d.count('1') #counting the number of "1"s
d.pop() #removing the right most element
d.popleft() #removing the left most element
d.remove('2')
d.reverse() #reversing the order of deque
d.rotate(2) #Rotate the deque n steps to the right. If n is negative, rotate to the left


