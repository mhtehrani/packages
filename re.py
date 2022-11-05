"""
re package
"""

########################regular expression
#https://regex101.com is a great website to check the pattern and help with almost all the references
import re
#check to see if an expression is regular expression (i.e., the syntax is correct)
re.compile('.*\+') #it results in the compiled expression because it's a regular expression
re.compile('.*+') #it raise an exception because it's not a regular expression



#splitting a string according to the specified pattern
re.split(r"-","+91-011-2711-1111")
re.split("-","+91-011-2711-1111")
re.split(r"[\,.]",'100,000,000.000')



#replacing a pattern by something else
#re.sub('pattern to search', 'replaced pattern', 'text corpus')
corpus = '''results in the re compiled expression because it's a regular expression'''
re.sub('compiled', 'nooooooooot', corpus)



#returns all the non-overlapping matches of patterns in a string as a list of 
#strings (it could be anywhere)
string = '''results in the re compiled expression because it's a regular expression'''
re.findall('re[a-z]*', string)
m = re.findall(r'(\w+)@(\w+)\.(\w+)','I am going to username@hackerrank.com and there is another email tehrani@hotmail.com')
m[0][0]



#find the first occurance (it could be anywhere)
string = '''results in the re compiled expression because it's a regular expression'''
re.search('re[a-z]*', string)

string = 'jafar 1234 porteghal'
m = re.search(r'\d+', string)
m.end() #getting the index of the end of the substring
m.start() #getting the index of the start of the substring
string[m.start() : m.end()]

string = 'I am going to username@hackerrank.com and there is another email m.h.tehrani@hotmail.com'
m = re.search(r'(\w+)@(\w+)\.(\w+)', string)
m.groups() #returns a tuple containing all the subgroups of the match
m.group(0) #entire match
m.group(1) #1st match
m.group(2) #2nd match
m.group(3) #3rd match



#find the first occurance (if only found in the beginning)
string = '''results in the re compiled expression because it's a regular expression'''
m = re.match('re[a-z]*', string)
string[m.start() : m.end()]

string = 'I am going to username@hackerrank.com'
m = re.match(r'(\w+)@(\w+)\.(\w+)', string) #it can't find at the beginning -> None

#returns a dictionary containing all the named subgroups of the match, keyed by 
#the subgroup name (e.g., ?P<user>)
m = re.match(r'(?P<user>\w+)@(?P<website>\w+)\.(?P<extension>\w+)','myname@hackerrank.com')
string[m.start() : m.end()]
m.groupdict()



#returns an iterator yielding MatchObject instances over all non-overlapping matches 
#for the pattern in the string. It can be put into a list and then find the spans
string = '''results in the re compiled expression because it's a regular expression'''
for n in re.finditer('re[a-z]*', string):
    print(n.span())
    print(n[0])

LIST = list(re.finditer('re[a-z]*', string))
LIST[0]
LIST[0].span()
LIST[0][0] #the value




