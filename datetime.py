"""
datetime package
"""

# importing the package
import datetime

#date()
#create a date object (year, month, day)
date1 = datetime.date(2022,3,27)



#time()
#create a time object (hour, minute, second, microsecond)
time1 = datetime.time(13,27,45,4600)



#datetime()
#create a datetime object (year, month, day, hour, minute, second, microsecond)
datetime1 = datetime.datetime(2032,3,27,13,56,45,46000)



#strftime()
#print a string formatted
#https://docs.python.org/3.4/library/datetime.html#strftime-strptime-behavior
datetime1.strftime("%Y") #Y: YYYY (4D year)
datetime1.strftime("%y") #y: YY (2D year)
datetime1.strftime("%m") #m: MM (2D month)
datetime1.strftime("%M") #M: MM (2D minute)
datetime1.strftime("%w") #w: Week day (1D minute, 0 is Sunday and 6 is Saturday)
datetime1.strftime("%d") #d: DD (2D day)
datetime1.strftime("%B") #B: Full Month Name
datetime1.strftime("%b") #b: Short Month Name
datetime1.strftime("%I") #I: HH (12h format)
datetime1.strftime("%p") #p: AM or PM
datetime1.strftime("%H") #H: HH (24h format)
datetime1.strftime("%S") #S: SS (2D second)
datetime1.strftime("%f") #f: SS (6D microsecond)
datetime1.strftime("%A") #A: Full day Name
datetime1.strftime("%a") #a: Short day Name
datetime1.strftime("%z") #z: time zone (UTC offset +HHMM or -HHMM)
datetime1.strftime("%Z") #Z: time zone name

#examples
datetime1.strftime("%B %d, %Y")
datetime1.strftime("%h %d, %Y")
datetime1.strftime("%B %d, %y")
datetime1.strftime("%b %d, %Y")
datetime1.strftime("%Y/%m/%d")
datetime1.strftime("%d %b %y")
datetime1.strftime("%Y-%m-%d %H:%M:%S")
datetime1.strftime("%Y-%m-%d %H:%M:%S")
datetime1.strftime("%I:%M %p")



#strptime()
#convert an string to datetime object
datetime2 = datetime.datetime.strptime('03/27/2022', "%m/%d/%Y")
datetime2 = datetime.datetime.strptime("December 25, 2010", "%B %d, %Y")
datetime2 = datetime.datetime.strptime("2022-08-23 16:01:23", "%Y-%m-%d %H:%M:%S")
datetime2 = datetime.datetime.strptime("2022-08-23 04:01:23PM", "%Y-%m-%d %I:%M:%S%p")



#retrieving a specific part of the datetime object
datetime1.year #Between MINYEAR and MAXYEAR inclusive
datetime1.month #Between 0 and 12
datetime1.day #Between 1 and the number of days in the given month of the given year
datetime1.hour #In range(24)
datetime1.minute #In range(60)
datetime1.second #In range(60)
datetime1.microsecond #In range(1000000)
datetime1.tzinfo #time zone (None if it's naive)



#earliest and latest possible datetimes
earliest_date = datetime.datetime.min
latest_date = datetime.datetime.max



#now() | today()
#current time
today_datetime = datetime.datetime.now()
today_date = datetime.date.today()



#calculate time difference
dt1 = datetime.datetime(2022,3,27,13,27,45,46000)
dt2 = datetime.datetime(2023,6,30,14,28)
tdelta = dt2 - dt1 
print(tdelta) # --> 460 days, 1:00:14.954000
#get the days component
tdelta.days
#get the seconds component
tdelta.seconds
#get the microseconds component
tdelta.microseconds
#get the total time difference in seconds
tdelta.total_seconds()
tdelta.days*24*60*60 + tdelta.seconds + tdelta.microseconds/10**6


#timedelta()
#adding or subtracting a timedelta to a datetime object
datetime.date(2022,1,1) + datetime.timedelta(days=10, seconds=10, microseconds=10, milliseconds=10, minutes=10, hours=10, weeks=10) 
datetime.date(2022,1,1) + datetime.timedelta(days=-10, seconds=10, microseconds=10, milliseconds=10, minutes=10, hours=10, weeks=10) 
datetime.datetime(2023,6,30,14,28) + datetime.timedelta(days=-10, seconds=10, microseconds=10, milliseconds=10, minutes=10, hours=10, weeks=10) 



#time comparisons
t1 = datetime.date(2022,1,1)
t2 = datetime.date(2022,1,2)
t3 = datetime.date(2022,1,3)
t1<t2
t2<t3
t1<t2 and t2<t3
t1<t2 and t2>t3
t1<t2 or t2>t3
L1 = [datetime.date(2022,1,1), datetime.date(2022,1,2), datetime.date(2022,1,3), datetime.date(2022,1,4)]
L2 = [datetime.date(2022,1,3), datetime.date(2022,1,1), datetime.date(2022,1,2), datetime.date(2022,1,3)]
[(l1 > l2) for (l1,l2) in zip(L1,L2)]







