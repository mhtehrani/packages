import pyodbc
#retrieve the list of available drivers
pyodbc.drivers()
#creating the connection to Microsoft SQL
conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                      'Server=MOHAMMAD;'
                      'Database=Indian;'
                      'Trusted_Connection=yes;')
#creating a cursor, which is used to manage the context of a fetch operation
cursor = conn.cursor()
#executing the SQL query
cursor.execute('SELECT * FROM Trips')
#printing the results
for i in cursor:
    print(i)
#end
#store all the fetched data
Data = cursor.fetchall()
#applying the changes (if we don't do this, nothing will be saved)
cursor.commit()


import pandas as pd
#reading the queried data to a pandas dataframe
df = pd.read_sql_query('SELECT * FROM Trips', conn)
print(df)

#closing the connection
cursor.close()