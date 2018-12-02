'''import MySQLdb

db = MySQLdb.connect(host="iz.database.windows.net",
user="izaks",
passwd="Fakenews!",
db="fake_news_DB")

cur = db.cursor()



cur.execute("SELECT domain FROM (SELECT top 2 domain,COUNT(domain) AS domainCount FROM ARTICLE GROUP BY domain ORDER BY domainCount DESC) ARTICLE_COUNT")
for row in cur.fetchall():
    print (row)

db.close()'''

'''import pymysql
import pymysql.cursors
# Connect to the database
connection = pymysql.connect(host='iz.database.windows.net',
                             user='izaks@iz.database.windows.net',
                             password='Fakenews!',
                             db='fake_news_DB',
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)
try:
    with connection.cursor() as cursor:
        # Create a new record
        # sql = "INSERT INTO `users` (`email`, `password`) VALUES (%s, %s)"
        sql = "INSERT INTO HISTORICAL_INFO (time, HIST_aID) VALUES (convert(datetime, '01-12-2018 16:00:00', 105) , 1)"
        cursor.execute(sql)

    # connection is not autocommit by default. So you must commit to save
    # your changes.
    connection.commit()
finally:
    connection.close()'''

'''import pymysql
# Open database connection
# db = pymysql.connect(host="iz.database.windows.net", port=1433, user="izaks", password="Fakenews!", db="fake_news_DB" )
db = pymysql.connect(host="iz.database.windows.net", port=3306, user="root", passwd="Fakenews!", db="fake_news_DB" )

# prepare a cursor object using cursor() method
cursor = db.cursor()
# execute SQL query using execute() method.
cursor.execute("SELECT VERSION()")
# Fetch a single row using fetchone() method.
data = cursor.fetchone()
print ("Database version : %s " % data)
# disconnect from server
db.close()'''



import pyodbc
server = 'iz.database.windows.net'
database = 'fake_news_DB'
username = 'izaks'
password = 'Fakenews!'
# driver= '{ODBC Driver 17 for SQL Server}'
driver = '{mySystemDriver}'
cnxn = pyodbc.connect('DRIVER='+driver+';SERVER='+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD='+ password)
cursor = cnxn.cursor()
cursor.execute("SELECT * FROM Article")
row = cursor.fetchone()
while row:
    print (str(row[0]) + " " + str(row[1]))
    row = cursor.fetchone()








