
import pymysql

def ins(username,password):
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='',
        db='Users',
    )

    #username = input("Enter title of your task: ")
    #password= input("Add some description to it: ")
    #date = input("Enter the date for this task (YYYY-MM-DD): ")

    try:
        with connection.cursor() as cursor:
            sql = "INSERT INTO Details (`username`, `password`) VALUES (%s, %s)"
            try:
                cursor.execute(sql, (username, password))
                print("Task added successfully")
            except:
                print("Oops! Something wrong")

        connection.commit()
    finally:
        connection.close()
