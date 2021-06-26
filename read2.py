import pymysql

def read3(Username,Password):
    con =  pymysql.connect(
        host='localhost',
        user='root',
        password='',
        db='Users',
    )

    try:
        with con.cursor() as cur:

            cur.execute('SELECT * FROM Details')

            rows = cur.fetchall()

            for row in rows:
                if Username==row[0] and Password==row[1]:
                    return 1
            return 0

    finally:

        con.close()
