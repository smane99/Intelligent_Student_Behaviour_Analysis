import xlwt
from xlwt import Workbook
import datetime;
def record1(maxi,c,c1,c2):
    wb=Workbook()
    sheet1 = wb.add_sheet('Sheet 1')
    sheet1.write(0, 0, 'Name')
    sheet1.write(0, 1, 'Activeness')
    sheet1.write(0, 2, 'Active Count')
    sheet1.write(0, 3, 'Inactive Count')

    t=0
    t1=1
#iterate the each key-value pair of dictionary & insert into sheet

    #row = row + 1
    #t1=t1+1
    k=sum(c1)
    k1=sum(c2)
    min=len(c1)
    per=k/(k+k1)*100
    for i in range(len(maxi)):
        sheet1.write(t1,0, maxi[i])
        sheet1.write(t1,1, c[i])
        sheet1.write(t1,2, c1[i])
        sheet1.write(t1,3, c2[i])
        t1=t1+1
    sheet1.write(t1+1,0,'Total')
    sheet1.write(t1+1,1,min)
    sheet1.write(t1+1,2,k)
    sheet1.write(t1+1,3,k1)
    sheet1.write(t1+2,0,per)
    wb.save('data3.xls')
