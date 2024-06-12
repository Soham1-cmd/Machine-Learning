import pandas as pd

mydataset = {
    'car':['BMW','Vovlo','Tata'],
    'passing':[3,5,7]
}
#check
#yar = pd.Series(mydataset)
#print(myar.loc[1])

a = [1,2,3]
myvar = pd.Series(a,index =['x','y','z'])
print(myvar)

#Create a DataFrame from two Series:
data = {
"calories": [420, 380, 390],
"duration": [50, 40, 45]
}
myvar = pd.DataFrame(data)
print(myvar)

#Create a simple Pandas DataFrame:
dtframe = {
    'name':['Soham',"Mate"],
    'value':[1,2]
}

#Locate Row
myname = pd.DataFrame(dtframe)
print(myname)
print("Location",myname.loc[0])

df = pd.read_csv('data.csv')
#print(df)
#print(df.to_string())
pd.options.display.max_rows = 9999
#print(pd.options.display.max_rows)
print(df.tail(10))