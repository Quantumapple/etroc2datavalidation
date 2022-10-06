import pandas as pd

df = pd.read_csv('simvision.csv')
print(df.columns) ## Print name of columns

# df['ETROC2_tb.DOR']==1 ## The selection when data is coming from the Right port
# df['ETROC2_tb.DOL']==1 ## The selection when data is coming from the Left port, currently the Left port was not used. 

print(df.head()) ## Print First five data
print(df.tail()) ## Print the last five data
