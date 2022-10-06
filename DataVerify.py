import pandas as pd
import numpy as np

df = pd.read_csv('simvision.csv')
print(df.columns) ## Print name of columns

# df['ETROC2_tb.DOR']==1 ## The selection when data is coming from the Right port
# df['ETROC2_tb.DOL']==1 ## The selection when data is coming from the Left port, currently the Left port was not used. 


print(df.head(), '\n') ## Print the first five data
print(df.tail(), '\n') ## Print the last five data
print(df['ETROC2_tb.DOR'][:20]) ## Print the first 20 data of ETROC2_tb.DOR column

### 3C5C pattern recognition
### Only check whether the data has 3C5C pattern at anywhere or not
### 3 C 5 C (hex)
### 0011 1100 0101 1100 (binary)
pattern = [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0]
matched = df.rolling(len(pattern)).apply(lambda x: all(np.equal(x, pattern)), raw=True)
matched = matched.sum(axis = 1).astype(bool)   #Sum to perform boolean OR
idx_matched = np.where(matched)[0]
subset = [range(match-len(pattern)+1, match+1) for match in idx_matched]

print(subset) ### Print results
