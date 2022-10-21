import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('simvision_pixel_TDC.csv', skiprows=1) ## Skip first row SimTime = 0

### Original column names
### 'SimTime',
### 'ETROC2_tb.ETROC2_inst.PixelMatrix_HTree_inst.PixelCol_inst_1.Pixel_inst_2.PixelAnalog_inst.TDC_CAL[9:0]'
### 'ETROC2_tb.ETROC2_inst.PixelMatrix_HTree_inst.PixelCol_inst_1.Pixel_inst_2.PixelAnalog_inst.TDC_TOA[9:0]'
### 'ETROC2_tb.ETROC2_inst.PixelMatrix_HTree_inst.PixelCol_inst_1.Pixel_inst_2.PixelAnalog_inst.TDC_TOT[8:0]'
### 'ETROC2_tb.ETROC2_inst.PixelMatrix_HTree_inst.PixelCol1_inst_14.Pixel1_inst_14.PixelAnalog1_inst.TDC_CAL[9:0]'
### 'ETROC2_tb.ETROC2_inst.PixelMatrix_HTree_inst.PixelCol1_inst_14.Pixel1_inst_14.PixelAnalog1_inst.TDC_TOA[9:0]'
### 'ETROC2_tb.ETROC2_inst.PixelMatrix_HTree_inst.PixelCol1_inst_14.Pixel1_inst_14.PixelAnalog1_inst.TDC_TOT[8:0]'

### Replace column names (only when TDC file has been opened)
df.columns = ['SimTime', 'Row1Col2_TDC_CAL', 'Row1Col2_TDC_TOA', 'Row1Col2_TDC_TOT', 
              'Row14Col14_TDC_CAL', 'Row14Col14_TDC_TOA', 'Row14Col14_TDC_TOT']

### Covert hex to decimal
row1col2_cal = df['Row1Col2_TDC_CAL'].squeeze()
row1col2_toa = df['Row1Col2_TDC_TOA'].squeeze()
b16 = lambda x: int(x, 16)
row1col2_cal = row1col2_cal.apply(b16)
row1col2_toa = row1col2_toa.apply(b16)

### plotting
get_ipython().run_line_magic('matplotlib', 'inline')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("1st row, 2nd column pixel TDC values", fontsize=14)
ax1.hist(row1col2_cal)
ax1.set_xlabel('CAL code', fontsize=12)

ax2.scatter(row1col2_toa, df['Row1Col2_TDC_TOT'])
ax2.grid()
ax2.set_xlabel('TOA code', fontsize=12)
ax2.set_ylabel('TOT code', fontsize=12)

### Covert hex to decimal
row14col14_cal = df['Row14Col14_TDC_CAL'].squeeze()
row14col14_toa = df['Row14Col14_TDC_TOA'].squeeze()
row14col14_cal = row14col14_cal.apply(b16)
row14col14_toa = row14col14_toa.apply(b16)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("14th row, 14th column pixel TDC values", fontsize=14)
ax1.hist(row14col14_cal)
ax1.set_xlabel('CAL code', fontsize=12)

ax2.scatter(row14col14_toa, df['Row14Col14_TDC_TOT'])
ax2.grid()
ax2.set_xlim(425, 438)
ax2.set_xlabel('TOA code', fontsize=12)
ax2.set_ylabel('TOT code', fontsize=12)

###################################################################
df2 = pd.read_csv('simv ision_fastcmd_data.csv', skiprows=1)

### Original column names
### 'SimTime' 
### 'ETROC2_tb.FCGenLocal_inst.FastComByte[7:0]'

### Replace column names (only when fast command file has been opened)
df2.columns = ['SimTime', 'FastComByte']
df2.head()

df2['FastComByte']
### F0 (1111 0000): Idle
### 69 (0110 1001): Charge injection
### 5A (0101 1010): BCR (Bunch counter reset at BC0, reset the counter to zero)
### 96 (1001 0110): L1A
### Reference: https://indico.cern.ch/event/1127562/contributions/4904781/attachments/2454504/4319592/ETROCemulator%20slides20220921.pdf

### create a list of our conditions
conditions = [
    (df2['FastComByte'] == 'F0'),
    (df2['FastComByte'] == '69'),
    (df2['FastComByte'] == '5A'),
    (df2['FastComByte'] == '96')
    ]

### create a list of the values we want to assign for each condition
values = ['Idle', 'Charge injection', 'BCR', 'L1A']

### create a new column and use np.select to assign values to it using our lists as arguments
df2['fcmd_defn'] = np.select(conditions, values)

### display updated DataFrame
print(df2)

###################################################################
df3 = pd.read_csv('simvision.csv', skiprows=1)
df3.columns = ['SimTime', 'DOR', 'DOL']
df3 = df3.drop(columns=['DOL'])
df3 = df3.iloc[:-1 , :] ## drop the last row

## Temporary data frame to calculate time difference
sim = df3.diff(periods=-1).fillna(0)
sim['SimTime'] = sim['SimTime'].abs()*0.001
sim['nth'] = sim['SimTime']/1560.

df3['diff'] = sim['SimTime'] ## Time difference between ((n-1)th, nth) row, n = 0, 1, 2, ...
df3['cycle'] = sim['nth']

### Repeat rows in a pandas DataFrame based on column value
### https://stackoverflow.com/questions/47336704/repeat-rows-in-a-pandas-dataframe-based-on-column-value
bit_df = df3.reindex(df3.index.repeat(df3.cycle))
bit_df = bit_df.drop(columns=['SimTime', 'diff', 'cycle'])
bit_df = bit_df.iloc[20:, :] ## drop the first 20 rows
bit_df = bit_df.reset_index(drop=True)


### 3C5C pattern recognition
### Only check whether the data has 3C5C pattern at anywhere or not
### 3 C 5 C (hex)
### 0011 1100 0101 1100 (binary)
pattern = [0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0] ## Since raw data bit is reversed!

matched = bit_df.rolling(len(pattern)).apply(lambda x: all(np.equal(x, pattern)), raw=True)
matched = matched.sum(axis = 1).astype(bool)   #Sum to perform boolean OR
idx_matched = np.where(matched)[0]
subset = [(matc-len(pattern)+1, matc+1, matc+25) for matc in idx_matched]

output = []
for subs in subset:
    bit16 = bit_df.iloc[range(subs[0], subs[1]), :].values.flatten().tolist()
    bit24 = bit_df.iloc[range(subs[1], subs[2]), :].values.flatten().tolist()
    bit16.reverse() ## Raw data bit order is reversed!
    bit24.reverse() ## Raw data bit order is reversed!
    s1 = ''.join([str(x) for x in bit16])
    s2 = ''.join([str(x) for x in bit24])
    output.append([s1, s2, format(int(s1, 2), 'X'), format(int(s2, 2), 'X')])


import os
os.system('mkdir -p output/')

new_df = pd.DataFrame(columns=['bit16', 'bit24', 'bit16_hex', 'bit24_hex'], data=output)
new_df.to_csv('output/result.csv', index=False)


#subset1 = []
#subset2 = []
#
#output = []
#for i in range(int(bit_df.shape[0]/40)):
#    x1, x2 = 0+40*i, 16+40*i
#    y1, y2 = 16+40*i, 40+40*i
#    bit16 = bit_df.iloc[range(x1, x2), :].values.flatten().tolist()
#    bit24 = bit_df.iloc[range(y1, y2), :].values.flatten().tolist()
#    bit16.reverse() ## Raw data bit order is reversed!
#    bit24.reverse() ## Raw data bit order is reversed!
#    s1 = ''.join([str(x) for x in bit16])
#    s2 = ''.join([str(x) for x in bit24])
#    output.append([s1, s2, format(int(s1, 2), 'X'), format(int(s2, 2), 'X')])
#
#    if i < 10:
#        print(s1, s2, format(int(s1, 2), 'X'), format(int(s2, 2), 'X'))
#
#
#new_df = pd.DataFrame(columns=['bit16', 'bit24', 'bit16_hex', 'bit24_hex'], data=output)
#new_df.to_csv('result1.csv', index=False)


### 6AF3 pattern
### 6 A F 3
### 0110 1010 1111 0011
#pattern = [1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0] ## Since raw data bit is reversed!
#
#matched = bit_df.rolling(len(pattern)).apply(lambda x: all(np.equal(x, pattern)), raw=True)
#matched = matched.sum(axis = 1).astype(bool)   #Sum to perform boolean OR
#idx_matched = np.where(matched)[0]
#subset = [(match-len(pattern)+1, match+1, match+25) for match in idx_matched]
