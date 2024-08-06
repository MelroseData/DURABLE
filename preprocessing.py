"""
for both this file and the purify file are all contain this changeable feature
the dynamic part is made up with located data path,
I got you ;),
just want you to know if you change the folder's name, you need to change that part as well.
"""

import pandas as pd
import glob
from pathlib import Path

"""
the first part 
get the right location and all files
here is a thing the "purify" and "preprocessing" both contains this function 
but with different apis

this file use the glob's glob function   
which will only return a list
because there is no need for iteration
"""

FatherPath = Path(__file__).parents[1]
CurrentPath = FatherPath / "clear"

Level1Path = str(CurrentPath / "level1")
Level2Path = str(CurrentPath / "level2")
Level3Path = str(CurrentPath / "level3")

level1files = glob.glob(Level1Path + "/*.csv")
level2files = glob.glob(Level2Path + "/*.csv")
level3files = glob.glob(Level3Path + "/*.csv")

"""
the second part
The DataFrames within each level are concatenated horizontally 
to create a merged dataset for each level. 
With the same loading, merge them all and return them at the same file path as the algorithms stay.
"""
dataframe1 = [pd.read_csv(file).set_index('Unnamed: 0') for file in level1files]
dataframe2 = [pd.read_csv(file).set_index('Unnamed: 0') for file in level2files]
dataframe3 = [pd.read_csv(file).set_index('Unnamed: 0') for file in level3files]

merged_dataset1 = pd.concat(dataframe1, axis=1)
merged_dataset2 = pd.concat(dataframe2, axis=1)
merged_dataset3 = pd.concat(dataframe3, axis=1)

merged_dataset1.to_csv("L1dataset.csv")
merged_dataset2.to_csv("L2dataset.csv")
merged_dataset3.to_csv("L3dataset.csv")

"""
# this part just showing you the info

csv_file1=pd.read_csv(str(CurrentPath/"L1dataset.csv")).set_index('Unnamed: 0')
csv_file2=pd.read_csv(str(CurrentPath/"L2dataset.csv")).set_index('Unnamed: 0')
csv_file3=pd.read_csv(str(CurrentPath/"L3dataset.csv")).set_index('Unnamed: 0')

csv_file1=csv_file1.reset_index()
csv_file1 = csv_file1.drop(columns=['Unnamed: 0'])
csv_file1 = csv_file1.fillna(csv_file1.mean())

# you also need to know you can do a for loop to print each of those files but i just printed one

print (csv_file1)
dataframe0 = [csv_file1,csv_file2,csv_file3]

dataframes=[dataframe1,dataframe2,dataframe3,dataframe0]
for i, df_list in enumerate(dataframes):
    for j, df in enumerate(df_list):
        print(f"Number of columns in DataFrame {j} in list {i}: {df.shape[1]}")
"""
"""
output is like that:
Number of columns in DataFrame 0 in list 0: 21840
Number of columns in DataFrame 1 in list 0: 16963
Number of columns in DataFrame 0 in list 1: 20219
Number of columns in DataFrame 1 in list 1: 16810
Number of columns in DataFrame 0 in list 2: 734
Number of columns in DataFrame 1 in list 2: 214
Number of columns in DataFrame 2 in list 2: 15576
Number of columns in DataFrame 3 in list 2: 19144
Number of columns in DataFrame 0 in list 3: 38803
Number of columns in DataFrame 1 in list 3: 37029
Number of columns in DataFrame 2 in list 3: 35668
"""
"""	The code reads multiple CSV files from different directories and concatenates them into dataframes. 
The time complexity is linear as it depends on the number of files being read and concatenated, 
which is represented by 'n'."""