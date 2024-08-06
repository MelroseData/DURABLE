"""
the initial data purify
combined with the preprocessing giving the first instinct

it's my first time making those process, building models, making algorithms, writing paper
so that can only stands for my own understanding,
I want to lay some highlights right here for better understanding,
since I am not major in the computer programs but still doing the coding (some people that I am aware are the same)
Personally taking the code more seriously, so I study the code before the algorithms and base knowledge
If you hold the same belief :0, I got you ;)

you can find both of these files is in the form of IDEAL instead of oop
because they are relatively simple
"""


import pandas as pd
from pathlib import Path
"""
the first part 
get the right location and all files
here is a thing the "purify" and "preprocessing" both contains this function 
but with different apis

this file use the pathlib's glob function   
which will return a generator
build for the later iterator
Loaded each CSV file into a DataFrame


located and return the value, that's all it's function does
"""

FatherPath = Path(__file__).parents[1]
print(FatherPath)
CurrentPath = FatherPath /"paper open source"/"clear"/"level4"
csv_files = CurrentPath.glob("*.csv")

"""
the second part
bsed on the original files' feature do certain improvement

The original dataset don’t have the title and contains missing values
so an additional column 'num' is created by extracting the numerical part of the 'Unnamed: 0' column
and the missing rows in the range from 1 to the maximum number in the 'num' column are identified and added to the DataFrame. 
Then sorted based on the 'num' column, which is dropped after usage.

The cleaned DataFrame is saved back to a new CSV file 
with a "_upgrade" suffix then grouped into three levels based on the biological layers. 
Each level's CSV files are read into separate DataFrames and set 'Unnamed: 0' as the index. 
"""

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df['num'] = df['Unnamed: 0'].str.extract('(\d+)').astype(int)
    max_num = df['num'].max()
    missing_nums = set(range(1, max_num + 1)) - set(df['num'])
    missing_df = pd.DataFrame({'Unnamed: 0': ['ACH-' + str(num).zfill(6) for num in missing_nums], 'num': list(missing_nums)})
    df = pd.concat([df, missing_df])
    df = df.sort_values('num')
    df = df.drop(columns=['num'])
    sorted_file_path = csv_file.stem + "_upgrade.csv"
    df.to_csv(sorted_file_path, index=False)
    """	The main operations in the code are reading the CSV file, extracting numbers from a column, 
    finding missing numbers, creating a DataFrame, concatenating DataFrames, 
    sorting, and writing to a CSV file. The most time-consuming operation is likely the sorting step, 
    which has a time complexity of O(n log n) where n is the number of rows in the DataFrame."""