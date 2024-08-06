"""
Because the code might not be able to fully handle the problem, so I just put some code here
this one could handle the csv file to split,
it's okay if your file is not in the form of csv, I got the little transfer down there
just want you to know I got you :)
# coding:utf-8
# split the large dataset(I try to use the traditional way but sometimes the number of lines is wrong and I will crash)
"""
import os
import csv
'''transfer the txt file into csv file'''
"""
with open('REAC23Q1.txt','r', encoding='utf-8') as file1:
    lines = csv.reader(file1, delimiter='$')
    destFileName = "REAC.csv"
    with open(destFileName, "w+", encoding='utf-8', newline='') as destFileData:
        writer = csv.writer(destFileData)
        for line in lines:
            writer.writerow(line)"""
""" 
I've use this function to get all protein arrays' name for search the PDB id
Thought you might need that so i just put that right here, make suree you know i got you bro
"""
'''slice them up'''
from pathlib import Path
FatherPath = Path(__file__).parents[1]
CurrentPath = FatherPath /"paper open source"/"clear"/"level3"
sourceFileName = "Protein_Array_upgrade.csv"
print(CurrentPath/sourceFileName)
def cutFile():
    print("Reading file...")
    sourceFileData = open(CurrentPath/sourceFileName, 'r', encoding='utf-8')
    # Line = csv.reader(sourceFileData, delimiter='$')
    ListOfLine =sourceFileData.read().splitlines()  # get all our file sliced then put them in a new file
    n = len(ListOfLine)
    print("Total files" + str(n) + "line")
    print("Please enter the number of files to be split:")
    m = int(input(""))  # Define the number of divided files
    p = n // m + 1
    print("The file needs to be divided into " + str(m) + "sub-files")
    print("Each file has at most " + str(p) + "lines")
    print("Start splitting...")
    for i in range(m):
        print("Generating the" + str(i + 1) + "th Subfile")
        destFileName = os.path.splitext(sourceFileName)[0] + "_part" + str(i) + ".csv"  # Define the newly generated file after splitting
        destFileData = open(destFileName, "w", encoding='utf-8')
        if (i == m - 1):
            for line in ListOfLine[i * p:]:
                destFileData.write(line + '\n')
        else:
            for line in ListOfLine[i * p:(i + 1) * p]:
                destFileData.write(line + '\n')
        destFileData.close()
    print("Split completed")

cutFile()