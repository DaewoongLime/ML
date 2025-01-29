import numpy as np
import csv

hours = np.array([])
grades = np.array([])

# get data from training set
with open('training_set.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        hours = np.append(hours, float(row[0]))
        grades = np.append(grades, int(row[1]))

print(hours)
print(grades)
