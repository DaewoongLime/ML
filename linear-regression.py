import numpy as np
import matplotlib.pyplot as plt
import math
import csv

hours = []
grades = []

# get data from training set
with open('training_set.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip header
    for row in reader:
        hours.append(float(row[0]))
        grades.append(float(row[1]))

#convert data into np arrays
hours = np.array(hours)
grades = np.array(grades)
n = len(hours) # number of rows

# hypothesis function
def f(x, w = 0, b = 0): 
    return w * x + b

# cost function
def c(w, b):
    return np.sum((f(hours, w, b) - grades) ** 2)

def grad_desc(w, b, a = 0.01):
    for _ in range(10000):
        tmp = f(hours, w, b) - grades
        tmpw = np.mean(tmp * hours)
        tmpb = np.mean(tmp)

        w -= a * tmpw
        b -= a * tmpb

        if abs(tmpw) < 0.000001 and abs(tmpb) < 0.000001:
            break
    return w,b

w, b = grad_desc(0,0)
x = math.floor(min(hours))
y = math.ceil(max(hours))
plt.plot(hours, grades, 'o')
plt.plot([x, y], [f(x, w, b), f(y, w, b)])
plt.show()
print("f(x) : %sx + %s" % (round(w,3), round(b,3)))