import csv
import numpy as np
from scipy.linalg import sqrtm
from math import sqrt

# Specify the path to your CSV file
file_path = '0_fisher_info.csv'
file_path1 = '1_fisher_info.csv'

# Open the CSV file
with open(file_path, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    a = []
    b=0
    # Iterate through the rows
    for row in csv_reader:
        for x in row:
            b+=1
            a.append(float(x))
with open(file_path1, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    c = []
    b=0
    # Iterate through the rows
    for row in csv_reader:
        for x in row:
            b+=1
            c.append(float(x))

def unit_normalize(arr):
    total_sum = np.sum(arr)
    normalized_arr = arr / total_sum
    return normalized_arr

def euclidean_distance(point1, point2):
    return np.sqrt((point1 - point2)**2)

def compute_frechet_distance(A, B):
    n = len(A)
    m = len(B)

    # Initialize two rows for rolling window
    prev_row = np.full(m, -1.0)
    curr_row = np.full(m, -1.0)

    # Compute Frechet distance using dynamic programming approach with rolling window
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                curr_row[j] = euclidean_distance(A[0], B[0])
            elif i > 0 and j == 0:
                curr_row[j] = max(prev_row[0], euclidean_distance(A[i], B[0]))
            elif i == 0 and j > 0:
                curr_row[j] = max(curr_row[j - 1], euclidean_distance(A[0], B[j]))
            elif i > 0 and j > 0:
                curr_row[j] = max(min(prev_row[j], prev_row[j - 1], curr_row[j - 1]), euclidean_distance(A[i], B[j]))

        # Move current row to previous row for next iteration
        prev_row[:] = curr_row[:]

    return curr_row[m - 1]


# Example usage
F1 = unit_normalize(a)
F2 = unit_normalize(c)
print("Frechet distance between A and B:", compute_frechet_distance(F1, F2))

