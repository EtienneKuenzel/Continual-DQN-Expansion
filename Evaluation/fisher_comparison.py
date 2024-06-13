import csv
import numpy as np
from scipy.linalg import sqrtm
from math import sqrt

# Specify the path to your CSV file
file_path = 'fisher_info.csv'

# Open the CSV file
with open(file_path, mode='r') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)
    a = []
    b=0
    # Iterate through the rows
    for row in csv_reader:
        for x in row:
            if b == 1000:
                break
            b+=1
            a.append(float(x))

def unit_normalize(arr):
    total_sum = sum(arr)
    normalized_arr = [x / total_sum for x in arr]
    return normalized_arr


def euclidean_distance(point1, point2):
    # Calculate Euclidean distance between two points
    return sqrt((point1 - point2)**2)


def compute_frechet_distance(A, B):
    n = len(A)
    m = len(B)

    # Initialize a distance matrix
    D = [[-1] * m for _ in range(n)]

    # Compute Frechet distance using dynamic programming approach
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                D[i][j] = euclidean_distance(A[0], B[0])
            elif i > 0 and j == 0:
                D[i][j] = max(D[i - 1][0], euclidean_distance(A[i], B[0]))
            elif i == 0 and j > 0:
                D[i][j] = max(D[0][j - 1], euclidean_distance(A[0], B[j]))
            elif i > 0 and j > 0:
                D[i][j] = max(min(D[i - 1][j], D[i - 1][j - 1], D[i][j - 1]), euclidean_distance(A[i], B[j]))

    return D[n - 1][m - 1]

# Example usage
F1 = unit_normalize(a)
F2 = unit_normalize(a)
print("Frechet distance between A and B:", compute_frechet_distance(F1, F2))

