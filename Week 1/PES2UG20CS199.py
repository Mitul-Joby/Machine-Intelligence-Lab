'''
UE20CS302 (D Section)
Machine Intelligence 
Week 1: Numpy and Pandas

Mitul Joby
PES2UG20CS199
'''

import numpy as np
import pandas as pd


def create_numpy_ones_array(shape):
    array = np.ones(shape)
    return array


def create_numpy_zeros_array(shape):
    array = np.zeros(shape)
    return array


def create_identity_numpy_array(order):
    array = np.identity(order)
    return array


def matrix_cofactor(array):
    determinant = np.linalg.det(array)
    if(determinant != 0):
        cofactor = np.linalg.inv(array).T * determinant
        return cofactor
    else:
        raise Exception("Singular Matrix")

def f1(X1, coef1, X2, coef2, seed1, seed2, seed3, shape1, shape2):
    np.random.seed(seed1)
    W1 = np.random.random(shape1)
    np.random.seed(seed2)
    W2 = np.random.random(shape2)
    
    np.random.seed(seed3)
    shapeX1 = X1.shape
    shapeX2 = X2.shape

    ans = None
    if (shape1[1] == shapeX1[0]) and (shape2[1] == shapeX2[0]) and (shape1[0] == shape2[0]) and (shapeX1[1] == shapeX2[1]):
        shapeB = (shape1[0], shapeX1[1])
        B = np.random.random(shapeB)
        ans = np.matmul(W1, (X1 ** coef1)) + np.matmul(W2, (X2 ** coef2)) + B
    else:
        ans = -1
    return ans


def fill_with_mode(filename, column):
    """
    Fill the missing values(NaN) in a column with the mode of that column
    Args:
        filename: Name of the CSV file.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df = pd.read_csv(filename)
    df[column].fillna(df[column].mode()[0], inplace = True)
    return df


def fill_with_group_average(df, group, column):
    """
    Fill the missing values(NaN) in column with the mean value of the 
    group the row belongs to.
    The rows are grouped based on the values of another column

    Args:
        df: A pandas DataFrame object representing the data.
        group: The column to group the rows with
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
        (Representing entire data and where 'column' does not contain NaN values)
        (Filled with above mentioned rules)
    """
    df[column].fillna(df.groupby(group)[column].transform('mean'), inplace=True)
    return df


def get_rows_greater_than_avg(df, column):
    """
    Return all the rows(with all columns) where the value in a certain 'column'
    is greater than the average value of that column.

    row where row.column > mean(data.column)

    Args:
        df: A pandas DataFrame object representing the data.
        column: Name of the column to fill
    Returns:
        df: Pandas DataFrame object.
    """
    return df[df[column] > df[column].mean()]
     
