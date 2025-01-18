from math import e

def add_matrixes(mat1, mat2):
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat2[i]))] for i in range(len(mat1))]


def sub_matrixes(mat1, mat2):
    return [[mat1[i][j] - mat2[i][j] for j in range(len(mat2[i]))] for i in range(len(mat1))]


def mul_matrixes(mat1, mat2):
    return [[mat1[i][j] * mat2[i][j] for j in range(len(mat2[i]))] for i in range(len(mat1))]


def neg_matrix(mat):
    return [[-mat[i][j] for j in range(len(mat[i]))] for i in range(len(mat))]


def sum_matrix(mat, dim=0):
    if dim == 0:
        return [[sum(slice_column(mat, column_idx)) for column_idx in range(len(mat[0]))]]

    elif dim == 1:
        return [[sum(mat[i]) for i in range(len(mat))]]


def sigmoid(matrix):
    return [[1 / (1 + e ** (-z_ij)) for z_ij in matrix[i]] for i in range(len(matrix))]


def vector_by_vector(vector1, vector2):
    if len(vector1) != len(vector2):
        raise Exception("dim1 != dim2")

    return sum([vector1[i] * vector2[i] for i in range(len(vector1))])


def slice_column(matrix, idx):
    return [matrix[i][idx] for i in range(len(matrix))]


def matrix_by_matrix(matrix1, matrix2):
    if len(matrix1[0]) != len(matrix2):
        raise Exception("dim1 != dim2")

    new_matrix = list()

    for i in range(len(matrix1)):
        new_matrix.append(list())

        for j in range(len(matrix2[0])):
            new_matrix[-1].append(vector_by_vector(matrix1[i], slice_column(matrix2, j)))

    return new_matrix


def transpose_matrix(matrix):
    new_matrix = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            new_matrix[j][i] = matrix[i][j]

    return new_matrix

from copy import copy

def expand_item(item, dim, copies_num):  # item - vector
    if dim == 0:
        return [copy(item) for _ in range(copies_num)]

    elif dim == 1:
        return [[item[j] for _ in range(copies_num)] for j in range(len(item))]


def ones(shape):
    if len(shape) == 1:
        return [[1 for _ in range(shape[0])]]

    elif len(shape) == 2:
        return [[1 for _ in range(shape[1])] for _ in range(shape[0])]

    else:
        raise Exception(f"ther's no operation for shape='{shape}'")


def zeros(shape):
    if len(shape) == 1:
        return [[0 for _ in range(shape[0])]]

    elif len(shape) == 2:
        return [[0 for _ in range(shape[1])] for _ in range(shape[0])]

    else:
        raise Exception(f"ther's no operation for shape='{shape}'")