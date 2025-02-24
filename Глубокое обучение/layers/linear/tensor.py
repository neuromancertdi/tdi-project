from math import e
from random import randint
from copy import copy

def add_matrixes(mat1, mat2):
    return [[mat1[i][j] + mat2[i][j] for j in range(len(mat2[i]))] for i in range(len(mat1))]


def sub_matrixes(mat1, mat2):
    return [[mat1[i][j] - mat2[i][j] for j in range(len(mat2[i]))] for i in range(len(mat1))]


def mul_matrixes(mat1, mat2):
    return [[mat1[i][j] * mat2[i][j] for j in range(len(mat2[i]))] for i in range(len(mat1))]


def matrix_div_matrix(mat1, mat2):
    return [[mat1[i][j] / mat2[i][j] for j in range(len(mat2[i]))] for i in range(len(mat1))]


def matrix_div_num(mat, num):
    return [[mat[i][j] / num for j in range(len(mat[i]))] for i in range(len(mat))]


def num_div_matrix(num, mat):
    return [[num / mat[i][j] for j in range(len(mat[i]))] for i in range(len(mat))]


def num_mul_matrix(number, mat):
    return [[mat[i][j] * number for j in range(len(mat[i]))] for i in range(len(mat))]


def neg_matrix(mat):
    return [[-mat[i][j] for j in range(len(mat[i]))] for i in range(len(mat))]


def sum_matrix(mat, dim=0):
    if dim == 0:
        # print(len(mat[0]))
        # print([[sum(slice_column(mat, column_idx)) for column_idx in range(len(mat[0]))]])
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


def zeros(shape):
    if len(shape) == 1:
        return [[0 for _ in range(shape[0])]]

    elif len(shape) == 2:
        return [[0 for _ in range(shape[1])] for _ in range(shape[0])]


def rand(shape, a=-100, b=100):
    if len(shape) == 1:
        return [[randint(a, b) / 100 for _ in range(shape[0])]]

    elif len(shape) == 2:
        return [[randint(a, b) / 100 for _ in range(shape[1])] for _ in range(shape[0])]



class Tensor:
    def __init__(self, data, creators=None, operation_name=""):
        self.data = data
        self.creators = creators
        self.operation_name = operation_name
        self.grad = None

    def shape(self):
        return len(self.data), len(self.data[0])

    def __sub__(self, other):  # вычитание операнда справа
        return Tensor(
            data=sub_matrixes(self.data, other.data),
            creators=[self, other],
            operation_name="__sub__"
        )

    def __mul__(self, other, data=None):  # умножение на тензор или число с правой стороны
        if isinstance(other, float) or isinstance(other, int):
            data = num_mul_matrix(other, self.data)

        elif isinstance(other, Tensor):
            data = mul_matrixes(self.data, other.data)

        return Tensor(
            data=data,
            creators=[self, other],
            operation_name="__mul__"
        )

    def __rmul__(self, other):  # умножение на число с левой стороны
        return self.__mul__(other)


    def __truediv__(self, other, data=None):  # разделить на число или на тензор
        if isinstance(other, int) or isinstance(other, float):
            data = matrix_div_num(self.data, other)

        elif isinstance(other, Tensor):
            data = matrix_div_matrix(self.data, other.data)

        return Tensor(
            data=data,
            creators=[self, other],
            operation_name="__truediv__"
        )

    def __rtruediv__(self, other):  # разделить число на тензор
        return Tensor(
            data=num_div_matrix(other, self.data),
            creators=[other, self],
            operation_name="__rtruediv__"
        )

    def sum(self, dim=0, copies_num=None):
        if dim == 0:
            copies_num = len(self.data)  # суммируем элементы колонки для каждой колонки

        elif dim == 1:
            copies_num = len(self.data[0])  # суммируем элементы строки для каждой строки

        return Tensor(
            data=sum_matrix(self.data, dim),
            creators=[self],
            operation_name="sum_" + str(dim) + '_' + str(copies_num)
        )

    def __add__(self, other):
        return Tensor(
            data=add_matrixes(self.data, other.data),
            creators=[self, other],
            operation_name="__add__"
        )

    def __neg__(self):
        return Tensor(
            data=neg_matrix(self.data),
            creators=[self],
            operation_name="__neg__"
        )

    def dot(self, other):
        return Tensor(
            data=matrix_by_matrix(self.data, other.data),
            creators=[self, other],
            operation_name="dot"
        )

    def transpose(self):
        return Tensor(
            data=transpose_matrix(self.data),
            creators=[self],
            operation_name="transpose"
        )

    def sigmoid(self):
        return Tensor(
            data=sigmoid(self.data),
            creators=[self],
            operation_name="sigmoid"
        )

    def expand(self, dim, copies_num, item=None):
        if dim == 0:
            item = self.data[0]

        elif dim == 1:
            item = slice_column(self.data, idx=0)

        return Tensor(
            data=expand_item(item, dim, copies_num),
            creators=[self],
            operation_name="expand_" + str(dim)
        )

    def __repr__(self):
        return str(self.data.__repr__())


    def backward(self, grad=None):
        if grad is None:
            grad = Tensor(data=ones(shape=self.shape()))

        if self.grad is None:
            self.grad = Tensor(grad.data)

        else:
            self.grad = self.grad + grad

        # если передавать не новый тензор градиента, а тот же, то из-за этого оригинальный
        # тензор будет изменен и вычисление градиента на других шагах может быть неправильным

        if self.operation_name == "__mul__":
            self_grad = grad * self.creators[1]  # new Tensor
            self.creators[0].backward(self_grad)

            if isinstance(self.creators[1], Tensor):
                other_grad = grad * self.creators[0]  # new Tensor
                self.creators[1].backward(other_grad)

        elif self.operation_name == "__truediv__":
            self_grad = grad / self.creators[1]  # new Tensor
            self.creators[0].backward(self_grad)

            if isinstance(self.creators[1], Tensor):
                other_grad = (-self * grad) / (self.creators[1] * self.creators[1])  # new Tensor
                self.creators[1].backward(other_grad)

        elif self.operation_name == "__rtruediv__":
            other_grad = (-self * grad) / (self.creators[1] * self.creators[1])  # new Tensor
            self.creators[1].backward(other_grad)


        elif self.operation_name == "__sub__":
            self_grad = Tensor(grad.data)  # new Tensor
            self.creators[0].backward(self_grad)
            other_grad = -grad  # new Tensor
            self.creators[1].backward(other_grad)

        elif self.operation_name.startswith("sum"):
            dim, copies_num = self.operation_name.split('_')[1:]
            dim, copies_num = int(dim), int(copies_num)
            new_grad = grad.expand(dim, copies_num)  # new Tensor
            self.creators[0].backward(new_grad)

        elif self.operation_name == "__add__":
            self.creators[0].backward(Tensor(grad.data))  # new Tensor
            self.creators[1].backward(Tensor(grad.data))  # new Tensor

        elif self.operation_name == "__neg__":
            self.creators[0].backward(-grad)  # new Tensor

        elif self.operation_name == "dot":
            self_grad = grad.dot(self.creators[1].transpose())  # new Tensor
            self.creators[0].backward(self_grad)
            other_grad = self.creators[0].transpose().dot(grad)  # new Tensor
            self.creators[1].backward(other_grad)

        elif self.operation_name == "transpose":
            new_grad = grad.transpose()  # new Tensor
            self.creators[0].backward(new_grad)

        elif self.operation_name.startswith("expand"):
            dim = int(self.operation_name.split('_')[1])
            new_grad = grad.sum(dim)  # new Tensor
            self.creators[0].backward(new_grad)

        elif self.operation_name == "sigmoid":
            ones_tensor = Tensor(ones(shape=self.shape()))
            new_grad = grad * self * (ones_tensor - self)  # new Tensor
            self.creators[0].backward(new_grad)