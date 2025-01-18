from matrix_operations import *

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

    def __mul__(self, other):  # умножение на тензор или число с правой стороны
        return Tensor(
            data=mul_matrixes(self.data, other.data),
            creators=[self, other],
            operation_name="__mul__"
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

        if dim == 1:
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
            self.grad += grad

        # если передавать не новый тензор градиента, а тот же, то из-за этого оригинальный
        # тензор будет изменен и вычисление градиента на других шагах может быть неправильным

        if self.operation_name == "__mul__":
            self_grad = grad * self.creators[1]  # new Tensor
            self.creators[0].backward(self_grad)
            other_grad = grad * self.creators[0]  # new Tensor
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
            ones_tensor = Tensor(data=ones(shape=self.shape()))
            new_grad = grad * self * (ones_tensor - self)  # new Tensor
            self.creators[0].backward(new_grad)



if __name__ == '__main__':
    def main():
        x0 = Tensor(data=[[0.8, 0.9]])
        x = x0.transpose()
        # print("x =", x)

        w1 = Tensor(data=[[0.1, 0.3], [0.2, 0.4]])
        # print("w1 =", w1)

        w2 = Tensor(data=[[0.5, 0.6]])
        # print("w2 =", w2)

        t = Tensor(data=[[1]])
        # print("t =", t)

        z1 = w1.dot(x)
        print("z1 =", z1)

        h = z1.sigmoid()
        print("h =", h)

        z2 = w2.dot(h)
        print("z2 =", z2)

        y = z2.sigmoid()
        print("y =", y)

        y1 = (y - t)
        print("y1 =", y1)
        y2 = (y - t)
        print("y2 =", y2)
        error = (y1 * y2).sum() #/ y.shape()[0]
        print("error =", error)

        print("=========================")

        error.backward()
        print("x0.grad =", x0.grad)
        print("x.grad =", x.grad)
        print("w1.grad =", w1.grad)
        print("z1.grad =", z1.grad)
        print("h.grad =", h.grad)
        print("w2.grad =", w2.grad)
        print("z2.grad =", z2.grad)
        print("y.grad =", y.grad)
        print("t.grad =", t.grad)
        print("y1.grad =", y1.grad)
        print("y2.grad =", y2.grad)
        print("error.grad =", error.grad)

    main()
