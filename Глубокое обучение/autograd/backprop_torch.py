from torch import tensor

x0 = tensor([[0.8, 0.9]])
x = x0.transpose(1, 0)
# x.requires_grad = True
print("x =", x)

w1 = tensor(data=[[0.1, 0.3], [0.2, 0.4]], requires_grad=True)
print("w1 =", w1)

w2 = tensor(data=[[0.5, 0.6]], requires_grad=True)
print("w2 =", w2)

t = tensor(data=[[1]])
print("t =", t)

z1 = w1.matmul(x)
print("z1 =", z1)

h = z1.sigmoid()
print("h =", h)

z2 = w2.matmul(h)
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

