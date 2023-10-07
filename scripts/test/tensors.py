import autodiff.dl.tensor as t

x1 = t.Tensor([1, 2, 3])
x2 = t.Tensor([4, 5, 6])

print(x1)
print(x2)
print('sum', x1 + x2)
print('sub', x1 - x2)
print('neg', -x1)
