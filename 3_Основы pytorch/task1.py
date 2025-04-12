# Задание 1:
# 1 Создайте тензор x целочисленного типа, хранящий случайное значение.
# 2 Преобразуйте тензор к типу float32;
# 3 Проведите с тензором x ряд операций:
# – возведение в степень n, где n = 3, если ваш номер по списку группы в ЭИОС – четный и n = 2, если ваш номер по списку группы в ЭИОС – нечетный;
# – умножение на случайное значение в диапазоне от 1 до 10;
# – взятие экспоненты от полученного числа.
# 4 Вычислите и выведите на экран значение производной для полученного в п.3 значения по x.

from random import randint

import torch
x = torch.randint(0, 10, (3, 3),dtype=torch.int32)
print(x)

x = x.to(dtype=torch.float32)
x.requires_grad = True
print(x)

x_pow = x ** 2
print(x_pow)
x_rand = x_pow * randint(1, 10)
print(x_rand)
x_exp = torch.exp(x_rand)
print(x_exp)

x_exp.mean().backward()
print(x.grad)