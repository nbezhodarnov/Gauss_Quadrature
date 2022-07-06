import numpy as np
import matplotlib.pyplot as plt
import math
import itertools

def function(x_input):
	return (x_input ** 2 + 2 * x_input + math.log10(x_input)) ** (1/2)

def Legendre_polynom(x_input, n):
	if (n == 0):
		return 1
	if (n == 1):
		return x_input
	return ((2 * n - 1) * x_input * Legendre_polynom(x_input, n - 1) - (n - 1) * Legendre_polynom(x_input, n - 2)) / n
	
def Legendre_polynom_derivative(x_input, n):
	if (n == 0):
		return 0
	if (n == 1):
		return 1
	return n * (Legendre_polynom(x_input, n - 1) - x_input * Legendre_polynom(x_input, n)) / (1 - x_input ** 2)
	
def Legendre_polynom_roots(n):
	roots = np.zeros(n, dtype = np.float64)
	for i in range(n // 2):
		if ((4 * (i + 1) - 1) / (4 * n + 2) == 1 / 2):
			temp = 0
		else:
			temp = math.cos(math.pi * (n - i - 0.25) / (n + 0.5))
		for k in range(7):
			temp -= Legendre_polynom(temp, n) / Legendre_polynom_derivative(temp, n)
		roots[i] = temp
	if (n % 2 == 1):
		roots[n // 2] = 0
	for i in range(n // 2 + n % 2, n):
		roots[i] = -roots[n - i - 1]
	return roots

def Gauss_Quadrature_rule(start, end, n, error):
	roots = Legendre_polynom_roots(n)
	a = np.zeros(n, dtype = np.float64)
	step = 1
	dots_count = n
	for i in range(n // 2):
		a[i] = 2 / ((1 - roots[i] ** 2) * (Legendre_polynom_derivative(roots[i], n) ** 2))
	if (n % 2 == 1):
		a[n // 2] = 2 / ((1 - roots[n // 2] ** 2) * (Legendre_polynom_derivative(roots[n // 2], n) ** 2))
	for i in range(n // 2 + n % 2, n):
		a[i] = a[n - i - 1]
	step = 2
	sum = 0
	for i in range(n):
		sum += a[i] * function((end + start) / 2 + (end - start) * roots[i] / 2)
	result = ((end - start) / 2) * sum
	sum = 0
	for i in range(2):
		for k in range(n):
			sum += a[k] * function(start + (end - start) * (roots[k] + 2 * i + 1) / 4)
	result_next = ((end - start) / 4) * sum
	while (abs(result - result_next) > error):
		step += 1
		result = result_next
		dots_count *= 2
		sum = 0
		for i in range(2 ** (step - 1)):
			for k in range(n):
				sum += a[k] * function(start + (end - start) * (roots[k] + 2 * i + 1) / (2 ** step))
		result_next = ((end - start) / (2 ** step)) * sum
	dots_count *= 2
	print('Result: ', result_next)
	print('Count of dots: ', dots_count)
	return result_next

def main():
	n1 = 4
	n2 = 7
	print(n1, ' dots:')
	Gauss_Quadrature_rule(2, 3, n1, 0.000005 / 10000000000)
	print(n2, ' dots:')
	Gauss_Quadrature_rule(2, 3, n2, 0.000005 / 10000000000)
	
if __name__ == '__main__':
    main()
