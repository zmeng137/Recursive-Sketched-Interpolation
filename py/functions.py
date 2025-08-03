import numpy as np

Function_Collection = {}

Function_Collection[0] = lambda x: 1

Function_Collection[1] = lambda x: 1.2 * x ** 4 - 0.2 * np.sqrt(x) - 1 + 0.6 * np.sin(7.3 * np.pi * x)  
Function_Collection[2] = lambda x: -1.1 * x ** 7 - 12 + np.exp(3.1 * x) - 0.81 * np.cos(6 * np.pi * x) - 2 * x ** 2 + 4 + np.tan(x) 

Function_Collection[3] = lambda x: 1 * np.exp(- (x - 1) * (x - 2) / (2 * 0.5 * 0.5))
Function_Collection[4] = lambda x: 1 * np.exp(- (x - 0) * (x + 1) / (2 * 0.5 * 0.5))

B_const = 7
Function_Collection[5] = lambda x: np.cos(x * (2 ** B_const)) * np.cos(x * (2 ** B_const) / (4 * np.sqrt(5))) * np.exp(- x * x) + 2 * np.exp(-x)
Function_Collection[6] = lambda x: np.sin(x * (2 ** B_const)) / (2 * np.sqrt(10)) * np.exp(- x * x * x) - 4 * np.exp(-np.sqrt(x))