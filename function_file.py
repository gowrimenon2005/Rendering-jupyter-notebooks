import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# define some random functions 


def plot(name):
    if name == 'sin':
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        sns.set_style("whitegrid")
        plt.plot(x, y)
        plt.show()
    elif name == 'cos':
        x = np.linspace(0, 2*np.pi, 100)
        y = np.cos(x)
        sns.set_style("whitegrid")
        plt.plot(x, y)
        plt.show()
    else:
        print('Error: no such function')
        return None


def calculate_2x(x):
    return 2*x

