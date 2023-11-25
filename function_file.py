import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
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


def something_with_widgets(rho):
    velocity = np.linspace(0, 100, 100)
    D = 0.5 * rho * 1.5 * 0.8 * velocity**2
    power_required = D * velocity / 1000

    plt.figure(figsize=(10, 5))

    # Plot drag
    plt.subplot(1, 2, 1)
    sns.lineplot(x=velocity, y=D, label='Drag', color='red')
    # sns.lineplot(x=velocity_mtow_sl, y=drag_mtow_sl, label='Drag for MTOW at sea level', color='blue', linestyle='--')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Drag [kN]')

    # Plot power required
    plt.subplot(1, 2, 2)
    sns.lineplot(x=velocity, y=power_required, label='Power required', color='red')
    # sns.lineplot(x=velocity_mtow_sl, y=power_required_mtow_sl, label='Power required for MTOW at sea level', color='blue', linestyle='--')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Power [kW]')

    plt.tight_layout()
    plt.show()


