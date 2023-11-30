import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import *
from data_file import *
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy 

current_fig = None

#######################################################################################################################	
# preliminaries 

def get_aircraft_data(aircraft_name):
    aircraft_data = pd.read_csv('Aircraft_data.csv')
    surface_area = aircraft_data.loc[aircraft_data['Aircraft_name'] == aircraft_name]['Wing_surface_area'].iloc[0]
    weight = aircraft_data.loc[aircraft_data['Aircraft_name'] == aircraft_name]['MTOW'].iloc[0]
    aspect_ratio = aircraft_data.loc[aircraft_data['Aircraft_name'] == aircraft_name]['Wing_aspect_ratio'].iloc[0]
    max_thrust = aircraft_data.loc[aircraft_data['Aircraft_name'] == aircraft_name]['Thrust'].iloc[0]
    return surface_area, weight, aspect_ratio, max_thrust

def plot_drag_power(surface_area, weight, aspect_ratio):
    C_D0 = 0.02
    e = 0.9
    rho = 0.985
    test_velocities = np.linspace(10, 400, 100)

    # calculate drag and power required
    D = (C_D0 * 0.5 * rho * test_velocities**2  * surface_area + 2 * weight**2 / 
         (np.pi * aspect_ratio * e * rho * test_velocities**2 * surface_area))/1000
    power_required = D * test_velocities

    # plot drag and thrust
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(test_velocities, D, label='Drag', color='red')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Drag [kN]')

    # plot power available and power required
    plt.subplot(1, 2, 2)
    plt.plot(test_velocities, power_required, label='Power required', color='red')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Power [kW]')

    plt.tight_layout()
    plt.show()

def plot_drag_power_2(rho, weight, name, weight_max, surface_area1):
    # Aircraft parameters
    aircraft_name = name
    surface_area = surface_area1

    # Calculate drag and power required   
    CL = np.linspace(0.2, 0.8, 100)
    CD = get_aircraft_polar(aircraft_name, CL)
    velocity = np.sqrt(2 * weight / (rho * CL * surface_area))
    D = CD/CL * weight  # Convert to kN
    power_required = D * velocity
    velocity_mtow_sl = np.sqrt(2 * weight_max / (1.225 * CL * surface_area))
    drag_mtow_sl = CD/CL * weight_max  # Convert to kN
    power_required_mtow_sl = drag_mtow_sl * velocity_mtow_sl

    plt.figure(figsize=(10, 5))

    # Plot drag
    plt.subplot(1, 2, 1)
    sns.lineplot(x=velocity, y=D, label='Drag', color='red')
    sns.lineplot(x=velocity_mtow_sl, y=drag_mtow_sl, label='Drag for MTOW at sea level', color='blue', linestyle='--')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Drag [kN]')

    # Plot power required
    plt.subplot(1, 2, 2)
    sns.lineplot(x=velocity, y=power_required, label='Power required', color='red')
    sns.lineplot(x=velocity_mtow_sl, y=power_required_mtow_sl, label='Power required for MTOW at sea level', color='blue', linestyle='--')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Power [kW]')

    plt.tight_layout()
    plt.show()
#######################################################################################################################
# lecture 1 functions

def td_aoc_curve(name, rho):
    sns.set_style('whitegrid')
    C_D0 = 0.02
    e = 0.9
    S, W, A, T_max = get_aircraft_data(name)
    W = W * 9.81
    velocities = np.linspace(50, 400, 351)
    velocities2 = np.linspace(1, 400, 400)
    D = (C_D0 * 0.5 * rho * velocities**2 * S + 2 * W**2 / (np.pi * A * e * rho * velocities**2 * S))
    T_plot = T_max * (1-0.02*velocities2**0.5)
    T = T_max * (1-0.02*velocities**0.5)
    AOC = (T - D)/W

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(velocities, D/1000, label='Thrust required', color='red')
    plt.plot(velocities2, T_plot/1000, label='Thrust available', color='blue')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Thrust [kN]')

    plt.subplot(1, 2, 2)
    plt.plot(velocities, AOC, label='AOC', color='red')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('AOC')

    plt.show()

def papr_roc_curve(name, rho):
    global current_fig
    if current_fig is not None:
        plt.close(current_fig)
        
    sns.set_style('whitegrid')
    C_D0 = 0.02
    e = 0.9
    S, W, A, T_max = get_aircraft_data(name)
    W = W * 9.81
    velocities = np.linspace(50, 400, 351)
    velocities2 = np.linspace(1, 400, 400)
    DV = (C_D0 * 0.5 * rho * velocities**2 * S + 2 * W**2 / (np.pi * A * e * rho * velocities**2 * S)) * velocities
    Pr_plot = T_max * (1-0.02*velocities2**0.5) * velocities2
    PR = T_max * (1-0.02*velocities**0.5) * velocities
    ROC = (PR - DV)/W

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(velocities, DV, label='Power required', color='red')
    plt.plot(velocities2, Pr_plot, label='Power available', color='blue')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Power [W]')

    plt.subplot(1, 2, 2)
    plt.plot(velocities, ROC, label='ROC', color='red')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('ROC [m/s]')
    plt.ylim(0, max(ROC)+2)
    # plt.xlim(0, 250)

    plt.show()

def visualise_altitude_effects_rc(name, rhol, rhou):
    sns.set_style('whitegrid')
    plt.clf()
    C_D0 = 0.02
    e = 0.9
    S, W, A, T_max = get_aircraft_data(name)
    W = W * 9.81
    velocities = np.linspace(50, 250, 201)
    velocities2 = np.linspace(1, 250, 350)
    DVl = (C_D0 * 0.5 * rhol * velocities**2 * S + 2 * W**2 / (np.pi * A * e * rhol * velocities**2 * S)) * velocities
    DVu = (C_D0 * 0.5 * rhou * velocities**2 * S + 2 * W**2 / (np.pi * A * e * rhou * velocities**2 * S)) * velocities
    
    Pr_plotl = T_max * (1-0.02*velocities2**0.5) * velocities2 * (rhol/1.225)**0.85
    Pr_plotu = T_max * (1-0.02*velocities2**0.5) * velocities2 * (rhou/1.225)**0.85
    PRl = T_max * (1-0.02*velocities**0.5) * velocities * (rhol/1.225)**0.85
    PRu = T_max * (1-0.02*velocities**0.5) * velocities * (rhou/1.225)**0.85
    
    ROCl = (PRl - DVl)/W
    ROCu = (PRu - DVu)/W
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(velocities, DVl, label=f'Power required at density = {rhol}', color='red', linestyle='--')
    plt.plot(velocities, DVu, label=f'Power required at density = {rhou}', color='red')
    plt.plot(velocities2, Pr_plotl, label=f'Power available at density = {rhol}', color='blue', linestyle='--')
    plt.plot(velocities2, Pr_plotu, label=f'Power available at density = {rhou}', color='blue')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Power [W]')
    plt.xlim(0, 250)

    plt.subplot(1, 2, 2)
    plt.plot(velocities, ROCl, label=f'ROC at density = {rhol}', color='green', linestyle='--')
    plt.plot(velocities, ROCu, label=f'ROC at density = {rhou}', color='green')
    plt.legend()
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('ROC [m/s]')
    plt.ylim(0, max(ROCu)+2)
    plt.xlim(0, 250)

    plt.tight_layout()
    plt.show()
    
    plt.show()

def altitude_effects_rc_v2(name):
    sns.set_style('whitegrid')
    C_D0 = 0.04
    e = 0.9
    S, W, A, T_max = get_aircraft_data(name)
    W = W * 9.81
    # W = 445600
    # S = 94
    # A = 30.65
    # T_max = 67200
    altitude_un = np.linspace(0, 10500, 200)
    velocity = np.linspace(10, 180, 200)
    altitude, velocity = np.meshgrid(altitude_un, velocity)

    rho = 1.225 * (1 - 0.0065 * altitude / 288.15) ** (9.81 / (0.0065 * 287))
    dv = (C_D0 * 0.5 * 1.225 * velocity**2 * S + 2 * W**2 
          / (np.pi * A * e * 1.225 * velocity**2 * S)) * velocity
    # pa = T_max * (1-0.02*velocity**0.5) * velocity * (rho/1.225)**0.85
    pa = T_max * velocity * (rho/1.225)**0.85
    roc = (pa - dv)/W
    # print(roc)
    roc1 = np.where(roc < 0, np.nan, roc)
    roc2 = np.where(roc < 0, 0, roc)
    # for i in range(roc.shape[0]):
    #     for j in range(roc.shape[1]):
    #         if roc[i, j] > 0:
    #             print(f'something works {i} and {j} and {roc[i, j]}')
    # print(roc.max(axis=1))

    # # subplot 1 
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    contour_plot = plt.contourf(velocity, altitude, roc1, levels=20, cmap='RdBu', vmin=0)
    contour_lines = plt.contour(velocity, altitude, roc1, levels=20, colors='k', linestyles='dashed', vmin=0)

    # # Add a colorbar
    plt.colorbar(contour_plot, label='ROC', extend='both')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Altitude [m]')

    # # subplot 2
    # print(roc.max(axis=0))
    plt.subplot(1, 2, 2)
    plt.plot(roc2.max(axis=0), altitude_un)
    plt.xlabel('ROC [m/s]')
    plt.ylabel('Altitude [m]')
    plt.show()
    return 

def flight_envelope(name):
    sns.set_style('whitegrid')
    C_D0 = 0.04
    e = 0.9
    S, W, A, T_max = get_aircraft_data(name)
    return 


#######################################################################################################################
# lecture 2 functions

def energy_height_curve():
    sns.set_style('whitegrid')
    altitude = np.linspace(0, 10000, 100)
    velocity = np.linspace(50, 250, 201)
    altitude, velocity = np.meshgrid(altitude, velocity)
    energy_height = 0.5 * velocity**2/(2*9.81) + altitude
    contour_plot = plt.contourf(velocity, altitude, energy_height, levels=20, cmap='RdBu')
    contour_lines = plt.contour(velocity, altitude, energy_height, levels=20, colors='k', linestyles='dashed')

    # Add a colorbar
    plt.colorbar(contour_plot, label='Energy Height')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Altitude [m]')
    plt.show()

def energy_height_roc(name):
    sns.set_style('whitegrid')
    C_D0 = 0.04
    e = 0.9
    S, W, A, T_max = get_aircraft_data(name)
    W = W * 9.81
    altitude = np.linspace(0, 10500, 200)
    velocity = np.linspace(10, 180, 200)

    en_vel = np.linspace(10, 180, 400)
    en_altitude = np.linspace(0, 10500, 400)

    altitude, velocity = np.meshgrid(altitude, velocity)
    en_altitude, en_vel = np.meshgrid(en_altitude, velocity)

    rho = 1.225 * (1 - 0.0065 * altitude / 288.15) ** (9.81 / (0.0065 * 287))
    dv = (C_D0 * 0.5 * 1.225 * velocity**2 * S + 2 * W**2 
          / (np.pi * A * e * 1.225 * velocity**2 * S)) * velocity
    pr = T_max * (1-0.02*velocity**0.5) * velocity * (rho/1.225)**0.85
    roc = (pr - dv)/W

    if roc.min() < 0:
        roc[roc < 0] = np.nan

    contour_plot = plt.contourf(velocity, altitude, roc, levels=20, cmap='RdBu', vmin=0)
    contour_lines = plt.contour(velocity, altitude, roc, levels=20, colors='k', linestyles='dotted', vmin=0)

    energy_height = 0.5 * en_vel**2/(2*9.81) + en_altitude
    energy_lines = plt.contour(en_vel, en_altitude, energy_height, levels=20, colors='g', linestyles='solid', linewidths=1)

    # Add a colorbar
    plt.colorbar(contour_plot, label='ROC', ticks=np.arange(0, 21, 2), extend='both')
    plt.xlabel('Velocity [m/s]')
    plt.ylabel('Altitude [m]')
    plt.show()

#######################################################################################################################
# lecture 3 functions

def bank_angle():
    sns.set_style('whitegrid')
    bank_angle = np.linspace(0, 89, 100)
    load_factor = 1/np.cos(bank_angle*np.pi/180)
    plt.plot(bank_angle, load_factor, 'blue')  
    plt.xlabel('Bank Angle (deg)')
    plt.ylabel('Load Factor')
    plt.show()
    return 

def turning_flight_v2(name):
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    # classical ex: 
    # W = 4.27e3  # N
    # S = 3.51  # m^2
    # A = 7.65
    # e = 0.67
    # CD0 = 0.02
    # k = 1 / (np.pi * A * e)

    # # Assuming
    # CLmax = 1.6
    # Tmax = 1e3  # N

    S, W, A, Tmax = get_aircraft_data(name)
    W = W * 9.81
    e = 0.67
    CD0 = 0.02
    k = 1 / (np.pi * A * e)

    # Assuming
    CLmax = 1.6
    # Tmax = 2e3  # N

    # Sweeps
    n = np.arange(1, 8.01, 0.01)
    V = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vs = np.full_like(n, np.nan)
    D = np.full((len(n), len(V)), np.nan)
    nmax = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vs[i] = np.sqrt((2 / 1.225) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / 1.225) * (n[i] * W / S) * (1 / V[j]**2)
                D[i, j] = 0.5 * 1.225 * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(D[i, j]) and D[i, j] <= Tmax:
                    nmax[j] = n[i]

    # Notable values
    V_star = np.sqrt((2 / 1.225) * (Tmax / S) * (1 / (CD0 + k * CLmax**2)))
    n_star = (Tmax / W) * (CLmax / (CD0 + k * CLmax**2))

    largest_zero_indices = np.zeros(D.shape[0], dtype=int)

    for i in range(D.shape[0]):
        zero_indices = np.where(D[i, :] == 0)[0]
        if zero_indices.size > 0:
            largest_zero_indices[i] = zero_indices.max()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for i in range(0, len(V), 100):
        plt.plot(V[largest_zero_indices[i]+1:], D[i, largest_zero_indices[i]+1:], label='n = {:.3}'.format(n[i]))
    plt.plot(Vs, 0.5 * 1.225 * Vs**2 * S * (CD0 + k * CLmax**2), 'k--')
    plt.plot(V, np.ones_like(V) * Tmax, 'r-.')
    plt.ylabel('D (N)')
    plt.xlabel('V (m/s)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(V, nmax, linewidth=2)
    plt.plot(V[V <= 1.2 * V_star], 0.5 * 1.225 * V[V <= 1.2 * V_star]**2 * S * CLmax / W, 'r--')
    plt.plot(V_star, n_star, 'bo', markersize=5, markerfacecolor='b')
    plt.xlabel('V (m/s)')

    plt.show()

def tightest_turns(name):
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    S, W, A, Tmax = get_aircraft_data(name)
    W = W * 9.81
    e = 0.67
    CD0 = 0.02
    k = 1 / (np.pi * A * e)

    # Assuming
    CLmax = 1.6
    # Tmax = 1e3  # N

    # Sweeps
    n = np.arange(1, 8.01, 0.01)
    V = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vs = np.full_like(n, np.nan)
    D = np.full((len(n), len(V)), np.nan)
    nmax = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vs[i] = np.sqrt((2 / 1.225) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / 1.225) * (n[i] * W / S) * (1 / V[j]**2)
                D[i, j] = 0.5 * 1.225 * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(D[i, j]) and D[i, j] <= Tmax:
                    nmax[j] = n[i]

    # remove np.nan values
    V = V[~np.isnan(nmax)]
    nmax = nmax[~np.isnan(nmax)]
    r_min = V[:-10]**2 / (9.81 * np.sqrt(nmax[:-10]**2 - 1))

    plt.plot(V[:-10], r_min)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Minimum radius required (m)')
    plt.show()

def fastest_turns(name):
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    S, W, A, Tmax = get_aircraft_data(name)
    W = W * 9.81
    e = 0.67
    CD0 = 0.02
    k = 1 / (np.pi * A * e)

    # Assuming
    CLmax = 1.6
    # Tmax = 1e3  # N

    # Sweeps
    n = np.arange(1, 8.01, 0.01)
    V = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vs = np.full_like(n, np.nan)
    D = np.full((len(n), len(V)), np.nan)
    nmax = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vs[i] = np.sqrt((2 / 1.225) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / 1.225) * (n[i] * W / S) * (1 / V[j]**2)
                D[i, j] = 0.5 * 1.225 * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(D[i, j]) and D[i, j] <= Tmax:
                    nmax[j] = n[i]

    # remove np.nan values
    V = V[~np.isnan(nmax)]
    nmax = nmax[~np.isnan(nmax)]
    r_min = V[:-10]**2 / (9.81 * np.sqrt(nmax[:-10]**2 - 1))
    t_min = r_min * 2 * np.pi / V[:-10]

    plt.plot(V[:-10], t_min)
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Minimum time required for full revolution (s)')
    plt.show()

def comparison_turns():
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    W = 4.27e3  # N
    S = 3.51  # m^2
    A = 7.65
    e = 0.67
    CD0 = 0.02
    k = 1 / (np.pi * A * e)

    # Assuming
    CLmax = 1.6
    Tmax = 1e3  # N

    # Sweeps
    n = np.arange(1, 8.01, 0.01)
    V = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vs = np.full_like(n, np.nan)
    D = np.full((len(n), len(V)), np.nan)
    nmax = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vs[i] = np.sqrt((2 / 1.225) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / 1.225) * (n[i] * W / S) * (1 / V[j]**2)
                D[i, j] = 0.5 * 1.225 * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(D[i, j]) and D[i, j] <= Tmax:
                    nmax[j] = n[i]

    # np.nan values = 0
    V = np.where(np.isnan(nmax), 1, V)
    nmax = np.where(np.isnan(nmax), 1, nmax)
    # V = V[~np.isnan(nmax)]
    # nmax = nmax[~np.isnan(nmax)]
    r_min = V[:]**2 / (9.81 * np.sqrt(nmax[:]**2 - 1))
    t_min = r_min * 2 * np.pi / V[:]

    # notable values 
    V_rmin = V[np.argmin(r_min)]
    V_tmin = V[np.argmin(t_min)]
    V_nmax = V[np.argmax(nmax)]
    R_tmin = V_tmin**2 / (9.81 * np.sqrt(nmax[np.argmin(t_min)]**2 - 1))
    R_nmax = V_nmax**2 / (9.81 * np.sqrt(nmax[np.argmax(nmax)]**2 - 1))
    t_nmax = R_nmax * 2 * np.pi / V_nmax

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(V[:-50], r_min[:-50])
    plt.plot(V_rmin, np.min(r_min), 'bo', markersize=5, markerfacecolor='b', label='Minimum radius')
    plt.plot(V_tmin, r_min[np.argmin(t_min)], 'ro', markersize=5, markerfacecolor='r', label='Minimum time')
    plt.plot(V[:-50], R_tmin/V_tmin * V[:-50], 'r', linestyle='--', dashes=(5, 5))
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Minimum radius required (m)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(V[:-50], t_min[:-50])
    plt.plot(V_tmin, np.min(t_min), 'ro', markersize=5, markerfacecolor='r', label='Minimum time')
    plt.plot(V_nmax, t_min[np.argmax(nmax)], 'go', markersize=5, markerfacecolor='g', label='Maximum load factor')
    plt.plot(V[:-50], t_nmax/V_nmax * V[:-50], 'b', linestyle='--', dashes=(4, 5))
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Minimum time required for full revolution (s)')
    plt.legend()

    plt.show()

def turning_flight_alt(name, rho):
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    # classical ex: 
    # W = 4.27e3  # N
    # S = 3.51  # m^2
    # A = 7.65
    # e = 0.67
    # CD0 = 0.02
    # k = 1 / (np.pi * A * e)

    # # Assuming
    # CLmax = 1.6
    # Tmax = 1e3  # N
    
    S, W, A, Tmax = get_aircraft_data(name)
    W = W * 9.81
    e = 0.67
    CD0 = 0.02
    k = 1 / (np.pi * A * e)

    # Assuming
    CLmax = 1.6
    # Tmax = 2e3  # N

    # Sweeps
    n = np.arange(1, 8.01, 0.01)
    V = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vs = np.full_like(n, np.nan)
    D = np.full((len(n), len(V)), np.nan)
    nmax = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vs[i] = np.sqrt((2 / rho) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / rho) * (n[i] * W / S) * (1 / V[j]**2)
                D[i, j] = 0.5 * rho * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(D[i, j]) and D[i, j] <= Tmax:
                    nmax[j] = n[i]

    # Notable values
    V_star = np.sqrt((2 / rho) * (Tmax / S) * (1 / (CD0 + k * CLmax**2)))
    n_star = (Tmax / W) * (CLmax / (CD0 + k * CLmax**2))

    largest_zero_indices = np.zeros(D.shape[0], dtype=int)

    for i in range(D.shape[0]):
        zero_indices = np.where(D[i, :] == 0)[0]
        if zero_indices.size > 0:
            largest_zero_indices[i] = zero_indices.max()

    # seal level stuff just for clarity 
    Sl, Wl, Al, Tmaxl = get_aircraft_data(name)
    el = 0.67
    CD0l = 0.02
    kl = 1 / (np.pi * A * e)

    # Assuming
    CLmaxl = 1.6
    # Tmax = 2e3  # N

    # Sweeps
    nl = np.arange(1, 8.01, 0.01)
    Vl = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vsl = np.full_like(nl, np.nan)
    Dl = np.full((len(nl), len(Vl)), np.nan)
    nmaxl = np.full_like(Vl, np.nan)

    for i in range(len(nl)):
        for j in range(len(Vl)):
            Vsl[i] = np.sqrt((2 / 1.225) * (nl[i] * Wl / Sl) * (1 / CLmaxl))
            
            if Vl[j] >= Vs[i]:
                CL = (2 / 1.225) * (nl[i] * Wl / Sl) * (1 / Vl[j]**2)
                Dl[i, j] = 0.5 * 1.225 * Vl[j]**2 * Sl * (CD0l + kl * CL**2)

                if not np.isnan(Dl[i, j]) and Dl[i, j] <= Tmax:
                    nmaxl[j] = nl[i]

    # Notable values
    V_starl = np.sqrt((2 / 1.225) * (Tmaxl / Sl) * (1 / (CD0l + kl * CLmaxl**2)))
    n_starl = (Tmaxl / Wl) * (CLmaxl / (CD0l + kl * CLmaxl**2))

    largest_zero_indicesl = np.zeros(Dl.shape[0], dtype=int)

    for i in range(Dl.shape[0]):
        zero_indicesl = np.where(Dl[i, :] == 0)[0]
        if zero_indicesl.size > 0:
            largest_zero_indicesl[i] = zero_indicesl.max()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    for i in range(0, len(V), 100):
        plt.plot(V[largest_zero_indices[i]+1:], D[i, largest_zero_indices[i]+1:], label='n = {:.3}'.format(n[i]))
    plt.plot(Vs, 0.5 * rho * Vs**2 * S * (CD0 + k * CLmax**2), 'k--')
    plt.plot(V, np.ones_like(V) * Tmax, 'r-.')
    plt.ylabel('D (N)')
    plt.xlabel('V (m/s)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(V, nmax, linewidth=2, label='At chosen air density')
    plt.plot(V[V <= 1.2 * V_star], 0.5 * rho * V[V <= 1.2 * V_star]**2 * S * CLmax / W, 'r--')
    plt.plot(V_star, n_star, 'bo', markersize=5, markerfacecolor='b')
    plt.plot(Vl, nmaxl, linewidth=2, color='g', label='At sea level')
    plt.plot(Vl[Vl <= 1.2 * V_starl], 0.5 * 1.225 * Vl[Vl <= 1.2 * V_starl]**2 * Sl * CLmaxl / Wl, 'g--')
    plt.legend()
    # plt.plot(V_star, n_star, 'bo', markersize=5, markerfacecolor='b')
    plt.xlabel('V (m/s)')
    plt.ylabel('Maximum load factor')

    plt.show()

def tightest_turns_alt(name, rho):
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    S, W, A, Tmax = get_aircraft_data(name)
    W = W * 9.81
    e = 0.67
    CD0 = 0.02
    k = 1 / (np.pi * A * e)

    # Assuming
    CLmax = 1.6
    # Tmax = 1e3  # N

    # Sweeps
    n = np.arange(1, 8.01, 0.01)
    V = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vs = np.full_like(n, np.nan)
    D = np.full((len(n), len(V)), np.nan)
    nmax = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vs[i] = np.sqrt((2 / rho) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / rho) * (n[i] * W / S) * (1 / V[j]**2)
                D[i, j] = 0.5 * rho * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(D[i, j]) and D[i, j] <= Tmax:
                    nmax[j] = n[i]

    # remove np.nan values
    V = V[~np.isnan(nmax)]
    nmax = nmax[~np.isnan(nmax)]
    r_min = V[:-10]**2 / (9.81 * np.sqrt(nmax[:-10]**2 - 1))

    # seal level stuff just for clarity
    nl = np.arange(1, 8.01, 0.01)
    Vl = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vsl = np.full_like(nl, np.nan)
    Dl = np.full((len(nl), len(Vl)), np.nan)
    nmaxl = np.full_like(Vl, np.nan)

    for i in range(len(nl)):
        for j in range(len(Vl)):
            Vsl[i] = np.sqrt((2 / 1.225) * (nl[i] * W / S) * (1 / CLmax))
            
            if Vl[j] >= Vsl[i]:
                CL = (2 / 1.225) * (n[i] * W / S) * (1 / V[j]**2)
                Dl[i, j] = 0.5 * 1.225 * Vl[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(Dl[i, j]) and [i, j] <= Tmax:
                    nmaxl[j] = nl[i]

    # remove np.nan values
    Vl = Vl[~np.isnan(nmaxl)]
    nmaxl = nmaxl[~np.isnan(nmaxl)]
    r_minl = Vl[:-10]**2 / (9.81 * np.sqrt(nmaxl[:-10]**2 - 1))

    plt.plot(V[:-10], r_min, label='Chosen air density', color='b')
    plt.plot(Vl[:-10], r_minl, label='Sea level', color='g')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Minimum radius required (m)')
    plt.show()

def fastest_turns_alt(name, rho):
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    S, W, A, Tmax = get_aircraft_data(name)
    W = W * 9.81
    e = 0.67
    CD0 = 0.02
    k = 1 / (np.pi * A * e)

    # Assuming
    CLmax = 1.6
    # Tmax = 1e3  # N

    # Sweeps
    n = np.arange(1, 8.01, 0.01)
    V = np.arange(0, 250.5, 0.5)

    # Dependent variables
    Vs = np.full_like(n, np.nan)
    D = np.full((len(n), len(V)), np.nan)
    nmax = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vs[i] = np.sqrt((2 / rho) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / rho) * (n[i] * W / S) * (1 / V[j]**2)
                D[i, j] = 0.5 * rho * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(D[i, j]) and D[i, j] <= Tmax:
                    nmax[j] = n[i]

    # remove np.nan values
    V = V[~np.isnan(nmax)]
    nmax = nmax[~np.isnan(nmax)]
    r_min = V[:-10]**2 / (9.81 * np.sqrt(nmax[:-10]**2 - 1))
    t_min = r_min * 2 * np.pi / V[:-10]

    # sea level stuff just for clarity
    Vsl = np.full_like(n, np.nan)
    Dl = np.full((len(n), len(V)), np.nan)
    nmaxl = np.full_like(V, np.nan)

    for i in range(len(n)):
        for j in range(len(V)):
            Vsl[i] = np.sqrt((2 / 1.225) * (n[i] * W / S) * (1 / CLmax))
            
            if V[j] >= Vs[i]:
                CL = (2 / 1.225) * (n[i] * W / S) * (1 / V[j]**2)
                Dl[i, j] = 0.5 * 1.225 * V[j]**2 * S * (CD0 + k * CL**2)

                if not np.isnan(Dl[i, j]) and Dl[i, j] <= Tmax:
                    nmaxl[j] = n[i]

    # remove np.nan values
    V = V[~np.isnan(nmax)]
    nmaxl = nmaxl[~np.isnan(nmax)]
    r_minl = V[:-10]**2 / (9.81 * np.sqrt(nmaxl[:-10]**2 - 1))
    t_minl = r_minl * 2 * np.pi / V[:-10]

    plt.plot(V[:-10], t_min, label='Chosen air density', color='b')
    plt.plot(V[:-10], t_minl, label='Sea level', color='g')
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Minimum time required for full revolution (s)')
    plt.legend()
    plt.show()

def flight_envelope_turns(name):
    return 
#######################################################################################################################
# lecture 6 functions

def transonic_cl_cd():
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    mach = [0.45, 0.65, 0.79, 0.81, 0.83, 0.85, 0.9, 0.92]
    for i in range(len(mach)):
        p_vals = get_supersonic_data(mach[i])
        cl = np.linspace(0, 0.7, 100)
        cd = p_vals[0] * cl**2 + p_vals[1] * cl + p_vals[2]
        plt.plot(cl, cd, label=f'Mach {mach[i]}')
    plt.xlabel('CL')
    plt.ylabel('CD')
    plt.legend()
    plt.show()

def transonic_cl_clcd():
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    mach = [0.45, 0.65, 0.79, 0.81, 0.83, 0.85, 0.9, 0.92]
    for i in range(len(mach)):
        p_vals = get_supersonic_data(mach[i])
        cl = np.linspace(0, 0.7, 100)
        cd = p_vals[0] * cl**2 + p_vals[1] * cl + p_vals[2]
        plt.plot(cl, cl/cd, label=f'Mach {mach[i]}')
        # plot max cl/cd
        max_clcd = cl[np.argmax(cl/cd)]
        plt.plot(max_clcd, max(cl/cd), 'bo', markersize=5, markerfacecolor='b')
    plt.xlabel('CL')
    plt.ylabel('CL/CD')
    plt.legend()
    plt.show()

def transonic_m_cl():
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    mach = [0.45, 0.65, 0.79, 0.81, 0.83, 0.85, 0.9, 0.92]
    cl_vals = np.linspace(0, 0.7, 100)
    mach_val = np.array(mach)
    cl, mach_vals = np.meshgrid(cl_vals, mach_val)
    contour = np.zeros((len(mach_vals), len(cl_vals)))
    
    for i in range(len(mach)):
        p_vals = get_supersonic_data(mach[i])
        cd = p_vals[0] * cl_vals**2 + p_vals[1] * cl_vals + p_vals[2]
        val = cl_vals / cd * mach[i]
        # print(len(val))
        contour[i, :] = val
    const_cl_m1 = cl_vals * mach_vals**2
    contours = plt.contourf(mach_vals, cl, contour, levels=15, cmap='viridis')
    plt.contour(mach_vals, cl, contour, levels=15, colors='k', linestyles='dashed')
 
    plt.xlabel('Mach')
    plt.ylabel('CL')
    plt.colorbar(contours, label='CL/CD * Mach')
    plt.show()

def transonic_m_cl_alt(weight, rho):
    sns.set_style('whitegrid')
    # Load factor limits: aerodynamic, propulsive, and structural
    mach = [0.45, 0.65, 0.79, 0.81, 0.83, 0.85, 0.9, 0.92]
    cl_vals = np.linspace(0, 0.7, 100)
    mach_val = np.array(mach)
    cl, mach_vals = np.meshgrid(cl_vals, mach_val)
    contour = np.zeros((len(mach_vals), len(cl_vals)))
    
    for i in range(len(mach)):
        p_vals = get_supersonic_data(mach[i])
        cd = p_vals[0] * cl_vals**2 + p_vals[1] * cl_vals + p_vals[2]
        val = cl_vals / cd * mach[i]
        # print(len(val))
        contour[i, :] = val
    
    # trim lines 
    alt = 288.15 / 0.0065 * (1 - (rho/1.225)**(-9.81/(0.0065*287)))
    # print(alt)
    trim_cl = weight / rho * 1/((287 * 1.4 * (288.15-0.0065*alt))) * np.ones_like(mach_val)
    trim_cl = trim_cl / mach_val**2
    contours = plt.contourf(mach_vals, cl, contour, levels=15, cmap='viridis')
    plt.contour(mach_vals, cl, contour, levels=15, colors='k', linestyles='dashed')
    plt.plot(mach_val, trim_cl, 'r--', label='Trim line')
    plt.xlabel('Mach')
    plt.ylabel('CL')
    plt.colorbar(contours, label='CL/CD * Mach')
    plt.show()

