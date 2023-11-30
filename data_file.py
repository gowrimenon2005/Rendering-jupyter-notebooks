import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Adding the dataset
Aircraft_name = ['ASG 29E', 'ATR 72-600', 'Fokker 100', 'Gulfstream G650', 'Boeing 737-800',
                 'A330-300', 'MD-11', 'Boeing 777-200', 'Boeing 747-400', 'A380-800']
Wing_surface_area = [10.5, 61.0, 93.5, 119.2, 125.0, 361.6, 338.9, 427.8, 541.2, 845]  # area given in square meters
Wing_span = [18.0, 27.05, 28.08, 30.35, 35.79, 60.30, 51.96, 60.93, 64.92, 79.80]  # span given in meters
Wing_aspect_ratio = [30.9, 12.0, 8.43, 6.8, 9.4, 10.1, 7.49, 8.7, 7.7, 7.5]  # aspect ratio is unitless
Overall_length = [6.59, 27.17, 35.53, 30.40, 39.47, 63.68, 61.73, 63.73, 70.67, 72.80]  # length given in meters
Overall_height = [1.3, 7.65, 8.51, 7.82, 12.55, 16.83, 17.65, 18.51, 18.77, 24.10]  # height given in meters
Fuselage_width = [0.66, 2.87, 3.10, 2.59, 3.76, 5.64, 6.01, 6.2, 6.5, 7.14]  # width given in meters
MTOW = [600, 22800, 44560, 41145, 70535, 2300, 273315, 247210, 362875, 560000]  # MTOW given in kilograms
OEW = [325, 13311, 24727, 24492, 42493, 124500, 130650, 139025, 180480, 270280]  # OEW given in kilograms
Max_fuel = [10.5, 5000, 10745, 20049, 21000, 78025, 117356, 94210, 162575, 259465]  # fuel given in kilograms
Max_payload = [202, 7500, 12013, 2948, 20240, 48500, 52632, 55670, 63917, 90720]  # payload given in kilograms
Range = ['nan', 1537, 2870, 12964, 3685, 10834, 12632, 9389, 11454, 14812]  # range given in kilometers
TO_field_length = ['nan', 1333, 1720, 1786, 2101, 2515, 3116, 2576, 2820, 3030]  # length given in meters
LND_field_length = ['nan', 915, 1350, 925, 1646, 1753, 2119, 1474, 1905, 2104]  # length given in meters
Service_ceiling = ['nan', 250, 350, 510, 410, 430, 430, 430, 450, 430]  # height given in Flight Levels
Cruise_speed = [0.085, 0.412, 0.597, 0.85, 0.785, 0.82, 0.82, 0.84, 0.85, 0.82]  # speed in Mach number
Max_operating_speed = [0.219, 0.55, 0.77, 0.925, 0.82, 0.86, 0.945, 0.89, 0.92, 0.89]  # speed in Mach number 
Approach_speed = ['nan', 113, 130, 120, 141, 137, 168, 136, 146, 141]  # speed given in knots 
No_of_engines = [1, 2, 2, 2, 2, 2, 3, 2, 4, 4]  # number of engines is unitless
Thrust = ['nan', 'nan', 67200, 75200, 117000, 287000, 274000, 343000, 263000, 332000]  # thrust given in Newtons

# Creating the dataframe
df = pd.DataFrame({'Aircraft_name': Aircraft_name, 'Wing_surface_area': Wing_surface_area,
                   'Wing_span': Wing_span, 'Wing_aspect_ratio': Wing_aspect_ratio,
                   'Overall_length': Overall_length, 'Overall_height': Overall_height,
                   'Fuselage_width': Fuselage_width, 'MTOW': MTOW, 'OEW': OEW, 'Max_fuel': Max_fuel,
                   'Max_payload': Max_payload, 'Range': Range, 'TO_field_length': TO_field_length,
                   'LND_field_length': LND_field_length, 'Service_ceiling': Service_ceiling,
                   'Cruise_speed': Cruise_speed, 'Max_operating_speed': Max_operating_speed,
                   'Approach_speed': Approach_speed, 'No_of_engines': No_of_engines, 'Thrust': Thrust})

# saving the dataframe to a csv file
df.to_csv('Aircraft_data.csv', index=False)

def get_aircraft_polar(aircraft_name, cl_data):
    if aircraft_name == 'ASG 29E':
        cd = 0.0087 + 0.0114 * cl_data ** 2
    elif aircraft_name == 'ATR 72-600':
        cd = 0.0317 + 0.0350 * cl_data ** 2
    elif aircraft_name == 'Fokker 100':
        cd = 0.0194 + 0.0437 * cl_data ** 2
    elif aircraft_name == 'Gulfstream G650':
        cd = 0.0153 + 0.0465 * cl_data ** 2
    elif aircraft_name == 'Boeing 737-800':
        cd = 0.0190 + 0.0382 * cl_data ** 2
    elif aircraft_name == 'A330-300':
        cd = 0.0138 + 0.0368 * cl_data ** 2
    elif aircraft_name == 'MD-11':
        cd = 0.0164 + 0.0430 * cl_data ** 2
    elif aircraft_name == 'Boeing 777-200':
        cd = 0.0133 + 0.0425 * cl_data ** 2
    elif aircraft_name == 'Boeing 747-400':
        cd = 0.0126 + 0.0518 * cl_data ** 2
    elif aircraft_name == 'A380-800':
        cd = 0.0133 + 0.0472 * cl_data ** 2
    else: 
        print('Aircraft name not found')
    return cd

def get_supersonic_data(mach):
    dat = pd.read_csv(f'./supersonic_data/cl_cd_mach_{mach}..csv')
    cl = dat['cl']
    cd = dat['cd']
    # quadratic fit
    p = np.polyfit(cl, cd, 2)
    return p

    