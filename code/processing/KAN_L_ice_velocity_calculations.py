#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to calculate KAN_L PROMICE ice velocities at different periods

Saves ice velocities to CSV file   GFSTS_base conda env
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import dates
import matplotlib as mpl
import math
import sys
from datetime import datetime, timedelta

# Need to download this from GitHub: https://github.com/nsidc/polarstereo-lonlat-convert-py.git
# 1) go to URL, click on "code", "download zip"
# 2) move downloaded folder to sensible location, and unzip
# 3) openup powershell on PC, navigate to where folder is
# 4) type: " pip install --editable polarstereo-lonlat-convert-py-main"
from polar_convert.constants import NORTH
from polar_convert import polar_lonlat_to_xy

from scipy.signal import wiener, filtfilt, butter, gaussian, freqz
from scipy.ndimage import filters

def ssqe(sm, s, npts):
	return np.sqrt(np.sum(np.power(s-sm,2)))/npts

def testGauss(x,y, npts):
    b = gaussian(npts, 10)
    ga = filters.convolve1d(y, b/b.sum())
    # plt.scatter(x, y, c='b', s=5, alpha=0.8)
    # plt.scatter(x, ga, c='r', s=5)


    return ga

#---------------MATPLOTLIB USER OPTIONS---------
tick_fontsize = 15
label_fontsize = 18
title_fontsize = 20
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.6
plt.rc('xtick', labelsize=tick_fontsize)
plt.rc('ytick', labelsize=tick_fontsize)
plt.rc('axes', labelsize=label_fontsize)
plt.rc('axes', titlesize=title_fontsize)
plt.rc('legend', fontsize=label_fontsize)    # legend fontsize
plt.rc('figure', titlesize=title_fontsize)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'None'

width = 6.88 
height = width/1.618 
#-------------------------------------------------
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

#---Function to calculate ice velocities (scroll past this to use function)------------
def calculate_ice_velocity(MET_DATA, start_date, end_date, save_csv, save_fig, plot_s1, name):

    MET_DATA['gps_lat'].replace('', np.nan, inplace=True)
    MET_DATA['gps_lon'].replace('', np.nan, inplace=True)
       
    #--------MASK TIME TO DURATION OF FIELDWORK-------------------
    print('\nSelecting data from', start_date, 'to', end_date, '\n')
    mask = (MET_DATA['time'] >= start_date) & (MET_DATA['time'] <= end_date)
    MET_DATA = MET_DATA.loc[mask]

    time_temp = MET_DATA['time']

    # CLEANING DATA UP
    MET_DATA = MET_DATA[MET_DATA['gps_hdop'] !=1] #Remove timestamps with a high horizontal dilution
    MET_DATA = MET_DATA[MET_DATA['gps_time'] !=-999.0] # remove timestamps with no GPS measurements
    MET_DATA = MET_DATA[MET_DATA['gps_time'] !=1]
    MET_DATA = MET_DATA.dropna(subset=['gps_lat'])

    lat = MET_DATA['gps_lat'].values
    lon = MET_DATA['gps_lon'].values
    time = pd.to_datetime(MET_DATA['time'].values)

    lon = testGauss(lat, lon, 200) # gaussian filter
    lat = testGauss(lon, lat, 200)
    print('\nCalculating ice velocity (m/day)')
    
    x = np.zeros(len(lat))
    y = np.zeros(len(lat))       
    #d = np.zeros(len(lat))
    ice_velocity = []#np.zeros(len(lat))
    velocity_error = []
    time_between_array = []
                                
    true_scale_lat = 70  # true-scale latitude in degrees
    re = 6378.137  # earth radius in km
    e = 0.01671 # earth eccentricity
    hemisphere = NORTH

    for i in range(0,len(lat)):
        x[i], y[i] = polar_lonlat_to_xy(lon[i], lat[i], true_scale_lat, re, e, hemisphere) #convert to projected coordinates
  
    for i in range(0,len(time)):
        print('\n')
        print(i, '/', len(time))
        time_wk = time[i] + timedelta(days=3) # weeks=1
        nearest1 = nearest(time, time_wk) # finds nearest time x days in advance
        jump = np.where(time == nearest1)

        x_diff = (x[jump] - x[i]) * 1000 #convert km to metres
        y_diff = (y[jump] - y[i]) * 1000 #convert km to metres
        d = math.sqrt( ((x_diff)**2) + ((y_diff)**2) )

        time_diff = (time[jump] - time[i])
        time_diff = time_diff.total_seconds().values
        time_diff = time_diff.astype(float)

        velocity_ms = d / time_diff #velocity in m/s
        ice_velocity_1 = velocity_ms * 60 * 60 * 24 * 365 #velocity in metres per year
        ice_velocity_1 = ice_velocity_1[0]
        ice_velocity.append(ice_velocity_1)

        if d==0:
            d=1
            vel_error_1 = ice_velocity_1 * math.sqrt( (2.5/d)**2 ) # in metres per year
            velocity_error.append(vel_error_1)
        else:
            vel_error_1 = ice_velocity_1 * math.sqrt( (2.5/d)**2 ) # in metres per year
            velocity_error.append(vel_error_1)
        
        #-----calculating time between 2 timestamps-----------
        time1 = time[i]
        time2 = time[jump]    
        time_between = time1 + (time2 - time1)/2
        time_between = pd.to_datetime(time_between).strftime('%Y-%m-%d %H:%M').values[0] # needed to get rid of square brackets and array
        time_between_array.append(time_between)
        print(ice_velocity_1)
        print(vel_error_1)

        # if i==10:
        #     sys.exit()
            
    print(len(ice_velocity))

    
    if save_csv=='True':       
        velo_name = 'ice_velocity'  
        data = {'time':time_between_array, velo_name: ice_velocity, 'velocity_error': velocity_error}
        dataframe = pd.DataFrame(data=data)

        dataframe['time'] = dataframe['time'].str.strip("[']")
        dataframe['time'] = pd.to_datetime(dataframe['time'].values)
        dataframe = dataframe.set_index('time')

        dataframe = dataframe.resample('3D').mean() # resampled to 3 days

        firstdate = pd.to_datetime(start_date).strftime('%d%m%y')
        lastdate = pd.to_datetime(end_date).strftime('%d%m%y')

        date_file = '%s-%s' % (firstdate, lastdate)
        outfile = data_dir + '\velocity\GPS\%s_ice_velocity_%s_3day_err_test_resampled.csv' % (name, date_file)
        dataframe.to_csv(outfile)
        print(dataframe)
        print('\nIce Velocity CSV saved to:', outfile)



#-----------------------------------------------------------------------------------------

# =============================================================================
# IMPORT DATA
# =============================================================================
data_dir = # data directory in github repo
met_filename = data_dir + '\met_data\KAN_L_hour.csv' #file directory to edited KAN_L file
MET_DATA = pd.read_csv(met_filename)
MET_DATA['time'] = pd.to_datetime(MET_DATA['time'].values)

# =============================================================================
#  USER OPTIONS
# =============================================================================
start_date = '2022-04-01 00:00:00' #date range wanted in local kangerluusauq time
end_date = '2022-10-28 00:00:00'

save_csv = 'True'
save_fig = 'False'
plot_s1 = 'False'

calculate_ice_velocity(MET_DATA, start_date, end_date, save_csv, save_fig, plot_s1, name='KANL')













