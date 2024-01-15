import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm # to build a LOWESS model
lowess = sm.nonparametric.lowess
import math
import sys
import os

import numpy as np
import matplotlib.dates as mdates
from matplotlib import gridspec
import matplotlib as mpl
import matplotlib
import matplotlib.font_manager as fm
#---------------MATPLOTLIB USER OPTIONS---------
tick_fontsize = 10
label_fontsize = 12
title_fontsize = 16
mpl.rcParams['font.family'] = 'Roboto'
plt.rcParams['axes.linewidth'] = 1.0
plt.rc('xtick', labelsize=tick_fontsize)
plt.rc('ytick', labelsize=tick_fontsize)
plt.rc('axes', labelsize=label_fontsize)
plt.rc('axes', titlesize=title_fontsize)
plt.rc('legend', fontsize=label_fontsize)    # legend fontsize
plt.rc('figure', titlesize=title_fontsize)  # fontsize of the figure title
plt.rcParams['axes.facecolor'] = 'None'

width = 6.88 
height = width/1.618 -1 
#-------------------------------------------------

def load_discharge_data(gate, skip):
    discharge_data = pd.read_csv(data_dir + 'discharge/%s_gate_ice_discharge.csv' % gate, index_col=1)
    discharge_data.index = pd.DatetimeIndex(discharge_data.index, dayfirst=True) # convert to datetime object
    # print(discharge_data.index)

    discharge_data = discharge_data.drop(discharge_data.columns[0],axis=1) # dropping first index column

    u_err = discharge_data['u_error'].resample('D').interpolate() # resampling to daily resolution (upper error)
    l_err = discharge_data['l_error'].resample('D').interpolate() # lower error

    discharge_data = discharge_data.drop(discharge_data.columns[1],axis=1)
    discharge_data = discharge_data.drop(discharge_data.columns[1],axis=1)

    discharge_data_resamp = discharge_data.resample('D').interpolate() # resample to daily resolution
    #discharge_data_resamp.to_csv('%s_discharge_daily.csv' % gate)

    return discharge_data_resamp, u_err, l_err

### Load in discharge data ############
data_dir = 'C:/Users/s1834371/Documents/GrIS_late-season-melt-event/data/'

skip = 0 
discharge_IS, uerr_IS, lerr_IS = load_discharge_data('IS', skip)
discharge_RUSSELL, uerr_RUSSELL, lerr_RUSSELL = load_discharge_data('RUSSELL', skip)
discharge_NORTH_NUNATAK, uerr_NORTH_NUNATAK, lerr_NORTH_NUNATAK = load_discharge_data('NORTH_NUNATAK', skip)
discharge_SOUTH_NUNATAK, uerr_SOUTH_NUNATAK, lerr_SOUTH_NUNATAK = load_discharge_data('SOUTH_NUNATAK', skip)
discharge_UNNAMED_SOUTH, uerr_UNNAMED_SOUTH, lerr_UNNAMED_SOUTH = load_discharge_data('UNNAMED_SOUTH', skip)

# SUM UP all gates in sector
sum_sector_discharge = discharge_IS + discharge_RUSSELL + discharge_NORTH_NUNATAK + discharge_SOUTH_NUNATAK + discharge_UNNAMED_SOUTH

#sum_sector_discharge.to_csv('summed_discharge.csv')

sum_u_error = uerr_IS.copy() # copy pd dataframe strucutre
sum_l_error = lerr_IS.copy() # copy pd dataframe strucutre
for t in range(len(uerr_IS.index)):
    sum_u_error[t] = math.sqrt(uerr_IS[t]**2+ uerr_RUSSELL[t]**2 + uerr_NORTH_NUNATAK[t]**2 + uerr_SOUTH_NUNATAK[t]**2 + uerr_UNNAMED_SOUTH[t]**2)
    sum_l_error[t] = math.sqrt(lerr_IS[t]**2 + lerr_RUSSELL[t]**2 + lerr_NORTH_NUNATAK[t]**2 + lerr_SOUTH_NUNATAK[t]**2 + lerr_UNNAMED_SOUTH[t]**2)

sum_u_error_22 = sum_u_error[(sum_u_error.index >= '2022-04-01') & (sum_u_error.index <= '2023-04-30')]
sum_l_error_22 = sum_l_error[(sum_l_error.index >= '2022-04-01') & (sum_l_error.index <= '2023-04-30')]

discharge23 = sum_sector_discharge[(sum_sector_discharge.index >= '2023-04-01') & (sum_sector_discharge.index <= '2023-06-30')]
discharge22 = sum_sector_discharge[(sum_sector_discharge.index >= '2022-04-01') & (sum_sector_discharge.index <= '2023-04-30')]
discharge21 = sum_sector_discharge[(sum_sector_discharge.index >= '2021-04-01') & (sum_sector_discharge.index <= '2022-04-30')]
discharge20 = sum_sector_discharge[(sum_sector_discharge.index >= '2020-04-01') & (sum_sector_discharge.index <= '2021-04-30')]
discharge19 = sum_sector_discharge[(sum_sector_discharge.index >= '2019-04-01') & (sum_sector_discharge.index <= '2020-04-30')]
discharge18 = sum_sector_discharge[(sum_sector_discharge.index >= '2018-04-01') & (sum_sector_discharge.index <= '2019-04-30')]
discharge17 = sum_sector_discharge[(sum_sector_discharge.index >= '2017-04-01') & (sum_sector_discharge.index <= '2018-04-30')]
discharge16 = sum_sector_discharge[(sum_sector_discharge.index >= '2016-04-01') & (sum_sector_discharge.index <= '2017-04-30')]

# calculating increase due to melt event
discharge22_df = discharge22[(discharge22.index >= '2022-05-01') & (discharge22.index <= '2023-04-30')]
discharge22_df['upper_err'] = sum_u_error_22
discharge22_df['lower_err'] = sum_l_error_22
discharge22_df['upp_discharge'] = discharge22_df['discharge'] + discharge22_df['upper_err']
discharge22_df['low_discharge'] = discharge22_df['discharge'] - discharge22_df['lower_err']

annual_totals = discharge22_df / 365 # Gigatonnes of ice per day
annual_totals = annual_totals.sum()

annual = annual_totals['discharge']
annual_upper_error = annual_totals['upper_err']
annual_lower_error = annual_totals['lower_err']

#----------------------------------------------
nomelt_discharge = discharge22_df[(discharge22_df.index < '2022-08-20') | (discharge22_df.index > '2022-09-15')] 
nomelt_discharge = nomelt_discharge.resample('D').interpolate()
nomelt_uerr = sum_u_error[(sum_u_error.index < '2022-08-20') | (sum_u_error.index > '2022-09-15')]
nomelt_uerr = nomelt_uerr.resample('D').interpolate()
nomelt_lerr = sum_l_error[(sum_l_error.index < '2022-08-20') | (sum_l_error.index > '2022-09-15')]
nomelt_lerr = nomelt_lerr.resample('D').interpolate()

discharge22_nomelt_df = nomelt_discharge
discharge22_nomelt_df['upper_err'] = nomelt_uerr
discharge22_nomelt_df['lower_err'] = nomelt_lerr
discharge22_nomelt_df['upp_discharge'] = discharge22_nomelt_df['discharge'] + discharge22_nomelt_df['upper_err']
discharge22_nomelt_df['low_discharge'] = discharge22_nomelt_df['discharge'] - discharge22_nomelt_df['lower_err']

annual_totals_nomlelt = discharge22_nomelt_df / 365 # Gigatonnes of ice per day
annual_totals_nomlelt = annual_totals_nomlelt.sum()

annual_nomelt = annual_totals_nomlelt['discharge']
annual_upper_error_nomelt = annual_totals_nomlelt['upper_err']
annual_lower_error_nomelt = annual_totals_nomlelt['lower_err']

################
dis_wmelt = discharge22_df / 365
dis_nomelt = nomelt_discharge / 365

discharge_dictionary = {}
discharge_dictionary['2022'] = discharge22
discharge_dictionary['2021'] = discharge21
discharge_dictionary['2020'] = discharge20
discharge_dictionary['2019'] = discharge19
discharge_dictionary['2018'] = discharge18
discharge_dictionary['2017'] = discharge17
discharge_dictionary['2016'] = discharge16


print(discharge_dictionary)
key_list = list(discharge_dictionary.keys())
annual_mean_list = []
for key in key_list:
    annual_mean = discharge_dictionary[key]['discharge'].mean()
    print('Annual mean for %s is: %s' % (key, annual_mean))
    
    annual_mean_list.append(annual_mean)

annual_mean_list = np.array([annual_mean_list])
print('All years STD', annual_mean_list.std())
print('All years MEAN', annual_mean_list.mean())

################

# changing index for other years to match 2022
discharge23.index = discharge23.index.map(lambda t: t.replace(year=2022))
discharge21.index = discharge22.index 
discharge20.index = discharge22.index 
discharge19 = discharge19.drop(pd.to_datetime('2020-02-29'))
discharge19.index = discharge22.index 
discharge18.index = discharge22.index 
discharge17.index = discharge22.index 
discharge16.index = discharge22.index 

# calculating the annual daily discharge from 2016 to 2022
annual_mean = sum_sector_discharge.groupby([sum_sector_discharge.index.day,sum_sector_discharge.index.month], as_index=False, sort=False).median()
annual_mean = annual_mean.drop(0)
annual_mean.index = pd.date_range(start='05-01-2022', end='04-30-2023', freq='1D') # month first?

discharge_allyears = np.stack([discharge22['discharge'], discharge21['discharge'], discharge20['discharge'], discharge19['discharge'], discharge18['discharge'], discharge17['discharge'], discharge16['discharge']])
discharge_median = np.median(discharge_allyears, axis=0)

melt_data = pd.read_csv(data_dir + 'met_data/KAN_L_daily_surfm.csv', index_col=0)
melt_data.index = pd.to_datetime(melt_data.index, dayfirst=True)
melt_data = melt_data['2022']

melt_data = melt_data[(melt_data.index >= '2022-05-01') & (melt_data.index <= '2023-04-30')]

#######################creating figure########################################
fig = plt.figure()
spec = gridspec.GridSpec(ncols=1, nrows=1)

ax = fig.add_subplot(spec[0])

melt_start = pd.to_datetime('2022-08-21')
melt_end = pd.to_datetime('2022-09-14') 
ax.axvspan(melt_start, melt_end, color='g', alpha=0.1, zorder=-1)

ax.xaxis.set_tick_params(which='major', size=5, width=1, direction='in', top='on')
ax.xaxis.set_tick_params(which='minor', size=5, width=1, direction='in', top='on')
ax.yaxis.set_tick_params(which='major', size=5, width=1, direction='in', right='on')
ax.yaxis.set_tick_params(which='minor', size=5, width=1, direction='in', right='on')

comb_error = [sum_l_error_22.values, sum_u_error_22.values]
ax.errorbar(discharge22.index, discharge22['discharge'], yerr=comb_error, zorder=1, ls='none', elinewidth=1.26, ecolor='dodgerblue', alpha=0.2, label='2022/23 error')
ax.plot(discharge22.index, discharge22['discharge'], color='k', zorder=2, linewidth=2)
ax.plot(discharge22.index, discharge22['discharge'], color='#29B6F6', zorder=2, label='2022/23', linewidth=1.5)

ax.plot(discharge21.index, discharge21, color='grey', zorder=1, alpha=0.5, label='Individual years \n2016 to 2022')
ax.plot(discharge20.index, discharge20, color='grey', zorder=1, alpha=0.5)
ax.plot(discharge19.index, discharge19, color='grey', zorder=1, alpha=0.5)
ax.plot(discharge18.index, discharge18, color='grey', zorder=1, alpha=0.5)
ax.plot(discharge17.index, discharge17, color='grey', zorder=1, alpha=0.5)
ax.plot(discharge16.index, discharge16, color='grey', zorder=1, alpha=0.5)

ax.plot(discharge22.index, discharge_median, zorder=1, color='k', alpha=0.6, linewidth=2, label='2016 to 2022 median')

ax.legend(loc='upper right', fontsize=8,  fancybox=True, framealpha=0.9, facecolor='w')
ax.set_ylabel('Ice Discharge (Gt yr$^{-1}$)')

bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
percent_increase = ((annual - annual_nomelt)/ annual_nomelt) * 100

text = 'Annual mean: %.2f  Gt yr$^{-1}$ \
        \nAnnual mean without melt event: %.2f  Gt yr$^{-1}$ \
        \nAnnual mean increase: %.2f %%' \
        % (annual, annual_nomelt, percent_increase)

ax.text(0.3, 0.8, text, transform=ax.transAxes, bbox=bbox_props, size=6)

myFmt = mdates.DateFormatter('%b')
ax.xaxis.set_minor_formatter(myFmt)
ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))

myFmt = mdates.DateFormatter('%b')
ax.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(mdates.YearLocator(1, month=1, day=1))
ax.set_ylim(0,2.2)

xlim_time = discharge22.index
ax.set_xlim(xlim_time[0], xlim_time[-1])

axb = ax.twinx()
axb.bar(melt_data.index, melt_data*100, color='r', zorder=-10, edgecolor='k', linewidth=0.3, width=1, label='Daily Surface Melt 2022')
axb.set_ylim(0, 20)
axb.set_yticks([0, 2, 4, 6, 8, 10])
axb.tick_params(axis='y', colors='red')
axb.set_ylabel('Surface Melt (cm w.e. day$^{-1}$)', color='red')
axb.legend(fontsize=8,  fancybox=True, framealpha=0.9, facecolor='w', loc='lower right')

ax.set_zorder(1)  
ax.patch.set_visible(False)  
fig.set_size_inches(width, height)

root = os.getcwd()
fig.savefig(root + '/code/figures/fig5_discharge_land.jpg', dpi=400, bbox_inches='tight')
plt.show()
