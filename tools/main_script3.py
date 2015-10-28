import pytools as pt
import numpy as np
import read_data as rd

import constants
import plotting as p

#import MVA


#-- if you want to plot in this script:
#import pylab as pl
#import matplotlib.pyplot as plt
#import os


#import os
#from scipy.stats.stats import pearsonr
#from scipy.stats import kurtosis
#import matplotlib.colors as colors
#import matplotlib.cm as cmx
#import matplotlib.axes as axes
##from scipy.signal import argrelextrema
#import math
#from numpy import linalg as LA
#import cmath
#from matplotlib.colors import LogNorm
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from scipy.stats import skew
#from scipy import signal, fftpack


# ADD constant =
# constants.RE() etc.


# --- MAIN PATH --- 
#main_path="/home/hoilijo/Documents/Projektit/msheath_antisymmetries/"
main_path="/home/hoilijo/Documents/Projektit/dayside_rec/"

path_to_vlsv="/home/hoilijo/Servers/lustre_alfthan/2D/BCB/bulk/"
# ----------------

# --- DEFINE PARAMS FOR RUN --- 

# name of the run:
run='BCB'
# how often is the output saved:
time_factor=0.5
# equatorial plane = 0, polar plane = 1:
polar=1
# name of the points you want to look, for example cut_along_x, streamline
line = 'mpause_sheath'
# filenumber you want to start to read the data
start=2100
# end treading the data
end=3710
# how often you want to read the file, 1 every file, 2 every other file, 3 every third file...
step=1



# options for what will be done during the calculations, 1 IS YES, 0 is no

use_coordinates= 1
points_in_RE=1

#read data from vlsv files = 1, anything else use saved values 

read_from_vlsv = 1

# Take interpolation between the points:
interpolate = 0
# if yes, then how many points you wnt to interpolate over:
points_for_interpolation=5

#------------------------------------------------------------------------------------------------------------------------------------------

print "running start"
point_number=1


path=main_path+run+"/"+line

# use variable names as they are listed in pytools.vlsvfile.VlsvReader('bulk.000xxxx.vlsv').list()
# otherwise you get chrash : 
# CellID B  E rho rho_v TParallel TPerpendicular TPerpOverPar betaPerpOverPar betaPerpendicular betaParallel Pressure beta v, you can add
# if you don't need everything listed, ypu can remove them from the variables, take out the whole 'paramter':0


# --- IF YOU WANT TO READ FROM VLSV: ---
if read_from_vlsv == 1:
   #for reading from vlsv:
   variables={'B':0, 'E':0, 'rho':0, 'v':0, 'Temperature':0, 'TPerpendicular':0, 'TParallel':0, 'betaPerpendicular':0, 'betaParallel':0, 'Pressure':0, 'beta':0}
   #retunrs vector fields as components Bx, By, Bz and additionally time, cellids and coordinates
   variable_values=rd.read_from_vlsv_dictionary(path, path_to_vlsv, variables, use_coordinates, points_in_RE, interpolate, points_for_interpolation, start, end, step)

# --- --- ---

# --- IF YOU WANT TO READ FROM  TXT FILES THAT WERE SAVED EARLIER ---
else:
   #for reading from file open up the vectors as components, add time and coordinate_x, coordinate_y, coordinate_z, if you need them
   variables={'Bx':0, 'By':0, 'Bz':0, 'Ex':0, 'Ey':0,'Ez':0, 'rho':0, 'vx':0, 'vy':0,'vz':0, 'Temperature':0, 'TPerpendicular':0, 'TParallel':0, 'betaPerpendicular':0, 'betaParallel':0, 'Pressure':0, 'beta':0,'coordinate_x':0,'coordinate_y':0,'coordinate_z':0, 'time':0 }

   variable_values=rd.read_from_local_file_dictionary(path, variables)

# --- --- ---


## HOW TO USE THESE VALUES: 
# now you have all the values in a python dictionary, you can get the certain paramer out:
# Bx = variables['Bx']
# coordinate_x = variables['coordinate_x']
# you can get the list of variable names with variables.keys()

# Each variable is now in 2-dimesional array, "outer dimension" is distance, "inner" is time
# meaning that if you want to loop over points
# for i in range(len(Bx)):
#    Bx_at_point=Bx[i]
#
# and if you want to loop over time
# for i in range(len(Bx[0])):
#    B_at_time=np.array(Bx)[:,i]



print variables





