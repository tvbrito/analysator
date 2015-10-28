import pytools as pt
import numpy as np
import read_data as rd

import constants
import plotting as p

import pylab as pl
import matplotlib.pyplot as plt
import os
#import MVA



#import os
#from scipy.stats.stats import pearsonr
#from scipy.stats import kurtosis
#import matplotlib.colors as colors
#import matplotlib.cm as cmx
#import matplotlib.axes as axes
##from scipy.signal import argrelextrema
import math
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
line = 'reconnection'
# filenumber you want to start to read the data
start=2630
# end treading the data
end=2820
# how often you want to read the file, 1 every file, 2 every other file, 3 every third file...
step=1



# options for what will be done during the calculations, 1 IS YES, 0 is no

use_coordinates= 0
points_in_RE=1

#read data from vlsv files = 1, anything else use saved values 

read_from_vlsv = 0

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



variables={'Bx':0, 'By':0, 'Bz':0, 'Ex':0, 'Ey':0,'Ez':0, 'rho':0, 'vx':0, 'vy':0,'vz':0, 'Temperature':0, 'TPerpendicular':0, 'TParallel':0, 'betaPerpendicular':0, 'betaParallel':0, 'Pressure':0, 'beta':0,'coordinate_x':0,'coordinate_y':0,'coordinate_z':0, 'time':0}

variables_reconnection=rd.read_from_local_file_dictionary(path, variables)




# --- --- ---
v_max_plus3 = []
v_max_plus2 = []
v_max_plus1 = []
v_max_minus1 = []
v_max_minus2 = []
v_max_minus3 = []

line = 'reconnection_plus3'
path=main_path+run+"/"+line

variables={'vx':0, 'vy':0, 'vz':0, 'Bz':0, 'rho':0, 'coordinate_x':0, 'coordinate_z':0}

variables_plus3=rd.read_from_local_file_dictionary(path, variables)

vz_plus3=variables_plus3['vz']

for i in range(len(vz_plus3[0])):
   Vz=vz_plus3[:,i]
   v_max_plus3.append(np.max(Vz))
   

line = 'reconnection_plus2'
path=main_path+run+"/"+line

variables={'vx':0, 'vy':0, 'vz':0, 'Bz':0, 'rho':0, 'coordinate_x':0, 'coordinate_z':0}

variables_plus2=rd.read_from_local_file_dictionary(path, variables)

vz_plus2=variables_plus2['vz']

for i in range(len(vz_plus2[0])):
   Vz=vz_plus2[:,i]
   v_max_plus2.append(np.max(Vz))
   
line = 'reconnection_plus1'
path=main_path+run+"/"+line

variables={'vx':0, 'vy':0, 'vz':0,'Bx':0, 'By':0, 'Bz':0, 'rho':0, 'coordinate_x':0, 'coordinate_z':0}

variables_plus1=rd.read_from_local_file_dictionary(path, variables)

vz_plus1=variables_plus1['vz']


Bx=variables_plus1['Bx']
By=variables_plus1['By']
Bz=variables_plus1['Bz']

B =np.sqrt( Bx**2 + By**2 + Bz**2)

rho=variables_plus1['rho']

for i in range(len(vz_plus1[0])):
   Vz=vz_plus1[:,i]
   v_max_plus1.append(np.max(Vz))
   
line = 'reconnection_minus3'
path=main_path+run+"/"+line

variables={'vx':0, 'vy':0, 'vz':0, 'Bz':0, 'rho':0, 'coordinate_x':0, 'coordinate_z':0}

variables_minus3=rd.read_from_local_file_dictionary(path, variables)

vz_minus3=variables_minus3['vz']

for i in range(len(vz_minus3[0])):
   Vz=vz_minus3[:,i]
   v_max_minus3.append(abs(np.min(Vz)))


line = 'reconnection_minus2'
path=main_path+run+"/"+line

variables={'vx':0, 'vy':0, 'vz':0, 'Bz':0, 'rho':0, 'coordinate_x':0, 'coordinate_z':0}

variables_minus2=rd.read_from_local_file_dictionary(path, variables)

vz_minus2=variables_minus2['vz']

for i in range(len(vz_minus2[0])):
   Vz=vz_minus2[:,i]
   v_max_minus2.append(abs(np.min(Vz)))


line = 'reconnection_minus1'
path=main_path+run+"/"+line

variables={'vx':0, 'vy':0, 'vz':0, 'Bz':0, 'rho':0, 'coordinate_x':0, 'coordinate_z':0}

variables_minus1=rd.read_from_local_file_dictionary(path, variables)

vz_minus1=variables_minus1['vz']

for i in range(len(vz_minus1[0])):
   Vz=vz_minus1[:,i]
   v_max_minus1.append(abs(np.min(Vz)))

# ---------------

time=variables_reconnection['time']

#coordinate_x=variables_reconnection['coordinate_x']
#Bz=variables_reconnection['Bz']
#rho=variables_reconnection['rho']

rho_sheath=[]
rho_sphere=[]
Bz_sheath=[]
Bz_sphere=[]
sphere=12  #7
sheath=33  #29  #len(coordinate_x)-1

for i in range(len(time)):
   b=Bz[:,i]
   n=rho[:,i]
   Bz_sheath.append(abs(b[sheath]))
   Bz_sphere.append(abs(b[sphere]))
   rho_sheath.append(n[sheath])
   rho_sphere.append(n[sphere])

Bz_sheath=np.array(Bz_sheath)
Bz_sphere=np.array(Bz_sphere)
rho_sheath=np.array(rho_sheath)
rho_sphere=np.array(rho_sphere)

v_expected=np.sqrt( Bz_sphere*Bz_sheath*(Bz_sphere+Bz_sheath)/ (constants.mu_0()*constants.proton_mass() * ( rho_sheath*Bz_sphere+ rho_sphere*Bz_sheath )))

#-----------------------


line = 'reconnection'
path=main_path+run+"/"+line

#pl.figure(1)
##pl.plot( time*time_factor, np.array(v_max_plus3)*1e-3 , 'r.', markersize=1, label="$v_{+3}$")
##pl.plot(time*time_factor, np.array(v_max_plus2)*1e-3 , 'r--',linewidth=2, label="$v_{+2}$")
##pl.plot(time*time_factor, np.array(v_max_plus1)*1e-3 , 'r',linewidth=3, label="$v_{+1}$")
#pl.plot(time*time_factor, np.array(v_max_minus3)*1e-3 , 'b.',markersize=1, label="$v_{-3}$")
#pl.plot(time*time_factor, np.array(v_max_minus2)*1e-3 , 'b--',linewidth=2, label="$v_{-2}$")
#pl.plot(time*time_factor, np.array(v_max_minus1)*1e-3 , 'b',linewidth=3, label="$v_{-1}$")
#pl.plot(time*time_factor, np.array(v_expected)*1e-3 , 'k',linewidth=4, label="$v_{exp}$")
#pl.tick_params(axis='x',labelsize=17)
#pl.tick_params(axis='y',labelsize=17)
#pl.xlabel(" Time",fontsize=17)
#pl.legend(loc='upper right', prop={'size':8}) 
#pl.ylabel("V$_{jet}$",fontsize=17)
#savepath = os.path.join(path+"/","reconnection_jet_speed_neg_"+str(i+1)+".png" )
#pl.savefig(savepath)
#pl.close()  


variable_step=0.001
#distance=variables_reconnection['coordinate_x']
time_mark=30

time_step_for_saving=10

#variable=variables_reconnection['rho']
#p.plot_as_a_function_of_distance_at_each_time(path, variables_reconnection['vz'], distance, time, time_factor, time_step_for_saving)

#p.plot_amplitude_as_a_function_of_time_at_each_point(path, variable, distance, time, time_factor)
distance=variables_plus1['coordinate_x']

for i in range(len(B[0])):
   if i%(10/time_factor)==0:
      pl.figure(1)
      title=" time ="+str(time[i]*time_factor)+"s" 
      pl.title(title)
      pl.subplot(211)
      pl.plot( np.array(distance)/constants.RE(), B[:,i]*1e9, 'k',linewidth=3) 
      pl.ylabel("B",fontsize=17)
      pl.subplot(212)
      pl.plot( np.array(distance)/constants.RE(), rho[:,i]*1e-6, 'k',linewidth=3)
      pl.ylabel("rho",fontsize=17)
      pl.xlabel("distance",fontsize=17) 
      savepath = os.path.join(path+"/B_rho_at_time_"+str(time[i]*time_factor)+".png" )
      pl.savefig(savepath)
      pl.close()
