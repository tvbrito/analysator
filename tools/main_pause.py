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
line = 'dayside_slice_3001to3710'
# filenumber you want to start to read the data
start=3001
# end treading the data
end=3710 
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



variables_pause={'Bx':0, 'By':0, 'Bz':0, 'Ex':0, 'Ey':0,'Ez':0, 'rho':0, 'vx':0, 'vy':0,'vz':0, 'Pressure':0, 'coordinate_x':0,'coordinate_y':0,'coordinate_z':0, 'time':0}

variables_mpause=rd.read_from_local_file_dictionary(path, variables_pause)

vz=variables_mpause['vz']
time=variables_mpause['time']
distance=variables_mpause['coordinate_z'][0]


path=main_path+run+"/mpause_sheath/"

variables_sheath={'Bx':0, 'By':0, 'Bz':0, 'Ex':0, 'Ey':0,'Ez':0, 'rho':0, 'vx':0, 'vy':0,'vz':0, 'Pressure':0, 'coordinate_x':0,'coordinate_y':0,'coordinate_z':0, 'time':0}


variables_sheath=rd.read_from_local_file_dictionary(path, variables_sheath)

path=main_path+run+"/mpause_sphere/"

variables_sphere={'Bx':0, 'By':0, 'Bz':0, 'Ex':0, 'Ey':0,'Ez':0, 'rho':0, 'vx':0, 'vy':0,'vz':0, 'Pressure':0, 'coordinate_x':0,'coordinate_y':0,'coordinate_z':0, 'time':0}


variables_sphere=rd.read_from_local_file_dictionary(path, variables_sphere)

rho_sheath=[]
rho_sphere=[]
Bz_sheath=[]
Bz_sphere=[]
sphere=12  #7
sheath=33  #29  #len(coordinate_x)-1


Bz_sheath=variables_sheath['Bz']
Bz_sphere=variables_sphere['Bz']
rho_sheath=variables_sheath['rho']
rho_sphere=variables_sphere['rho']

#v_expected=np.sqrt( Bz_sphere*Bz_sheath*(Bz_sphere+Bz_sheath)/ (constants.mu_0()*constants.proton_mass() * ( rho_sheath*Bz_sphere+ rho_sphere*Bz_sheath )))

#-----------------------
path=main_path+run+"/"+line


variable_step=20e3
time_mark=30



vx=variables_mpause['vx']
vy=variables_mpause['vy']
vz=variables_mpause['vz']

v =np.sqrt( vx**2 + vy**2 + vz**2)


p.plot_stack_lines(path, v, 'v', variable_step, distance, time_factor, time_mark)


Bx=variables_mpause['Bx']
By=variables_mpause['By']
Bz=variables_mpause['Bz']

B =np.sqrt( Bx**2 + By**2 + Bz**2)

variable_step=2e-9

#p.plot_stack_lines(path, B, 'B', variable_step, distance, time_factor, time_mark)





p.plot_as_a_function_of_distance_at_each_time(path, vz, distance, time, time_factor, 20)

variable =B
variable_name='B'   

fig = plt.figure()
pl.subplots_adjust(hspace=0.3)
for t in range(len(variable[0])):
  
   if t%(time_mark/time_factor)==0:
      pl.plot( np.array(distance)/constants.RE(), B[:,t]+variable_step*t, 'r', lw=0.2 )
   else:
      pl.plot( np.array(distance)/constants.RE(), B[:,t]+variable_step*t, 'k', lw=0.1 ) 

   for j in range(len(vz)):
      if abs(vz[j,t]) < 50000 :
         plt.scatter( distance[j]/constants.RE(), B[j,t]+variable_step*t, marker='o',color="blue",s=1.0)
         
pl.xlim((-6,6))
pl.ylim((0,1.6e-6))
savepath = os.path.join(path+"/stack_"+variable_name+".png" )
pl.savefig(savepath, dpi = 400)
pl.close()