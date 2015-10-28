import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.axes as axes
import math
from numpy import linalg as LA
import cmath
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


import constants



def plot_stack_lines(path, variable, variable_name, variable_step, distance, time_factor, time_mark):
  
   fig = plt.figure()
   pl.subplots_adjust(hspace=0.3)
   for t in range(len(variable[0])):

      if t%(time_mark/time_factor)==0:
         pl.plot( np.array(distance)/constants.RE(), variable[:,t]+variable_step*t, 'r', lw=0.2 )
      else:
         pl.plot( np.array(distance)/constants.RE(), variable[:,t]+variable_step*t, 'k', lw=0.1 ) 

   savepath = os.path.join(path+"/stack_"+variable_name+".png" )
   pl.savefig(savepath, dpi = 400)
   pl.close()
   
   return

def plot_stack_color(path, variable, variable_name, distance, time, time_factor):

   fig = plt.figure()

   p = plt.pcolor(np.array(distance)/constants.RE(), np.multiply(time, time_factor), variable.T, cmap=cmx.jet) 

   pl.tick_params(axis='x',labelsize=15)
   pl.tick_params(axis='y',labelsize=15)

   pl.xlabel(" distance (R$_{E}$)",fontsize=15)

   savepath = os.path.join(path+"/stack_color_"+variable_name+".png" )
   pl.savefig(savepath, dpi = 400)
   pl.close()

   


def plot_stack_color_with_plasma_propagation(path, variable, variable_name, distance, v, time, time_factor):

   # PLASMA VELOCITY CALCULATIONS ##
   spacing = len(variable)/15
   
   #distance = coordinate_x
   
   b=[spacing*0]
   blue = [distance[b[0]]]
   r=[spacing*2]
   red = [distance[r[0]]]
   g=[spacing*4]
   green = [distance[g[0]]]
   y=[spacing*6]
   yellow = [distance[y[0]]]
   bl=[spacing*8]
   black = [distance[bl[0]]]
   m=[spacing*10]
   magenta = [distance[m[0]]]
   c=[spacing*10]
   cyan = [distance[c[0]]]
   r2=[spacing*12]
   red2 = [distance[r2[0]]]
   g2=[spacing*12]
   green2 = [distance[g2[0]]]
   y2=[spacing*15]
   yellow2 = [distance[y2[0]]]
   bl2=[spacing*15]
   black2 = [distance[bl2[0]]]
   
   

   for t in range(len(B_mag[0])):
      if t>0 :
         cell = 0

         for cell in range(len(B_mag)-1):
            v= np.sqrt(vx[cell][t]**2+vz[cell][t]**2)
            #if  vx[cell][t] < 0:
               #v=-v
            if t==len(blue) and blue[t-1] <= distance[cell] and  blue[t-1] > distance[cell+1]:
               b.append(cell)
               blue.append(blue[t-1]+v*step*time_factor)
            if t==len(red) and red[t-1] <= distance[cell] and  red[t-1] > distance[cell+1]:
               r.append(cell)
               red.append(red[t-1]+v*step*time_factor)
            if t==len(green) and green[t-1] <= distance[cell] and  green[t-1] > distance[cell+1]:
               g.append(cell)
               green.append(green[t-1]+v*step*time_factor)
            if t==len(yellow) and yellow[t-1] <= distance[cell] and  yellow[t-1] > distance[cell+1]:
               y.append(cell)
               yellow.append(yellow[t-1]+v*step*time_factor)
            if t==len(black) and black[t-1] <= distance[cell] and  black[t-1] > distance[cell+1]:
               bl.append(cell)
               black.append(black[t-1]+v*step*time_factor)
            if t==len(magenta) and magenta[t-1] <= distance[cell] and  magenta[t-1] > distance[cell+1]:
               m.append(cell)
               magenta.append(magenta[t-1]+v*step*time_factor)
            if t==len(cyan) and cyan[t-1] <= distance[cell] and  cyan[t-1] > distance[cell+1]:
               c.append(cell)
               cyan.append(cyan[t-1]+v*step*time_factor)
            if t==len(red2) and red2[t-1] <= distance[cell] and  red2[t-1] > distance[cell+1]:
               r2.append(cell)
               red2.append(red2[t-1]+v*step*time_factor)
            if t==len(green2) and green2[t-1] <= distance[cell] and  green2[t-1] > distance[cell+1]:
               g2.append(cell)
               green2.append(green2[t-1]+v*step*time_factor)
            if t==len(yellow2) and yellow2[t-1] <= distance[cell] and  yellow2[t-1] > distance[cell+1]:
               y2.append(cell)
               yellow2.append(yellow2[t-1]+v*step*time_factor)
            if t==len(black2) and black2[t-1] <= distance[cell] and  black2[t-1] > distance[cell+1]:
               bl2.append(cell)
               black2.append(black2[t-1]+v*step*time_factor)
               

               


#------------------------------------------------------------------------------------------   

   # PLOTTING THE STACkS:

### PLOT THE STACKS USING:

   
   p = ax.pcolor(np.array(distance)/constants.RE(), np.multiply(time, time_factor), variable.T, cmap=cmx.jet)# , vmin=-12e-9 , vmax=-5e-9, norm=LogNorm()) 

   pl.tick_params(axis='x',labelsize=15)
   pl.tick_params(axis='y',labelsize=15)
   pl.xlabel(" distance (R$_{E}$)",fontsize=15)

   for  t in range(len(B_mag[0])):
      
      if t<len(b)-1 and b[t]!=b[t-1]:
         ax.scatter(blue[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black", s=1.0)
      if t<len(r)-1 and r[t]!=r[t-1]:   
         ax.scatter(red[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(g)-1 and g[t]!=g[t-1]:
         ax.scatter(green[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(y)-1 and y[t]!=y[t-1]: 
         ax.scatter(yellow[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(bl)-1 and bl[t]!=bl[t-1]:
         ax.scatter(black[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(m)-1 and m[t]!=m[t-1]:
         ax.scatter(magenta[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(c)-1 and c[t]!=c[t-1]:
         ax.scatter(cyan[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(r2)-1 and r2[t]!=r2[t-1]:
         ax.scatter(red2[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(g2)-1 and g2[t]!=g2[t-1]:
         ax.scatter(green2[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(y2)-1 and y2[t]!=y2[t-1]:
         ax.scatter(yellow2[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)
      if t<len(bl2)-1 and  bl2[t]!=bl2[t-1]:
         ax.scatter(black2[t]*1.5704887e-7, time[t]*time_factor, marker='o',color="black",s=1.0)

   savepath = os.path.join(path+"/stack_"+variable_name+".png" )
   pl.savefig(savepath, dpi = 400)
   pl.close()


def plot_as_a_function_of_distance_at_each_time(path, variable, distance, time, time_factor, time_step_for_saving):
   for i in range(len(variable[0])):
      if i%(time_step_for_saving/time_factor)==0:
         pl.figure(1)
         title=" time ="+str(time[i]*time_factor)+"s" 
         pl.title(title)
         #pl.subplot(212)
         pl.plot( np.array(distance)/constants.RE(), variable[:,i], 'k',linewidth=3) 
         pl.axhline(y=0, color='k', lw=1)
         pl.tick_params(axis='x',labelsize=17)
         pl.tick_params(axis='y',labelsize=17)
         pl.xlabel("distance",fontsize=17) 
         savepath = os.path.join(path+"/variable_at_time_"+str(time[i]*time_factor)+".png" )
         pl.savefig(savepath)
         pl.close()
      
   
   
def plot_as_a_function_of_time_at_each_point(path, variable, distance, time, time_factor):
   
   for i in range(len(variable)):
      pl.figure(1)
      title=" distance ="+str(round(distance[i]/constants.RE(),3))+"Re" 
      pl.title(title)
      #pl.subplot(212)
      pl.plot(np.multiply(time,time_factor), variable[i], 'k--',linewidth=3)      
      pl.tick_params(axis='x',labelsize=17)
      pl.tick_params(axis='y',labelsize=17)
      pl.xlabel(" Time",fontsize=17) 
      savepath = os.path.join(path+"/", "variable_at_point_"+str(i+1)+".png" )
      pl.savefig(savepath)
      pl.close()


def plot_amplitude_as_a_function_of_time_at_each_point(path, variable, distance, time, time_factor):
      
   for i in range(len(variable)):
      
      var=variable[i]
      var_mean=np.mean(var)
      var_diff=var-var_mean 
      
      
      pl.figure(1)
      title=" distance ="+str(round(distance[i]/constants.RE(),3))+"Re" 
      pl.title(title)      
      pl.plot(np.multiply(time,time_factor), np.multiply(var_diff, 100.0/var_mean)  , 'k',linewidth=3)
 
      pl.tick_params(axis='x',labelsize=17)
      pl.tick_params(axis='y',labelsize=17)
      pl.xlabel(" Time",fontsize=17)
      pl.ylabel("Fluctuation amplitude (%)",fontsize=17)
      savepath = os.path.join(path+"/","fluctuation_amplitude_"+str(i+1)+".png" )
      pl.savefig(savepath)
      pl.close()  
   
   
   
   
   

