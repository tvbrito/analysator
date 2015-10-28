## THIS HAVE NOT BEEN TESTED YET


import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.axes as axes
import math
from numpy import linalg as LA
import cmath
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


import functions 


def mva(Bmag, Bx, By, Bz):

   lambda_max_int=[]
   lambda_int_min=[]
   angle_between=[]
   time=movingaverage(time, moving_average_time_window)


   for i in range(len(Bmag)):

      BX=Bx[i]#[180:]
      BY=By[i]#[180:]
      BZ=Bz[i]#[180:]
      B_magnitude=Bmag[i]#[180:]
      
      
      
      BX_sum=sum(BX)
      BY_sum=sum(BY)
      BZ_sum=sum(BZ) 

      BX_mean=np.mean(BX)
      BY_mean=np.mean(BY)
      BZ_mean=np.mean(BZ)
      B_mean=np.mean(B_magnitude)
      B_diff=(B_magnitude-B_mean)

      
      #BX=movingaverage(BX,moving_average_time_window)
      #BY=movingaverage(BY,moving_average_time_window)
      #BZ=movingaverage(BZ,moving_average_time_window)
      #B_magnitude=movingaverage(B_magnitude, moving_average_time_window)
      
      
      N=len(BX)
      
      
      # Variance Matrix 1st way
      
      M11=0 #BXX_sum/(N*1.0)+BX_sum**2/(1.0*N**2)
      M22=0#BYY_sum/(N*1.0)+BY_sum**2/(1.0*N**2)
      M33=0#BZZ_sum/(N*1.0)+BZ_sum**2/(1.0*N**2)
      
      M12=0#BXY_sum/(N*1.0)+BX_sum*BY_sum/(1.0*N**2)
      M13=0#BYY_sum/(N*1.0)+BX_sum*BZ_sum/(1.0*N**2)
      M23=0#BYZ_sum/(N*1.0)+BY_sum*BZ_sum/(1.0*N**2)
      
      for n in range(N):
         M11=M11+(BX[n]-BX_mean)**2/(1.0*N)
         M22=M22+(BY[n]-BY_mean)**2/(1.0*N)
         M33=M33+(BZ[n]-BZ_mean)**2/(1.0*N)
         M12=M12+(BX[n]-BX_mean)*(BY[n]-BY_mean)/(1.0*N)
         M13=M13+(BX[n]-BX_mean)*(BZ[n]-BZ_mean)/(1.0*N)
         M23=M23+(BY[n]-BY_mean)*(BZ[n]-BZ_mean)/(1.0*N)
      
      
      #M11=np.mean(np.multiply((BX-BX_mean),(BX-BX_mean)))  
      #M22=np.mean(np.multiply((BY-BY_mean),(BY-BY_mean)))
      #M33=np.mean(np.multiply((BZ-BZ_mean),(BZ-BZ_mean)))
      #M12=np.mean(np.multiply((BX-BX_mean),(BY-BY_mean)))
      #M13=np.mean(np.multiply((BX-BX_mean),(BZ-BZ_mean)))
      #M23=np.mean(np.multiply((BZ-BZ_mean),(BY-BY_mean)))
      
      Variance_M=np.array([[M11, M12, M13], [M12, M22, M23], [M13, M23, M33]])
      
      
      eigen_values1, eigen_vectors1= LA.eig(Variance_M)
      
      #eigen_values, eigen_vectors = (list(x) for x in zip(*sorted(zip(eigen_values1, eigen_vectors1))))
      
      eigen_values=np.copy(eigen_values1)
      lamda_A=eigen_values[0]
      lamda_B=eigen_values[1]
      lamda_C=eigen_values[2]
      
      e_vector_A=eigen_vectors1[:,0]
      e_vector_B=eigen_vectors1[:,1]
      e_vector_C=eigen_vectors1[:,2]
      
   # eigen_values1=-eigen_values1
      eigen_values1.sort()
   # eigen_values1=-eigen_values1
      
      if eigen_values1[0]==lamda_A:
         lamdaMin=lamda_A
         eig_vec_Min=e_vector_A
      elif eigen_values1[0]==lamda_B:
         lamdaMin=lamda_B
         eig_vec_Min=e_vector_B
      elif eigen_values1[0]==lamda_C:
         lamdaMin=lamda_C
         eig_vec_Min=e_vector_C
         
      if eigen_values1[1]==lamda_A:
         lamdaInt=lamda_A
         eig_vec_Int=e_vector_A
      elif eigen_values1[1]==lamda_B:
         lamdaInt=lamda_B
         eig_vec_Int=e_vector_B
      elif eigen_values1[1]==lamda_C:
         lamdaInt=lamda_C
         eig_vec_Int=e_vector_C
         
      if eigen_values1[2]==lamda_A:
         lamdaMax=lamda_A
         eig_vec_Max=e_vector_A
      elif eigen_values1[2]==lamda_B:
         lamdaMax=lamda_B
         eig_vec_Max=e_vector_B
      elif eigen_values1[2]==lamda_C:
         lamdaMax=lamda_C
         eig_vec_Max=e_vector_C
     
     
     
     #This is for setting the desirable right handedness of the eigen vectors
      #if eig_vec_Max[0]>0:
         #eig_vec_Max=-eig_vec_Max
      #if eig_vec_Int[1]<0:
         #eig_vec_Int=-eig_vec_Int
      #if eig_vec_Min[2]>0:
         #eig_vec_Min=-eig_vec_Min
      
      lamda_rel1=lamdaMax/lamdaInt
      lamda_rel2=lamdaInt/lamdaMin
      lamda_rel3=lamdaMin/lamdaInt
      
      #print "largest value "+str(lamda1)+" and its eig vector "+str(eig_vec_1)
      #print "middle value "+str(lamda2)+" and its eig vector "+str(eig_vec_2)
      #print "smallest value "+str(lamda3)+" and its eig vector "+str(eig_vec_3)
      #print "the relation lamda1/lamda2 = "+str(lamda_rel1)+" and the relation lamda2/lamda3 = "+str(lamda_rel2)
      
      lambda_int_min.append(lamda_rel2)
      lambda_max_int.append(lamda_rel1)
      
      print "Point "+str(point_number)+":"
      print "lambda_{min} ="+str(lamdaMin)+" min var direction = "+str(eig_vec_Min)
      print "lambda_{int} ="+str(lamdaInt)+" int var direction = "+str(eig_vec_Int)
      print "lambda_{max} ="+str(lamdaMax)+" max var direction = "+str(eig_vec_Max)
      print "lambda_{max}/lambda_{int} ="+str(lamda_rel1)
      print "lambda_{int}/lambda_{min} ="+str(lamda_rel2)
      print "lambda_{min}/lambda_{int} ="+str(lamda_rel3)
            
      BMin=[]
      BInt=[]
      BMax=[]
      
      for n in range(N):
         BMin.append(eig_vec_Min[0]*BX[n]+eig_vec_Min[1]*BY[n]+eig_vec_Min[2]*BZ[n])
         BInt.append(eig_vec_Int[0]*BX[n]+eig_vec_Int[1]*BY[n]+eig_vec_Int[2]*BZ[n])
         BMax.append(eig_vec_Max[0]*BX[n]+eig_vec_Max[1]*BY[n]+eig_vec_Max[2]*BZ[n])
         
      B_mean_vec=np.array([BX_mean, BY_mean, BZ_mean])
      
      dot_prd=eig_vec_Max[0]*BX_mean+eig_vec_Max[1]*BY_mean+eig_vec_Max[2]*BZ_mean
      theta_max=(180/math.pi)*np.arccos(dot_prd/np.sqrt(BX_mean**2+BY_mean**2+BZ_mean**2))
      if theta_max >180 :
         theta_max = 360-theta_max
         
      dot_prd2=eig_vec_Min[0]*BX_mean+eig_vec_Min[1]*BY_mean+eig_vec_Min[2]*BZ_mean
      theta_min=180/math.pi*np.arccos(dot_prd2/np.sqrt(BX_mean**2+BY_mean**2+BZ_mean**2))
      if theta_min >90 :
         theta_min = 180-theta_min 
         
      angle_between.append(theta_max)
      
      print "angle between max var dir and ambient B is "+str(theta_max)   
      print "angle between min var dir and ambient B is "+str(theta_min)
      
      
      BMin_min=1e9*min(BMin)
      BMin_max=1e9*max(BMin)
      BMin_mid=0.5*(BMin_max+BMin_min)  
      BInt_min=1e9*min(BInt)
      BInt_max=1e9*max(BInt)
      BInt_mid=0.5*(BInt_max+BInt_min) 
      BMax_min=1e9*min(BMax)
      BMax_max=1e9*max(BMax)
      BMax_mid=0.5*(BMax_max-BMax_min)
      BMax_mid_point=0.5*(BMax_max+BMax_min)
      
      BX_diff=BX-BX_mean
      BY_diff=BY-BY_mean
      BZ_diff=BZ-BZ_mean

      
      B_rel=np.multiply(B_diff, 100/B_mean)
      
      label_text="$\lambda_{max}/\lambda_{int}=$"+str(lamda_rel1)+" $\lambda_{min}/\lambda_{int}=$"+str(lamda_rel3)
      
      
         
      maxim=max(BX_diff)
      minim=min(BX_diff)

         
      if max(BY_diff) > maxim:
         maxim = max(BY_diff)
      if max(BZ_diff) > maxim:
         maxim = max(BZ_diff)
      if min(BY_diff) < minim:
         minim = min(BY_diff)
      if min(BZ_diff) < minim:
         minim = min(BZ_diff)
      
      ###
      BX_diff=movingaverage(BX_diff,moving_average_time_window)
      BY_diff=movingaverage(BY_diff,moving_average_time_window)
      BZ_diff=movingaverage(BZ_diff,moving_average_time_window)
      B_diff=movingaverage(B_diff,moving_average_time_window)
      BMin=movingaverage(BMin, moving_average_time_window)
      BInt=movingaverage(BInt, moving_average_time_window)
      BMax=movingaverage(BMax, moving_average_time_window)
      


      
      #--- plot hodograms
      pl.figure(1)
      pl.subplots_adjust(hspace=0.5)
      pl.subplots_adjust(wspace=0.2)
      
      # --------------------------
      pl.subplot((311), aspect='equal')
      pl.plot(np.multiply(1e9,BX_diff), np.multiply(1e9,BY_diff), 'k',linewidth=1)
      pl.plot(1e9*BX_diff[0], 1e9*BY_diff[0], 'ro', markersize=4 )
      pl.xlabel("$B_{x} (nT)$")
      pl.ylabel("$B_{y} (nT)$")
      pl.xlim([-BMax_mid, BMax_mid])
      pl.ylim([-BMax_mid, BMax_mid])
      #pl.xlim([(BX_min+BX_mid)-BY_mid, (BX_min+BX_mid)+BY_mid])
      #pl.ylim([BY_min, BY_max])
      pl.subplot((312), aspect='equal')
      pl.plot(np.multiply(1e9,BX_diff), np.multiply(1e9,BZ_diff), 'k',linewidth=1)
      pl.plot(1e9*BX_diff[0], 1e9*BZ_diff[0], 'ro', markersize=4 )
      pl.xlabel("$B_{x}(nT)$")
      pl.ylabel("$B_{z}(nT)$")
      pl.xlim([-BMax_mid, BMax_mid])
      pl.ylim([-BMax_mid, BMax_mid])
      # pl.xlim([(BX_min+BX_mid)-BZ_mid, (BX_min+BX_mid)+BZ_mid])
      # pl.ylim([BZ_min, BZ_max])
      pl.subplot((313), aspect='equal')
      pl.plot(np.multiply(1e9,BY_diff),np.multiply(1e9,BZ_diff), 'k',linewidth=1)
      pl.plot(1e9*BY_diff[0], 1e9*BZ_diff[0], 'ro', markersize=4 )
      pl.xlabel("$B_{y}(nT)$")
      pl.ylabel("$B_{z}(nT)$")
      pl.xlim([-BMax_mid, BMax_mid])
      pl.ylim([-BMax_mid, BMax_mid])
      #pl.xlim([(BY_min+BY_mid)-BZ_mid, (BY_min+BY_mid)+BZ_mid])
      # pl.ylim([BZ_min, BZ_max])  
      
      savepath = os.path.join(path_to_file,"hodograms_gse"+str(point_number)+".png" )
      pl.savefig(savepath)
      pl.close()
      
      #pl.figure(1)
      #pl.subplots_adjust(hspace=0.5)
      #pl.subplot((311), aspect='equal')
      #pl.plot(np.multiply(1e9,BMin), np.multiply(1e9,BInt), 'k',linewidth=1)
      #pl.plot(1e9*BMin[0], 1e9*BInt[0], 'ro', markersize=4 )
      #pl.xlim([BMin_mid-BMax_mid, BMin_mid+BMax_mid])
      #pl.ylim([BInt_mid-BMax_mid, BInt_mid+BMax_mid])
      #pl.xlabel("$B_{min} (nT)$")
      #pl.ylabel("$B_{int} (nT)$")
      
      #pl.subplot((312), aspect='equal')
      #pl.plot(np.multiply(1e9,BMin), np.multiply(1e9,BMax), 'k',linewidth=1)
      #pl.plot(1e9*BMin[0], 1e9*BMax[0], 'ro', markersize=4 )
      #pl.xlabel("$B_{min} (nT)$")
      #pl.ylabel("$B_{max} (nT)$")
      #pl.xlim([BMin_mid-BMax_mid, BMin_mid+BMax_mid])
      #pl.ylim([BMax_mid_point-BMax_mid, BMax_mid_point+BMax_mid])
      ##plt.axis('equal')
      
      #pl.subplot((313), aspect='equal')
      #pl.plot(np.multiply(1e9,BInt), np.multiply(1e9,BMax), 'k',linewidth=1)
      #pl.plot(1e9*BInt[0], 1e9*BMax[0], 'ro', markersize=4 )
      #pl.xlabel("$B_{int} (nT)$")
      #pl.ylabel("$B_{max} (nT)$")
      #pl.xlim([BInt_mid-BMax_mid, BInt_mid+BMax_mid])
      #pl.ylim([BMax_mid_point-BMax_mid, BMax_mid_point+BMax_mid])



      #savepath = os.path.join(path_to_file,"hodograms_minvar_dir"+str(point_number)+".png" )
      #pl.savefig(savepath)
      #pl.close()

      #fig = plt.figure()
      #ax = fig.add_subplot(111, projection='3d')
      #ax.plot(BX_diff, BY_diff, np.multiply(np.ones(len(BZ_diff)),minim ), "k",linewidth=0.5 )
      #ax.scatter(BX_diff[0], BY_diff[0], minim, marker='o',color="k",s=2)
      #ax.plot(BX_diff, np.multiply(np.ones(len(BY_diff)),maxim ), BZ_diff, "k",linewidth=0.5 )
      #ax.scatter(BX_diff[0], maxim, BZ_diff[0], marker='o',color="k",s=2)
      #ax.plot(np.multiply(np.ones(len(BX_diff)),minim), BY_diff, BZ_diff, "k",linewidth=0.5 )
      #ax.scatter(minim, BY_diff[0], BZ_diff[0], marker='o',color="k",s=2)
      
      #ax.plot(BX_diff, BY_diff, BZ_diff, linewidth=2)
      #ax.scatter(BX_diff[0], BY_diff[0], BZ_diff[0], marker='o',color="red",s=6)

      #ax.set_xlabel('Bx')
      #ax.set_xlim(minim, maxim)
      #ax.set_ylabel('By')
      #ax.set_ylim(minim, maxim)
      #ax.set_zlabel('Bz')
      #ax.set_zlim(minim, maxim)
      
      #savepath = os.path.join(path_to_file,"hodograms_GSE_3D"+str(point_number)+".png" )
      #pl.savefig(savepath)
      #pl.close()
      
      point_number=point_number+1 

   #silentremove(path_to_file+"/lambda_int_min")
   #silentremove(path_to_file+"/lambda_max_int")
   #silentremove(path_to_file+"/lambda_max_int")

   #np.savetxt(path_to_file+"/lambda_int_min", lambda_int_min)
   #np.savetxt(path_to_file+"/lambda_max_int", lambda_max_int)
   #np.savetxt(path_to_file+"/angle_between", angle_between)





