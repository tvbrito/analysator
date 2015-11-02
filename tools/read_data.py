import pytools as pt
import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import functions
import constants
import copy





def read_from_local_file(path):
#This reads the data to from the locally save files, faster because does not require opening the vlsv files
   cellids=np.loadtxt(path+"/cellids_array")
   Bmag=np.loadtxt(path+"/B_array")
   Bx=np.loadtxt(path+"/Bx_array")
   By=np.loadtxt(path+"/By_array")
   Bz=np.loadtxt(path+"/Bz_array")
   Emag=np.loadtxt(path+"/E_array")
   Ex=np.loadtxt(path+"/Ex_array")
   Ey=np.loadtxt(path+"/Ey_array")
   Ez=np.loadtxt(path+"/Ez_array")
   density=np.loadtxt(path+"/density_array")
   Vmag=np.loadtxt(path+"/V_array")
   Vx=np.loadtxt(path+"/Vx_array")
   Vy=np.loadtxt(path+"/Vy_array")
   Vz=np.loadtxt(path+"/Vz_array")
   T_perp=np.loadtxt(path+"/Tperp_array")
   T_par=np.loadtxt(path+"/Tpar_array")
   beta_perp=np.loadtxt(path+"/betaperp_array")
   beta_par=np.loadtxt(path+"/betapar_array")
   pressure=np.loadtxt(path+"/pressure_array")
   time=np.loadtxt(path+"/time_array")
  # distance=np.loadtxt(path+"/distance_array")
   coordinate_x=np.loadtxt(path+"/coordinate_x")
   coordinate_y=np.loadtxt(path+"/coordinate_y")
   coordinate_z=np.loadtxt(path+"/coordinate_z")

   return {'cellids':cellids, 'Bmag':Bmag, 'Bx':Bx, 'By':By, 'Bz':Bz, 'Emag':Emag, 'Ex':Ex, 'Ey':Ey, 'Ez':Ez, 'density':density, 'Vmag':vmag, 'Vx':vx, 'Vy':vy, 'Vz':vz, 'T_perp':T_perp, 'Tpar':Tpar, 'beta_perp':beta_perp, 'beta_par':beta_par, 'pressure':pressure, 'time':time, 'coordinate_x':coordinate_x, 'coordinate_y':coordinate_y, 'coordinate_z':coordinate_z}


def read_from_local_file_dictionary(path, variables):
#This reads the data to from the locally save files, faster because does not require opening the vlsv files

   for index in range(len(variables)):
      
      variables[variables.keys()[index]] = np.loadtxt(path+"/"+variables.keys()[index]+".txt")

   return variables





def read_from_vlsv(path, path_to_vlsv, use_points, points_in_RE, interpolate, points_for_interpolation, start, end, step):
#this reads the data from vlsv files along given points/cellids from given time and retunrs the values and also saves them in files so that next time you use these values you can use read_from_local_file instead, which is faster

   i=0
   cellids=[]
   cell_points=[]

   
   
   if use_points==1:
      vlsvReader = pt.vlsvfile.VlsvReader(path_to_vlsv+"bulk.000"+str(start).zfill(4)+".vlsv")
      for line in open(path+'/points.txt'): 
         points = line.split( )
      

         if points_in_RE==1:
            cell_points.append([float(points[0])*constants.RE(), float(points[1])*constants.RE(), float(points[2])*constants.RE()])
            cellids.append(vlsvReader.get_cellid([float(points[0])*constants.RE(), float(points[1])*constants.RE(), float(points[2])*constants.RE()]))
         else:
            cell_points.append([float(points[0]), float(points[1]), float(points[2])])
            cellids.append(vlsvReader.get_cellid([float(points[0]), float(points[1]), float(points[2])]))   
   else:
   
      for line in open(path+'/cellids.txt'): 
         cells = line.split( )
         cellids.append(int(float(cells[0])))

      
      vlsvReader = pt.vlsvfile.VlsvReader(path_to_vlsv+"bulk.000"+str(start).zfill(4)+".vlsv")   


      for _cellid in cellids:
         cell_points.append(vlsvReader.get_cell_coordinates(_cellid))
     
   
   coordinate_x=[]
   coordinate_y=[]
   coordinate_z=[]
   
   coordinate_x=np.array(cell_points)[:,0]
   coordinate_y=np.array(cell_points)[:,1]
   coordinate_z=np.array(cell_points)[:,2]


   t=start+step
   time=[start]
   filenames= [path_to_vlsv+"bulk.000"+str(start).zfill(4)+".vlsv"] 
      
   while t<end :
      #if t!=826: this is for ABA, this file number is broken
      filenames.extend([path_to_vlsv+"bulk.000"+str(t).zfill(4)+".vlsv"])
      time.append(t)
      t=t+step
      
      
   # LOOP OVER LIST OF CELLIDS, FOR EXAMPLE ALONG STREAMLINE
   

   
   if interpolate == 1:
      
      Bmag = [[] for i in range(len(time))]
      Bx=[[] for i in range(len(time))]
      By=[[] for i in range(len(time))]
      Bz=[[] for i in range(len(time))]
      density=[[] for i in range(len(time))]
      Vmag = [[] for i in range(len(time))]
      Vx=[[] for i in range(len(time))]
      Vy=[[] for i in range(len(time))]
      Vz=[[] for i in range(len(time))]
      T_perp=[[] for i in range(len(time))]
      T_par=[[] for i in range(len(time))]
      beta_perp=[[] for i in range(len(time))]
      beta_par=[[] for i in range(len(time))]
      pressure= [[] for i in range(len(time))]
      coordinates=[]

      


      for f in xrange(len(filenames)):
         print "Looping through timestep " + str(f)
         vlsvReader = pt.vlsvfile.VlsvReader(filenames[f])
         # Get a cut-through
         # Get cell ids and distances separately
         variables = []
         # Optimize file read:
         vlsvReader.optimize_open_file()
         #_cellid=vlsvReader.get_cellid(coordinates=point1)
      #  print "file read "+str(f)
         id=0
         
         for point_id in range(len(cell_points)-1):
            
            variable_mag = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='B', operator='magnitude', interpolation_order=1, points=points_for_interpolation )
            variable_X = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='B', operator='x', interpolation_order=1, points=points_for_interpolation )
            variable_Y = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='B', operator='y', interpolation_order=1, points=points_for_interpolation )          
            variable_Z = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='B', operator='z', interpolation_order=1, points=points_for_interpolation )        
            var_rho = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='rho', interpolation_order=1, points=points_for_interpolation )       
            var_v = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='v', operator='magnitude', interpolation_order=1, points=points_for_interpolation )        
            v_x = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='v', operator='x', interpolation_order=1, points=points_for_interpolation )        
            v_y = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='v', operator='y', interpolation_order=1, points=points_for_interpolation )         
            v_z = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='v', operator='z', interpolation_order=1, points=points_for_interpolation )        
            Tp = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='TPerpendicular', interpolation_order=1, points=points_for_interpolation )
            Tpa = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='TParallel', interpolation_order=1, points=points_for_interpolation )
            betap = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='betaPerpendicular', interpolation_order=1, points=points_for_interpolation )
            betapa = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='betaParallel', interpolation_order=1, points=points_for_interpolation )
            press= pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable='Pressure', interpolation_order=1, points=points_for_interpolation )

            if f==0:
              # print variable_mag[1][:(len(variable_mag[2])-1)]
               coordinates.extend(variable_mag[1][:(len(variable_mag[2])-1)])
               #coordinate_y.append(coords[1])
            

            Bmag[f].extend(variable_mag[2][:(len(variable_mag[2])-1)])
            Bx[f].extend(variable_X[2][:(len(variable_mag[2])-1)])
            By[f].extend(variable_Y[2][:(len(variable_mag[2])-1)]) 
            Bz[f].extend(variable_Z[2][:(len(variable_mag[2])-1)])
            density[f].extend(var_rho[2][:(len(variable_mag[2])-1)])
            Vmag[f].extend(var_v[2][:(len(variable_mag[2])-1)])
            Vx[f].extend(v_x[2][:(len(variable_mag[2])-1)])
            Vy[f].extend(v_y[2][:(len(variable_mag[2])-1)])
            Vz[f].extend(v_z[2][:(len(variable_mag[2])-1)])
            T_perp[f].extend(Tp[2][:(len(variable_mag[2])-1)])
            T_par[f].extend(Tpa[2][:(len(variable_mag[2])-1)])
            beta_perp[f].extend(betap[2][:(len(variable_mag[2])-1)])
            beta_par[f].extend(betapa[2][:(len(variable_mag[2])-1)])
            pressure[f].extend(press[2][:(len(variable_mag[2])-1)])
         
            #id=id+1

      # Save the distance between two successive cellids
      distance_between_cellids=[]
      distance=[]

      coordinate_x=np.array(coordinates)[:,0]
      coordinate_y=np.array(coordinates)[:,1]
      coordinate_z=np.array(coordinates)[:,2]
      
      for i in range(len(coordinate_x)):
         if i == 0 :
            distance_between_cellids.append(0)
            distance.append(0)
         if i>0 :
            distance_between_cellids.append(np.sqrt((coordinate_x[i]-coordinate_x[i-1])**2+(coordinate_y[i]-coordinate_y[i-1])**2+(coordinate_z[i]-coordinate_z[i-1])**2))
            distance.append(distance[i-1]+distance_between_cellids[i])
            

   
                     
   else:      
   
      distance=[0]
      
      B_mag = []
      Bx=[]
      By=[]
      Bz=[]
      E_mag = []
      Ex=[]
      Ey=[]
      Ez=[]
      number_density=[]
      v_mag = []
      vx=[]
      vy=[]
      vz=[]
      T=[]
      T_perp=[]
      T_par=[]
      beta_perp=[]
      beta_par=[]
      pressure= []
      

      for f in xrange(len(filenames)):
         print "Looping through timestep " + str(f)
         vlsvReader = pt.vlsvfile.VlsvReader(filenames[f])
         # Get a cut-through
         # Get cell ids and distances separately
         variables = []
         # Optimize file read:
         vlsvReader.optimize_open_file()
         #_cellid=vlsvReader.get_cellid(coordinates=point1)
      #  print "file read "+str(f)
         id=0

         
        # for _cellid in cellids:
         b_mag = vlsvReader.read_variable( name="B", cellids=cellids, operator="magnitude" )
         b_X = vlsvReader.read_variable( name="B", cellids=cellids, operator="x" )         
         b_Y = vlsvReader.read_variable( name="B", cellids=cellids, operator="y" )          
         b_Z = vlsvReader.read_variable( name="B", cellids=cellids, operator="z" )
         e_mag = vlsvReader.read_variable( name="E", cellids=cellids, operator="magnitude" )
         e_X = vlsvReader.read_variable( name="E", cellids=cellids, operator="x" )         
         e_Y = vlsvReader.read_variable( name="E", cellids=cellids, operator="y" )          
         e_Z = vlsvReader.read_variable( name="E", cellids=cellids, operator="z" )   
         var_rho = vlsvReader.read_variable( name="rho", cellids=cellids)         
         var_v = vlsvReader.read_variable( name="v", cellids=cellids, operator="magnitude")         
         v_x = vlsvReader.read_variable( name="v", cellids=cellids, operator="x")         
         v_y = vlsvReader.read_variable( name="v", cellids=cellids, operator="y")         
         v_z = vlsvReader.read_variable( name="v", cellids=cellids, operator="z")         
         Tem = vlsvReader.read_variable( name="Temperature", cellids=cellids)
         Tp = vlsvReader.read_variable( name="TPerpendicular", cellids=cellids)
         Tpa = vlsvReader.read_variable( name="TParallel", cellids=cellids)
         betap = vlsvReader.read_variable( name="betaPerpendicular", cellids=cellids)
         betapa = vlsvReader.read_variable( name="betaParallel", cellids=cellids)
         press= vlsvReader.read_variable( name="Pressure", cellids=cellids)
            
         B_mag.append(b_mag)
         Bx.append(b_X)
         By.append(b_Y)
         Bz.append(b_Z)
         E_mag.append(e_mag)
         Ex.append(e_X)
         Ey.append(e_Y)
         Ez.append(e_Z)
         number_density.append(var_rho)
         v_mag.append(var_v)
         vx.append(v_x)
         vy.append(v_y)
         vz.append(v_z)
         T.append(Tem)
         T_perp.append(Tp)
         T_par.append(Tpa)
         beta_perp.append(betap)
         beta_par.append(betapa)
         pressure.append(press)
               
          #  id=id+1


      distance_between_cellids=[]
      for i in range(len(coordinate_x)):
         if i == 0 :
            distance_between_cellids.append(0)
            distance.append(0)
         if i>0 :
            distance_between_cellids.append(np.sqrt((coordinate_x[i]-coordinate_x[i-1])**2+(coordinate_y[i]-coordinate_y[i-1])**2+(coordinate_z[i]-coordinate_z[i-1])**2))
            distance.append(distance[i-1]+distance_between_cellids[i])
      
      
   time=np.array(time)
   distance=np.array(distance)
   
   Bmag=np.array(B_mag).T
   Bx=np.array(Bx).T
   By=np.array(By).T
   Bz=np.array(Bz).T
   Emag=np.array(B_mag).T
   Ex=np.array(Bx).T
   Ey=np.array(By).T
   Ez=np.array(Bz).T
   density=np.array(number_density).T
   Vmag=np.array(v_mag).T
   Vx=np.array(vx).T
   Vy=np.array(vy).T
   Vz=np.array(vz).T
   T=np.array(T).T
   T_perp=np.array(T_perp).T
   T_par=np.array(T_par).T
   beta_perp=np.array(beta_perp).T
   beta_par=np.array(beta_par).T
   pressure=np.array(pressure).T
   

   # REMOVE old files if they exist
   functions.silentremove(path+"/cellids_array")
   functions.silentremove(path+"/B_array")
   functions.silentremove(path+"/Bx_array")
   functions.silentremove(path+"/By_array")
   functions.silentremove(path+"/Bz_array")
   functions.silentremove(path+"/E_array")
   functions.silentremove(path+"/Ex_array")
   functions.silentremove(path+"/Ey_array")
   functions.silentremove(path+"/Ez_array")
   functions.silentremove(path+"/density_array")
   functions.silentremove(path+"/V_array")
   functions.silentremove(path+"/Vx_array")
   functions.silentremove(path+"/Vy_array")
   functions.silentremove(path+"/Vz_array")
   functions.silentremove(path+"/T_array")
   functions.silentremove(path+"/Tperp_array")
   functions.silentremove(path+"/Tpar_array")
   functions.silentremove(path+"/betaperp_array")
   functions.silentremove(path+"/betapar_array")
   functions.silentremove(path+"/pressure_array")
   functions.silentremove(path+"/time_array")
   functions.silentremove(path+"/distance_array")
   functions.silentremove(path+"/coordinate_x")
   functions.silentremove(path+"/coordinate_y")
   functions.silentremove(path+"/coordinate_z")
   
   
   #Save new files
   
   np.savetxt(path+"/cellids_array", cellids)
   np.savetxt(path+"/B_array", Bmag)
   np.savetxt(path+"/Bx_array", Bx)
   np.savetxt(path+"/By_array", By)
   np.savetxt(path+"/Bz_array", Bz)
   np.savetxt(path+"/E_array", Emag)
   np.savetxt(path+"/Ex_array", Ex)
   np.savetxt(path+"/Ey_array", Ey)
   np.savetxt(path+"/Ez_array", Ez)
   np.savetxt(path+"/rho_array", density)
   np.savetxt(path+"/v_array", Vmag)
   np.savetxt(path+"/vx_array", Vx)
   np.savetxt(path+"/vy_array", Vy)
   np.savetxt(path+"/vz_array", Vz)
   np.savetxt(path+"/T_array", T)
   np.savetxt(path+"/Tperp_array", T_perp)
   np.savetxt(path+"/Tpar_array", T_par)
   np.savetxt(path+"/betaperp_array", beta_perp)
   np.savetxt(path+"/betapar_array", beta_par)
   np.savetxt(path+"/pressure_array", pressure)
   np.savetxt(path+"/time_array", time)
   np.savetxt(path+"/distance_array", distance)
   np.savetxt(path+"/coordinate_x", coordinate_x)
   np.savetxt(path+"/coordinate_y", coordinate_y)
   np.savetxt(path+"/coordinate_z", coordinate_z)
   
   
   
   return {'cellids':cellids, 'Bmag':Bmag, 'Bx':Bx, 'By':By, 'Bz':Bz, 'Emag':Emag, 'Ex':Ex, 'Ey':Ey, 'Ez':Ez, 'density':density, 'Vmag':vmag, 'Vx':vx, 'Vy':vy, 'Vz':vz, 'T':T, 'T_perp':T_perp, 'Tpar':Tpar, 'beta_perp':beta_perp, 'beta_par':beta_par, 'pressure':pressure, 'time':time, 'coordinate_x':coordinate_x, 'coordinate_y':coordinate_y, 'coordinate_z':coordinate_z}





def read_from_vlsv_dictionary(path, path_to_vlsv, variables, use_points, points_in_RE, interpolate, points_for_interpolation, start, end, step):
   #this reads the data from vlsv files along given points/cellids from given time and retunrs the values and also saves them in files so that next time you use these values you can use read_from_local_file instead, which is faster
  
   cellids=[]
   cell_points=[]

   
   
   if use_points==1:
      vlsvReader = pt.vlsvfile.VlsvReader(path_to_vlsv+"bulk.000"+str(start).zfill(4)+".vlsv")
      for line in open(path+'/points.txt'): 
         points = line.split( )
      

         if points_in_RE==1:
            cell_points.append([float(points[0])*constants.RE(), float(points[1])*constants.RE(), float(points[2])*constants.RE()])
            cellids.append(vlsvReader.get_cellid([float(points[0])*constants.RE(), float(points[1])*constants.RE(), float(points[2])*constants.RE()]))
         else:
            cell_points.append([float(points[0]), float(points[1]), float(points[2])])
            cellids.append(vlsvReader.get_cellid([float(points[0]), float(points[1]), float(points[2])]))   
   else:
   
      for line in open(path+'/cellids.txt'): 
         cells = line.split( )
         cellids.append(int(float(cells[0])))

      
      vlsvReader = pt.vlsvfile.VlsvReader(path_to_vlsv+"bulk.000"+str(start).zfill(4)+".vlsv")   


      for _cellid in cellids:
         cell_points.append(vlsvReader.get_cell_coordinates(_cellid))
     
   
   coordinate_x=[]
   coordinate_y=[]
   coordinate_z=[]
   
   coordinate_x=np.array(cell_points)[:,0]
   coordinate_y=np.array(cell_points)[:,1]
   coordinate_z=np.array(cell_points)[:,2]


   t=start+step
   time=[start]
   filenames= [path_to_vlsv+"bulk.000"+str(start).zfill(4)+".vlsv"] 
      
   while t<end :
      #if t!=826: this is for ABA, this file number is broken
      filenames.extend([path_to_vlsv+"bulk.000"+str(t).zfill(4)+".vlsv"])
      time.append(t)
      t=t+step
      
      
   # LOOP OVER LIST OF CELLIDS, FOR EXAMPLE ALONG STREAMLINE
   

   
   if interpolate == 1:
      
      for index in range(len(variables)):
         variables[variables.keys()[index]] = []# [[] for i in range(len(time))]
      
  
      coordinates=[]
      
      
      for f in xrange(len(filenames)):
         print "Looping through timestep " + str(f)
         vlsvReader = pt.vlsvfile.VlsvReader(filenames[f])
         # Optimize file read:
         vlsvReader.optimize_open_file()
         #_cellid=vlsvReader.get_cellid(coordinates=point1)
      #  print "file read "+str(f)
   

         
         for point_id in range(len(cell_points)-1):
            
            for index in range(len(variables)):
               variable = pt.calculations.lineout(vlsvReader, point1=cell_points[point_id], point2=cell_points[point_id+1], variable=variables.keys()[index], interpolation_order=1, points=points_for_interpolation )
   
               variables[variables.keys()[index]].extend(variable[2][:(len(variable[2])-1)])
               
               
               
            if f==0:
            # print variable_mag[1][:(len(variable_mag[2])-1)]
               coordinates.extend(variable[1][:(len(variable[2])-1)])
               #coordinate_y.append(coords[1])
            

   
      coordinate_x=np.array(coordinates)[:,0]
      coordinate_y=np.array(coordinates)[:,1]
      coordinate_z=np.array(coordinates)[:,2]
         
  
                     
   else:      
   
      distance=[0]
      
      
      for index in range(len(variables)):
         variables[variables.keys()[index]] = []
      
      
      for f in xrange(len(filenames)):
         print "Looping through timestep " + str(f)
         vlsvReader = pt.vlsvfile.VlsvReader(filenames[f])

         # Optimize file read:
         vlsvReader.optimize_open_file()

         for index in range(len(variables)):
            variable = vlsvReader.read_variable( name=variables.keys()[index] , cellids=cellids)
            
            variables[variables.keys()[index]].append(variable)
            

      # Now save the files and return them
   original_variables = copy.deepcopy(variables)
   for index in range(len(original_variables)):

      variables[original_variables.keys()[index]] = np.array(original_variables.values()[index]).T
   
      #save new ones
      if len(variables[original_variables.keys()[index]].shape) > 2: 
         #vectors as components
         variables[original_variables.keys()[index]+'x'] = variables[original_variables.keys()[index]][0]
         variables[original_variables.keys()[index]+'y'] = variables[original_variables.keys()[index]][1]
         variables[original_variables.keys()[index]+'z'] = variables[original_variables.keys()[index]][2]
         del variables[original_variables.keys()[index]]
   
   
   time=np.array(time)
   cellids=np.array(cellids)
   variables['coordinate_x']=coordinate_x
   variables['coordinate_y']=coordinate_y
   variables['coordinate_z']=coordinate_z
   variables['time']=time
   variables['cellids']=cellids
   
   print variables.values()
   print variables.keys()
   for index in range(len(variables)):
      print variables.values()[index]
      
      functions.silentremove(path+"/"+variables.keys()[index]+".txt")
      np.savetxt(path+"/"+variables.keys()[index]+".txt",variables.values()[index])
         
   
   
   return variables
