#!/bin/bash -f
#$ -N Barents_AICE_grid
#$ -l h_rt=00:05:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=2G,mem_free=2G,h_data=2G
#$ -q research-r8.q
#$ -t 1-366
##$ -j y
##$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y


source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate production-08-2024

echo "Got $NSLOTS slots for job $SGE_TASK_ID."

cat > "/home/cyrilp/Documents/PROG/PCAPS_Barents_reggriding_""$SGE_TASK_ID"".py" << EOF
###################################################################################################
#!/usr/bin/env python
# coding: utf-8

# In[127]:


import time
import os
import netCDF4
import numpy as np
import scipy
import pyproj
import datetime
import matplotlib.pyplot as plt


# # Constants

# In[128]:


#
date_min = "20240401"
date_max = "20250331"
#
N_Barents_members = 6
#
paths = {}
paths["Barents"] = "/lustre/storeB/project/fou/hi/oper/barents_eps/archive/surface/"
paths["AICE"] = "/lustre/storeB/project/copernicus/cosi/AICE/archive/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/PCAPS/Barents_AICE_grid/"
#
crs = {}
crs["latlon"] = pyproj.CRS.from_proj4("+proj=latlon")
crs["Barents"] = pyproj.CRS.from_proj4("+proj=lcc +lat_0=77.5 +lon_0=-25 +lat_1=77.5 +lat_2=77.5 +no_defs +R=6.371e+06")
crs["AICE"] = pyproj.CRS.from_proj4("+ellps=WGS84 +lat_0=90 +lon_0=0 +no_defs=None +proj=laea +type=crs +units=m +x_0=0 +y_0=0")
#
coordinate_variables = ["X", "Y", "lat", "lon"]
ice_variables = ["ice_concentration", "ice_thickness", "drift_speed", "drift_direction"]


# # List dates

# In[129]:


def make_list_dates(date_min, date_max):
    current_date = datetime.datetime.strptime(date_min, "%Y%m%d")
    end_date = datetime.datetime.strptime(date_max, "%Y%m%d")
    list_dates = []
    while current_date <= end_date:
        date_str = current_date.strftime('%Y%m%d')
        list_dates.append(date_str)
        current_date = current_date + datetime.timedelta(days = 1)
    return(list_dates)


# # Load datasets

# In[130]:


def load_AICE_grid(date_task, paths):
    AICE_coordinates = {}
    filename = paths["AICE"] + "AICE_forecasts_" + date_task + "T000000Z.nc"
    nc = netCDF4.Dataset(filename, "r")
    AICE_coordinates["x"] = nc.variables["x"][:]
    AICE_coordinates["y"] = nc.variables["y"][:]
    AICE_coordinates["lat"] = nc.variables["lat"][:,:]
    AICE_coordinates["lon"] = nc.variables["lon"][:,:]
    nc.close()
    return(AICE_coordinates)


# In[131]:


class load_Barents_data():
    def __init__(self, date_task, paths, N_Barents_members, crs, coordinate_variables, ice_variables):
        self.date_task = date_task
        self.paths = paths
        self.N_Barents_members = N_Barents_members
        self.crs = crs
        self.init_time = "T00Z" 
        self.coordinate_variables = coordinate_variables
        self.ice_variables = ice_variables
        #
        Barents_ice_variables = self.ice_variables.copy()
        idx_drift_speed = self.ice_variables.index("drift_speed")
        idx_drift_direction = self.ice_variables.index("drift_direction")
        Barents_ice_variables[idx_drift_speed] = "ice_u"
        Barents_ice_variables[idx_drift_direction] = "ice_v"
        self.Barents_ice_variables = Barents_ice_variables
#
    def calculate_initial_compass_bearing(self, lat1, lon1, lat2, lon2):
            lat1_rad = np.radians(lat1)
            lat2_rad = np.radians(lat2)
            diff_lon = np.radians(lon2 - lon1)
            xbear = np.sin(diff_lon) * np.cos(lat2_rad)
            ybear = np.cos(lat1_rad) * np.sin(lat2_rad) - (np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(diff_lon))
            initial_bearing = np.degrees(np.arctan2(xbear, ybear))
            compass_bearing = (initial_bearing + 360) % 360
            return compass_bearing
    #
    def great_circle_distance(self, lon1, lat1, lon2, lat2):
            # Convert from degrees to radians
            pi = 3.14159265
            lon1 = lon1 * 2 * pi / 360.
            lat1 = lat1 * 2 * pi / 360.
            lon2 = lon2 * 2 * pi / 360.
            lat2 = lat2 * 2 * pi / 360.
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2.) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6.371e6 * c
            return distance
    #
    def load_Barents(self):
        Daily_dataset = {}
        for em in range(0, self.N_Barents_members):
            member = "{:02d}".format(em)
            path_data = self.paths["Barents"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/" + self.date_task[6:8] + "/" + self.init_time + "/"
            filename = path_data + "barents_sfc_" + self.date_task + self.init_time + "m" + member + ".nc"
            with netCDF4.Dataset(filename, "r") as nc:
    
                if member == "00":
                    Daily_dataset["sea_mask"] = nc.variables["sea_mask"][:,:]
                    for var in self.coordinate_variables:
                        Daily_dataset[var] = nc.variables[var][:]
            
                for var in self.Barents_ice_variables:
                    Daily_dataset[member + "_" + var] = np.full((4, len(Daily_dataset["Y"]), len(Daily_dataset["X"])), np.nan) 
                    for ts in range(0, 4):
                        ts_start = ts * 24
                        ts_end = ts_start + 24
                        Daily_dataset[member + "_" + var][ts,:,:] = np.nanmean(nc.variables[var][ts_start:ts_end,:,:], axis = 0)
                    
                    if var == "ice_concentration":
                        Daily_dataset[member + "_" + var + "_first_hour"] = np.copy(nc.variables[var][0,:,:])
                        Daily_dataset[member + "_" + var + "_first_hour"][nc.variables[var][0,:,:] > 1e10] = np.nan

        return Daily_dataset
    #
    def sea_ice_drift(self, Daily_dataset):
        transform = pyproj.Transformer.from_crs(self.crs["Barents"], self.crs["latlon"], always_xy = True)
        xx_start, yy_start = np.meshgrid(Daily_dataset["X"], Daily_dataset["Y"])
        for em in range(0, 6):
            member = "{:02d}".format(em)
            xx_end = xx_start + Daily_dataset[member + "_" + "ice_u"] * 24 * 60 * 60  # m.s-1 to m.day-1
            yy_end = yy_start + Daily_dataset[member + "_" + "ice_v"] * 24 * 60 * 60
            lon_end, lat_end = transform.transform(xx_end, yy_end)
            Daily_dataset[member + "_drift_speed"] = self.great_circle_distance(Daily_dataset["lon"], Daily_dataset["lat"], lon_end, lat_end)
            Daily_dataset[member + "_drift_direction"] = self.calculate_initial_compass_bearing(Daily_dataset["lat"], Daily_dataset["lon"], lat_end, lon_end)
            #
            sea_mask = np.repeat(np.expand_dims(Daily_dataset["sea_mask"], axis = 0), 4, axis = 0)
            Daily_dataset[member + "_drift_speed"][sea_mask < 0.1] = np.nan
            Daily_dataset[member + "_drift_speed"][Daily_dataset[member + "_ice_concentration"] < 0.01] = np.nan
            Daily_dataset[member + "_drift_direction"][Daily_dataset[member + "_ice_concentration"] < 0.01] = np.nan
            del Daily_dataset[member + "_" + "ice_u"]
            del Daily_dataset[member + "_" + "ice_v"]
        return Daily_dataset 
    #
    def __call__(self):
        Daily_dataset = self.load_Barents()
        Daily_dataset = self.sea_ice_drift(Daily_dataset)
        return Daily_dataset


# # Reggriding

# In[132]:


class nearest_neighbor_interpolation_2D():
    def __init__(self, AICE_coordinates, Barents_dataset, N_Barents_members, crs, coordinate_variables):
        self.AICE_coordinates = AICE_coordinates
        self.Barents_dataset = Barents_dataset
        self.N_Barents_members = N_Barents_members
        self.crs = crs
        self.coordinate_variables = coordinate_variables
    #
    def make_padding(self, x, y, field):
        dx = x[1] - x[0]
        x_extent = np.pad(x, (1, 1), constant_values = np.nan)    
        x_extent[0] = x_extent[1] - dx
        x_extent[-1] = x_extent[-2] + dx
        #
        dy = y[1] - y[0]
        y_extent = np.pad(y, (1, 1), constant_values = np.nan)
        y_extent[0] = y_extent[1] - dy
        y_extent[-1] = y_extent[-2] + dy
        #
        if field.ndim == 2:
            field_extent = np.pad(field, (1,1), constant_values = np.nan)
        elif field.ndim == 3:
            time_dim = len(field[:,0,0])
            field_extent = np.full((time_dim, len(y_extent), len(x_extent)), np.nan)
            #
            for t in range(0, time_dim):
                field_extent[t,:,:] = np.pad(field[t,:,:], (1,1), constant_values = np.nan)
        #
        return(x_extent, y_extent, field_extent)
    #
    def nearest_neighbor_indexes(self, x_input, y_input, x_output, y_output):
        x_input = np.expand_dims(x_input, axis = 1)
        y_input = np.expand_dims(y_input, axis = 1)
        x_output = np.expand_dims(x_output, axis = 1)
        y_output = np.expand_dims(y_output, axis = 1)
        #
        coord_input = np.concatenate((x_input, y_input), axis = 1)
        coord_output = np.concatenate((x_output, y_output), axis = 1)
        #
        tree = scipy.spatial.KDTree(coord_input)
        dist, idx = tree.query(coord_output)
        #
        return idx
    #
    def extract_idx(self, Barents_padded_dataset):
        transform = pyproj.Transformer.from_crs(self.crs["Barents"], self.crs["AICE"], always_xy = True)
        #
        xx_input, yy_input = np.meshgrid(Barents_padded_dataset["X"], Barents_padded_dataset["Y"])
        xx_output, yy_output = np.meshgrid(self.AICE_coordinates["x"], self.AICE_coordinates["y"])
        #
        xx_input_data_on_output_proj, yy_input_data_on_output_proj = transform.transform(xx_input, yy_input)
        #
        xx_input_data_on_output_proj_flat = np.ndarray.flatten(xx_input_data_on_output_proj)
        yy_input_data_on_output_proj_flat = np.ndarray.flatten(yy_input_data_on_output_proj)
        xx_output_flat = np.ndarray.flatten(xx_output)
        yy_output_flat = np.ndarray.flatten(yy_output)
        #
        idx = self.nearest_neighbor_indexes(xx_input_data_on_output_proj_flat, yy_input_data_on_output_proj_flat, xx_output_flat, yy_output_flat)
        #
        return idx
    #
    def nearest_neighbor_interpolation(self):
        Barents_on_AICE_grid = {}
        #
        for vi, var in enumerate(self.Barents_dataset):
            if var in self.coordinate_variables:
                if var == "X":
                    var = "x"
                elif var == "Y":
                    var = "y"
                Barents_on_AICE_grid[var] = np.copy(self.AICE_coordinates[var])
            else:
                Barents_padded_dataset = {}
                Barents_padded_dataset["X"], Barents_padded_dataset["Y"], Barents_padded_dataset[var] = self.make_padding(self.Barents_dataset["X"], self.Barents_dataset["Y"], self.Barents_dataset[var]) 
                #
                if "idx" not in locals():
                    idx = self.extract_idx(Barents_padded_dataset)
                #
                if (var == "sea_mask") or ("ice_concentration_first_hour" in var):
                    field_flat = np.ndarray.flatten(Barents_padded_dataset[var][:,:])
                    field_interp = field_flat[idx]
                    field_regrid = np.reshape(field_interp, (len(self.AICE_coordinates["y"]), len(self.AICE_coordinates["x"])), order = "C")
                else:
                    time_dim = len(Barents_padded_dataset[var][:,0,0])
                    field_regrid = np.full((time_dim, len(self.AICE_coordinates["y"]), len(self.AICE_coordinates["x"])), np.nan)
                    for t in range(0, time_dim):
                        field_flat = np.ndarray.flatten(Barents_padded_dataset[var][t,:,:])
                        field_interp = field_flat[idx]
                        field_regrid[t,:,:] = np.reshape(field_interp, (len(self.AICE_coordinates["y"]), len(self.AICE_coordinates["x"])), order = "C")
                Barents_on_AICE_grid[var] = np.copy(field_regrid)
        #
        return(Barents_on_AICE_grid)
    #
    def __call__(self):
        Barents_on_AICE_grid = self.nearest_neighbor_interpolation()
        return Barents_on_AICE_grid


# # Write output netCDF file

# In[133]:


class write_netCDF_file():
    def __init__(self, Barents_on_AICE_grid, crs, coordinate_variables, ice_variables, N_Barents_members, date_task, paths):
        self.Barents_on_AICE_grid = Barents_on_AICE_grid
        self.crs = crs
        self.coordinate_variables = coordinate_variables
        self.ice_variables = ice_variables
        self.N_Barents_members = N_Barents_members
        self.date_task = date_task
        self.paths = paths
    #
    def concatenate_fields(self):
        Dataset_all_members = {}
        #
        for var in self.ice_variables:
            dim_t, dim_y, dim_x = np.shape(self.Barents_on_AICE_grid["00_" + var])
            Dataset_all_members[var] = np.full((self.N_Barents_members, dim_t, dim_y, dim_x), np.nan)
            #
            for em in range(0, self.N_Barents_members):
                member = "{:02d}".format(em)
                Dataset_all_members[var][em,:,:,:] = self.Barents_on_AICE_grid[member + "_" + var]
                
        Dataset_all_members["ice_concentration_first_hour"] = np.full((self.N_Barents_members, 1, dim_y, dim_x), np.nan)
        for em in range(0, self.N_Barents_members):
            member = "{:02d}".format(em)
            Dataset_all_members["ice_concentration_first_hour"][em,:,:,:] = np.copy(self.Barents_on_AICE_grid[member + "_ice_concentration_first_hour"])
        #
        return(Dataset_all_members, dim_t, dim_y, dim_x)
    #
    def write_netCDF(self):
        Dataset_all_members, dim_t, dim_y, dim_x = self.concatenate_fields()
        path_output = self.paths["output"] + self.date_task[0:4] + "/" + self.date_task[4:6] + "/"
        if os.path.exists(path_output) == False:
            os.system("mkdir -p " + path_output)   
        output_filename = path_output + "Barents_on_AICE_grid_" + self.date_task + ".nc"
        if os.path.isfile(output_filename):
            os.system("rm " + output_filename)
        #
        with netCDF4.Dataset(str(output_filename), "w", format = "NETCDF4") as output_netcdf:
            proj4 = output_netcdf.createDimension("proj4", 1)
            first_hour = output_netcdf.createDimension("first_hour", 1)
            member = output_netcdf.createDimension("member", self.N_Barents_members)
            time = output_netcdf.createDimension("time", dim_t)
            x = output_netcdf.createDimension("x", dim_x)
            y = output_netcdf.createDimension("y", dim_y)
            #
            Outputs = vars()
            #
            Outputs["proj4"] = output_netcdf.createVariable("proj4", "S1", ("proj4"))
            Outputs["proj4"].long_name = self.crs["AICE"].to_proj4()
            #
            Outputs["member"] = output_netcdf.createVariable("member", "d", ("member"))
            Outputs["member"][:] = np.arange(self.N_Barents_members)
            Outputs["member"].long_name = "ensemble member ID"
            #
            Outputs["time"] = output_netcdf.createVariable("time", "d", ("time"))
            Outputs["time"][:] = [0, 24, 48, 72]
            Outputs["time"].units = "hours"
            Outputs["time"].long_name = "lead time in hours"
            #
            Outputs["first_hour"] = output_netcdf.createVariable("first_hour", "d", ("first_hour"))
            Outputs["first_hour"][:] = [0]
            Outputs["first_hour"].units = "hours"
            Outputs["first_hour"].long_name = "lead time in hours"
            #
            Outputs["sea_mask"] = output_netcdf.createVariable("sea_mask", "d", ("y", "x"))
            Outputs["sea_mask"][:,:] = self.Barents_on_AICE_grid["sea_mask"]
            Outputs["sea_mask"].units = "fraction of sea"
            Outputs["sea_mask"].long_name = "sea land mask (0: land, 1: sea)"
            #
            Outputs["ice_concentration_first_hour"] = output_netcdf.createVariable("ice_concentration_first_hour", "d", ("member", "first_hour", "y", "x"))
            Outputs["ice_concentration_first_hour"][:,:,:,:] = Dataset_all_members["ice_concentration_first_hour"]
            #
            for var in self.coordinate_variables:
                if var == "X":
                    Outputs["x"] = output_netcdf.createVariable("x", "d", ("x"))
                    Outputs["x"][:] = self.Barents_on_AICE_grid["x"]
                elif var == "Y":
                    Outputs["y"] = output_netcdf.createVariable("y", "d", ("y"))
                    Outputs["y"][:] = self.Barents_on_AICE_grid["y"]
                elif self.Barents_on_AICE_grid[var].ndim == 2:
                    Outputs[var] = output_netcdf.createVariable(var, "d", ("y", "x"))
                    Outputs[var][:,:] = self.Barents_on_AICE_grid[var]
            for var in self.ice_variables:
                Outputs[var] = output_netcdf.createVariable(var, "d", ("member", "time", "y", "x"))
                Outputs[var][:,:,:,:] = Dataset_all_members[var]
    #
    def __call__(self):
        self.write_netCDF()


# # Data processing

# In[134]:


t0 = time.time()
list_dates = make_list_dates(date_min, date_max)
date_task = list_dates[$SGE_TASK_ID - 1]
AICE_coordinates = load_AICE_grid(date_task, paths)
Barents_dataset = load_Barents_data(date_task, paths, N_Barents_members, crs, coordinate_variables, ice_variables)()
Barents_on_AICE_grid = nearest_neighbor_interpolation_2D(AICE_coordinates, Barents_dataset, N_Barents_members, crs, coordinate_variables)()
write_netCDF_file(Barents_on_AICE_grid, crs, coordinate_variables, ice_variables, N_Barents_members, date_task, paths)()
tf = time.time()
print("Computing time: ", tf - t0)
###################################################################################################
EOF
python3 "/home/cyrilp/Documents/PROG/PCAPS_Barents_reggriding_""$SGE_TASK_ID"".py"
