#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import time
import scipy
import pyproj
import netCDF4
import datetime
import numpy as np


# In[16]:


date_task = "20250408"
#
paths = {}
paths["ice_charts"] = "/lustre/storeB/project/metkl/istjenesten/Icecharts/ice_conc_gridded_nc/"
paths["AICE_training"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/Training/"
paths["output"] = "/lustre/storeB/project/copernicus/cosi/WP3/Operational/Ice_charts/"
#
proj4_str = {}
proj4_str["ice_charts"] = "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=sphere +units=m +no_defs +type=crs"
proj4_str["AICE"] = "+ellps=WGS84 +lat_0=90 +lon_0=0 +no_defs=None +proj=laea +type=crs +units=m +x_0=0 +y_0=0"


# In[17]:


def load_AICE_coordinates(paths, date_static = "20220101"):
    Data = {}
    filename = paths["AICE_training"] + date_static[0:4] + "/" + date_static[4:6] + "/" + "Dataset_" + date_static + ".nc"
    nc = netCDF4.Dataset(filename, "r")
    Data["x"] = nc.variables["x"][:]
    Data["y"] = nc.variables["y"][:]
    Data["lat"] = nc.variables["lat"][:,:]
    Data["lon"] = nc.variables["lon"][:,:]
    Data["LSM"] = nc.variables["LSM"][:,:]
    nc.close()
    return(Data)


# In[18]:


class extract_ice_charts():
    def __init__(self, paths, date_task):
        self.paths = paths
        self.date_task = date_task
        
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

    def load_ice_chart_data(self):
        Data = {}
        filename = self.paths["ice_charts"] + "ice_conc_svalbard_" + self.date_task + "1500.nc"
        nc = netCDF4.Dataset(filename, "r")
        x = nc.variables["xc"][:]
        y = nc.variables["yc"][:]
        SIC = nc.variables["ice_concentration"][0,:,:]
        Data["x"], Data["y"], Data["SIC"] = self.make_padding(x, y, SIC)
        LSM_field = np.zeros(np.shape(Data["SIC"]))
        LSM_field[Data["SIC"] > -90] = 1
        Data["LSM"] = np.copy(LSM_field)
        nc.close()
        return(Data)


# In[19]:


class nearest_neighbor_interpolation_2D():
    def __init__(self, input_dataset_name, output_dataset_name, input_dataset, output_dataset, proj4_str):
        self.input_dataset_name = input_dataset_name
        self.output_dataset_name = output_dataset_name
        self.input_dataset = input_dataset # must be a dictionary containing x and y coordinates, the 2D fields
        self.output_dataset = output_dataset # must be a dictionary containing all the coordinates (x and y, but also time, lat, lon if they exist in the dataset) and the land sea mask (LSM)
        self.proj4_str = proj4_str 
        self.crs_input = pyproj.CRS.from_proj4(proj4_str[input_dataset_name])
        self.crs_output = pyproj.CRS.from_proj4(proj4_str[output_dataset_name])
        
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
        return(idx)
    
    def extract_idx(self):
        transform_input_to_output = pyproj.Transformer.from_crs(self.crs_input, self.crs_output, always_xy = True)
        #
        xx_input, yy_input = np.meshgrid(self.input_dataset["x"], self.input_dataset["y"])
        xx_output, yy_output = np.meshgrid(self.output_dataset["x"], self.output_dataset["y"])
        #
        xx_input_data_on_output_proj, yy_input_data_on_output_proj = transform_input_to_output.transform(xx_input, yy_input)
        #
        xx_input_data_on_output_proj_flat = np.ndarray.flatten(xx_input_data_on_output_proj)
        yy_input_data_on_output_proj_flat = np.ndarray.flatten(yy_input_data_on_output_proj)
        xx_output_flat = np.ndarray.flatten(xx_output)
        yy_output_flat = np.ndarray.flatten(yy_output)
        #
        idx = self.nearest_neighbor_indexes(xx_input_data_on_output_proj_flat, yy_input_data_on_output_proj_flat, xx_output_flat, yy_output_flat)
        #
        return(idx)
    
    def nearest_neighbor_interp(self):
        idx = self.extract_idx()
        #
        interp_datasets = {}
        dim_fields = ["time", "x", "y", "lat", "lon", "latitude", "longitude"]
        for var in self.input_dataset:
            if var not in dim_fields:
                interp_datasets[var] = np.ndarray.flatten(self.input_dataset[var])[idx]
        #
        Data_output_grid = {}
        for var in self.output_dataset:
            if var in dim_fields:
                Data_output_grid[var] = np.copy(self.output_dataset[var])
        #
        for var in interp_datasets:
            Data_output_grid[var] = np.reshape(interp_datasets[var], (len(self.output_dataset["y"]), len(self.output_dataset["x"])), order = "C")
        #
        for var in Data_output_grid:
            if (var not in dim_fields) and (var != "LSM"):
                Data_output_grid[var][Data_output_grid["LSM"] == 0] = np.nan
        #
        return(Data_output_grid)


# In[20]:


def write_netCDF_ice_charts_on_AICE_grid(date_task, Datasets, paths):
    timestamp = datetime.datetime.strptime(date_task, "%Y%m%d").timestamp()
    #
    path_output = paths["output"] + date_task[0:4] + "/" + date_task[4:6] + "/" 
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)
    output_filename = path_output + "Ice_charts_AICE_grid_" + date_task + ".nc"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    output_netcdf = netCDF4.Dataset(output_filename, "w", format = "NETCDF4")
    #
    time = output_netcdf.createDimension("time", 1)
    x = output_netcdf.createDimension("x", len(Datasets["Domain_data"]["x"]))
    y = output_netcdf.createDimension("y", len(Datasets["Domain_data"]["y"]))
    #
    Lambert_Azimuthal_Grid = output_netcdf.createVariable("Lambert_Azimuthal_Grid", "d")
    time = output_netcdf.createVariable("time", "d", ("time"))
    x = output_netcdf.createVariable("x", "d", ("x"))
    y = output_netcdf.createVariable("y", "d", ("y"))
    lat = output_netcdf.createVariable("lat", "d", ("y", "x"))
    lon = output_netcdf.createVariable("lon", "d", ("y", "x"))
    SIC = output_netcdf.createVariable("SIC", "d", ("time", "y", "x"))
    #
    Lambert_Azimuthal_Grid.grid_mapping_name = "lambert_azimuthal_equal_area"
    Lambert_Azimuthal_Grid.semi_major_axis = 6378137
    Lambert_Azimuthal_Grid.semi_minor_axis = 6356752.31424518
    Lambert_Azimuthal_Grid.reference_ellipsoid_name = "WGS 84"
    Lambert_Azimuthal_Grid.longitude_of_prime_meridian = "0.0"
    Lambert_Azimuthal_Grid.prime_meridian_name = "Greenwich"
    Lambert_Azimuthal_Grid.geographic_crs_name = "unknown"
    Lambert_Azimuthal_Grid.horizontal_datum_name = "Unknown based on WGS84 ellipsoid"
    Lambert_Azimuthal_Grid.projected_crs_name = "unknown"
    Lambert_Azimuthal_Grid.latitude_of_projection_origin = 90.0
    Lambert_Azimuthal_Grid.longitude_of_projection_origin = 0.0
    Lambert_Azimuthal_Grid.false_easting = 0.0
    Lambert_Azimuthal_Grid.false_northing = 0.0
    Lambert_Azimuthal_Grid.proj4_string = "+ellps=WGS84 +lat_0=90 +lon_0=0 +no_defs=None +proj=laea +type=crs +units=m +x_0=0 +y_0=0"
    time.standard_name = "time"
    time.units = "seconds since 1970-01-01 00:00:00 +0000"
    x.standard_name = "projection_x_coordinate"
    x.units = "m"
    y.standard_name = "projection_y_coordinate"
    y.units = "m"
    lat.standard_name = "latitude"
    lat.units = "degrees_north"
    lon.standard_name = "longitude"
    lon.units = "degrees_east"
    SIC.standard_name = "Sea ice concentration from AMSR2 product"
    SIC.units = "%"
    #
    time[:] = timestamp
    x[:] = np.copy(Datasets["Domain_data"]["x"])
    y[:] = np.copy(Datasets["Domain_data"]["y"])
    lat[:,:] = np.copy(Datasets["Domain_data"]["lat"])
    lon[:,:] = np.copy(Datasets["Domain_data"]["lon"])
    SIC[:,:,:] = np.copy(np.expand_dims(Datasets["ice_charts_AICE_grid"]["SIC"], axis = 0))
    #
    output_netcdf.close()


# In[21]:


Datasets = {}
Datasets["Domain_data"] = load_AICE_coordinates(paths, date_static = "20220101")
Datasets["ice_charts"] = extract_ice_charts(paths, date_task).load_ice_chart_data()
#
params_2D_interp = {"input_dataset_name": "ice_charts",
                    "output_dataset_name": "AICE",
                    "input_dataset": Datasets["ice_charts"],
                    "output_dataset": Datasets["Domain_data"],
                    "proj4_str": proj4_str,
                   }
#
Datasets["ice_charts_AICE_grid"] = nearest_neighbor_interpolation_2D(**params_2D_interp).nearest_neighbor_interp()
#
write_netCDF_ice_charts_on_AICE_grid(date_task, Datasets, paths)
