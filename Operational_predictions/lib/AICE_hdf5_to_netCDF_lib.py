# -*- coding: utf-8 -*-
import os
import h5py
import netCDF4
import datetime
###################################################################################################
def load_forecasts(paths, today_date):
    filename = paths["forecasts_temp"] + "AICE_forecasts_" + today_date + "T000000Z.h5"
    with h5py.File(filename, "r") as file:
        Dataset = {}
        Dataset["time"] = file["time"][:]
        Dataset["x"] = file["x"][:]
        Dataset["y"] = file["y"][:]
        Dataset["lat"] = file["lat"][:,:]
        Dataset["lon"] = file["lon"][:,:]
        Dataset["SIC"] = file["SIC"][:,:,:]
    return(Dataset)
###################################################################################################
def write_netCDF_forecasts(Dataset, paths, today_date):
    lead_time_max = 10
    time_coverage_first = datetime.datetime.strptime(today_date, "%Y%m%d").strftime("%Y-%m-%dT000000Z")
    time_coverage_last = (datetime.datetime.strptime(today_date, "%Y%m%d") + datetime.timedelta(days = lead_time_max)).strftime("%Y-%m-%dT000000Z")
    #
    path_output = paths["forecasts"] 
    if os.path.exists(path_output) == False:
        os.system("mkdir -p " + path_output)    
    output_filename = path_output + "AICE_forecasts_" + today_date + "T000000Z.nc"
    if os.path.isfile(output_filename):
        os.system("rm " + output_filename)
    output_netcdf = netCDF4.Dataset(output_filename, "w", format = "NETCDF4")
    #
    time = output_netcdf.createDimension("time", lead_time_max)
    x = output_netcdf.createDimension("x", len(Dataset["x"]))
    y = output_netcdf.createDimension("y", len(Dataset["y"]))
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
    time.long_name = "time"
    time.units = "seconds since 1970-01-01 00:00:00 +0000"
    x.standard_name = "projection_x_coordinate"
    x.long_name = "projection_x_coordinate"
    x.units = "m"
    x.grid_mapping = "Lambert_Azimuthal_Grid"
    y.standard_name = "projection_y_coordinate"
    y.long_name = "projection_y_coordinate"
    y.units = "m"
    y.grid_mapping = "Lambert_Azimuthal_Grid"
    lat.standard_name = "latitude"
    lat.long_name = "latitude coordinate"
    lat.units = "degrees_north"
    lon.standard_name = "longitude"
    lon.long_name = "longitude coordinate"
    lon.units = "degrees_east"
    SIC.standard_name = "sea_ice_area_fraction"
    SIC.long_name = "Sea ice concentration"
    SIC.units = "%"
    SIC.grid_mapping = "Lambert_Azimuthal_Grid"
    SIC.coordinates = "time lat lon"
    SIC.coverage_content_type = "modelResult"    
    #
    time[:] = Dataset["time"]
    x[:] = Dataset["x"]
    y[:] = Dataset["y"]
    lat[:,:] = Dataset["lat"]
    lon[:,:] = Dataset["lon"]
    SIC[:,:,:] = Dataset["SIC"]
    #
    output_netcdf.Conventions = "CF-1.8, ACDD-1.3"
    output_netcdf.title = "AICE sea ice concentration forecasts"
    output_netcdf.summary = "Short-range sea ice concentration forecasts produced using deep learning at a spatial resolution of 5 km."
    output_netcdf.keywords = "Sea ice concentration forecasts, Deep learning, European Arctic"
    output_netcdf.area = "European Arctic"
    output_netcdf.institution = "Norwegian Meteorological Institute"
    output_netcdf.PI_name = "Cyril Palerme"
    output_netcdf.contact = "cyril.palerme@met.no"
    output_netcdf.bulletin_type = "Forecast"
    output_netcdf.forecast_range = "10 days"
    output_netcdf.time_coverage_start = time_coverage_first
    output_netcdf.time_coverage_stop = time_coverage_last
    output_netcdf.close()
###################################################################################################
