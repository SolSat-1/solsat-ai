import pandas as pd
from pathlib import Path
import ee
from tqdm import tqdm

SELECT_BANDS = ['dewpoint_temperature_2m', 'temperature_2m', 'skin_temperature',
        'soil_temperature_level_1', 'soil_temperature_level_2',
        'soil_temperature_level_3', 'soil_temperature_level_4',
        'skin_reservoir_content', 'volumetric_soil_water_layer_1',
        'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3',
        'volumetric_soil_water_layer_4', 'surface_latent_heat_flux_sum',
        'surface_net_solar_radiation_sum', 'surface_net_thermal_radiation_sum',
        'surface_sensible_heat_flux_sum',
        'surface_solar_radiation_downwards_sum',
        'surface_thermal_radiation_downwards_sum', 'u_component_of_wind_10m',
        'v_component_of_wind_10m', 'surface_pressure',
        'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation']

def get_station_info(station_data_path):
    station_df = pd.read_csv(station_data_path, sep='\t')
    
    # Get min and max of station location
    ltt_min, ltt_max = station_df['Latitude [deg]'].min(), station_df['Latitude [deg]'].max()
    lgt_min, lgt_max = station_df['Longitude [deg]'].min(), station_df['Longitude [deg]'].max()
    boundary = [ltt_min, lgt_min, ltt_max, lgt_max]
    
    # list of station latitude longitude
    station_locations = list(zip(station_df['Latitude [deg]'], station_df['Longitude [deg]'], station_df['LocationID']))
    
    return boundary, station_locations

def get_gee_info(start_date, end_date, boundary, latlong_list, select_bands=SELECT_BANDS):
    # Authentication
    ee.Authenticate()
    ee.Initialize()
    
    # Get Geometry infomation
    roi = ee.Geometry.Rectangle(boundary)
    ltt_lgt_ee_list = ee.List(latlong_list)  # Convert to an Earth Engine list

    # Define and generate list of dates from range
    start_date = ee.Date(start_date)
    end_date = ee.Date(end_date)
    date_list = ee.List.sequence(0, end_date.difference(start_date, 'day').subtract(1)).map(
        lambda n: start_date.advance(n, 'day')
    )
 
    # Function to get reduced value for each day and each lat/lgt tuple
    def get_daily_info(date):
        date = ee.Date(date)
        
        # Function to get reduced value for each lat/lgt tuple
        def get_point_info(lttlgt_list):
            ltt = ee.Number(ee.List(lttlgt_list).get(0))
            lgt = ee.Number(ee.List(lttlgt_list).get(1))
            station_id = ee.Number(ee.List(lttlgt_list).get(2))
            point = ee.Geometry.Point([lgt, ltt])

            daily_data = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR').select(
                select_bands).filterBounds(roi).filterDate(date, date.advance(1, 'day'))

            reduced_value = daily_data.mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=point,
                scale=0.5
            )

            # Add date and ltt/lgt info to the result
            return ee.Dictionary(reduced_value).set('date', date.format('YYYY-MM-dd')).set('ltt', ltt).set('lgt', lgt).set('station_id', station_id)
        
        # Map over all ltt/lgt tuples (converted to EE List)
        return ltt_lgt_ee_list.map(get_point_info)
    
    # get daily and latlong information
    daily_values = date_list.map(get_daily_info)
    flattened_values = daily_values.flatten()
    daily_values_info = flattened_values.getInfo()
    
    return daily_values_info

def next_month(date_string):
    dates = date_string.split('-')
    year = int(dates[0])
    month = int(dates[1])
    day = int(dates[2])
    
    month += 1
    if month > 12:
        month = 1
        year += 1
        
    return f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"

def pipe(data_dir_path, output_dir=None):
    data_dir = Path(data_dir_path)
    subdata_list = list(data_dir.glob('*'))
    
    for subdata_dir in tqdm(subdata_list):
        # Station
        station_data_path = subdata_dir / f"metadata_stations_{subdata_dir.name}.tsv"
        boundary, station_locations = get_station_info(station_data_path)
        start_date = f"{subdata_dir.name}-01"
        end_date = next_month(start_date)
        
        # GEE
        daily_info = get_gee_info(start_date, end_date, boundary, station_locations, select_bands=SELECT_BANDS)
        gee_df = pd.DataFrame(daily_info)
        if output_dir is not None:
            gee_df.to_csv(Path(output_dir) / f'gee_{subdata_dir.name}.csv', index=False)

if __name__ == '__main__':
    pipe('./data', './gee_data')
    
    
    