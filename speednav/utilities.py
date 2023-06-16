import os
import numpy as np
import pandas as pd
import geoplot as gplt
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import math

'''
Define global variables to be used throughout package.
 '''

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(FILE_DIR, '..', 'data',
                         "ccasr_speed_three_country_sample_v1.0.0.csv")
SHAPE_PATH_PHL = os.path.join(FILE_DIR, '..', 'data', "gadm36_PHL_shp",
                              'gadm36_PHL_0.shp')
SHAPE_PATH_SLE = os.path.join(FILE_DIR, '..', 'data', "gadm36_SLE_shp",
                              'gadm36_SLE_0.shp')
SHAPE_PATH_LBR = os.path.join(FILE_DIR, '..', 'data', "gadm36_LBR_shp",
                              'gadm36_LBR_0.shp')

# Dictionary for the variable codes.
DATA_VARS = {
            'AEI0001': 'Article identification number',
            'AEI0002': 'Publication date',
            'AEI0003': 'Event identification number',
            'AEI0004': 'Source of the article',
            'DL0001': 'Type of date information available',
            'DL0002': 'Earliest possible date for the event',
            'DL0003': 'Latest possible date for the event',
            'DL0004': 'Average date for the event',
            'DL0005': 'Precision of geospatial unit',
            'DL0006': 'Country name',
            'DL0007': 'Lowest-level spatial unit',
            'DL0008': 'Correlates of war country code',
            'DL0009': 'World region of the event',
            'DL0010': 'Latitude',
            'DL0011': 'Longitude',
            'DL0012': 'Physical location of the event',
            'ER0001': 'Recapitulation event',
            'ER0002': 'Event category',
            'ER0003': 'Type of political expression event',
            'ER0004': 'Reduced type of political expression event',
            'ER0005': 'Type of symbolic act event',
            'ER0006': 'Advocating/threatening of violent acts during political expression',
            'ER0007': 'Type of politically motived attack',
            'ER0008': 'Realized versus unrealized attacks',
            'ER0009': 'Type of disruptive state act category',
            'ER0010': 'Type of minimal disruptive state act',
            'ER0011': 'Type of formal disruptive state act',
            'ER0012': 'Type of coercive disruptive state act',
            'ER0013': 'Type of extraordinary state act',
            'INI0001': 'Are the initiators unknown, known, suspected or ambiguous',
            'INI0002': 'Type of initiators',
            'INI0003': 'Non-domestic status of initiators',
            'INI0004': 'Country of non-domestic initiators',
            'INI0005_1': 'First formal name of known or suspected initiators',
            'INI0005_2': 'Second formal name of known or suspected initiators',
            'INI0005_3': 'Third formal name of known or suspected initiators',
            'INI0005_4': 'Fourth formal name of known or suspected initiators',
            'INI0005_5': 'Fifth formal name of known or suspected initiators',
            'INI0005_6': 'Sixth formal name of known or suspected initiators',
            'INI0005_7': 'Seventh formal name of known or suspected initiators',
            'INI0005_8': 'Eighth formal name of known or suspected initiators',
            'INI0005_9': 'Ninth formal name of known or suspected initiators',
            'INI0005_10': 'Tenth formal name of known or suspected initiators',
            'INI0005_11': 'Eleventh formal name of known or suspected initiators',
            'INI0005_12': 'Twelfth formal name of known or suspected initiators',
            'INI0005_13': 'Thirteenth formal name of known or suspected initiators',
            'INI0005_14': 'Fourteenth formal name of known or suspected initiators',
            'INI0005_15': 'Fifteenth formal name of known or suspected initiators',
            'INI0005_16': 'Sixteenth formal name of known or suspected initiators',
            'INI0005_17': 'Seventeenth formal name of known or suspected initiators',
            'INI0005_18': 'Eighteenth formal name of known or suspected initiators',
            'INI0006': 'Type of non-governmental initiators',
            'INI0007': 'Level of government',
            'INI0008': 'Type of governmental initiators',
            'INI0009': 'Type of weapon used by known or suspected initiators',
            'INI0010': 'Reduced weapon type for known or suspected initiators',
            'INI0011_1': 'First named group for known or suspected initiators',
            'INI0011_2': 'Second named group for known or suspected initiators',
            'INI0011_3': 'Third named group for known or suspected initiators',
            'INI0011_4': 'Fourth named group for known or suspected initiators',
            'INI0011_5': 'Fifth named group for known or suspected initiators',
            'INI0012': 'Highest number of known or suspected initiators',
            'INI0013': 'Lowest number of known or suspected initiators',
            'INI0014': 'Type of ambiguous initiators (1)',
            'INI0015': 'Non-domestic status of ambiguous initiators (1)',
            'INI0016': 'Country of non-domestic ambiguous initiators (1)',
            'INI0017_1': 'First formal name of ambiguous initiators (1)',
            'INI0017_2': 'Second formal name of ambiguous initiators (1)',
            'INI0017_3': 'Third formal name of ambiguous initiators (1)',
            'INI0017_4': 'Fourth formal name of ambiguous initiators (1)',
            'INI0017_5': 'Fifth formal name of ambiguous initiators (1)',
            'INI0018': 'Type of non-governmental ambiguous initiators (1)',
            'INI0019': 'Level of government for ambiguous initiators (1)',
            'INI0020': 'Type of governmental ambiguous initiators (1)',
            'INI0021': 'Type of weapon used by ambiguous initiators (1)',
            'INI0022': 'Reduced weapon type for ambiguous initiators (1)',
            'INI0023_1': 'First named group for ambiguous initiators (1)',
            'INI0023_2': 'Second named group for ambiguous initiators (1)',
            'INI0023_3': 'Third named group for ambiguous initiators (1)',
            'INI0023_4': 'Fourth named group for ambiguous initiators (1)',
            'INI0023_5': 'Fifth named group for ambiguous initiators (1)',
            'INI0024': 'Highest number of ambiguous initiators (1)',
            'INI0025': 'Lowest number of ambiguous initiators (1)',
            'INI0026': 'Type of ambiguous initiators (2)',
            'INI0027': 'Non-domestic status of ambiguous initiators (2)',
            'INI0028': 'Country of non-domestic ambiguous initiators (2)',
            'INI0029_1': 'First formal name of ambiguous initiators (2)',
            'INI0029_2': 'Second formal name of ambiguous initiators (2)',
            'INI0029_3': 'Third formal name of ambiguous initiators (2)',
            'INI0029_4': 'Fourth formal name of ambiguous initiators (2)',
            'INI0029_5': 'Fifth formal name of ambiguous initiators (2)',
            'INI0030': 'Type of non-governmental ambiguous initiators (2)',
            'INI0031': 'Level of government for ambiguous initiators (2)',
            'INI0032': 'Type of governmental ambiguous initiators (2)',
            'INI0033': 'Type of weapon used by ambiguous initiators (2)',
            'INI0034': 'Reduced weapon type for ambiguous initiators (2)',
            'INI0035_1': 'First named group for ambiguous initiators (2)',
            'INI0035_2': 'Second named group for ambiguous initiators (2)',
            'INI0035_3': 'Third named groups for ambiguous initiators (2)',
            'INI0035_4': 'Fourth named group for ambiguous initiators (2)',
            'INI0035_5': 'Fifth named group for ambiguous initiators (2)',
            'INI0036': 'Highest number of ambiguous initiators (2)',
            'INI0037': 'Lowest number of ambiguous initiators (2)',
            'TE0001': 'Type of non-governmental targets',
            'TE0002': 'Level of government',
            'TE0003': 'Type of governmental targets',
            'TE0004': 'Non-domestic status of targets',
            'TE0005': 'Country of non-domestic targets',
            'TE0006_1': 'First formal name of targets',
            'TE0006_2': 'Second formal name of targets',
            'TE0006_3': 'Third formal name of targets',
            'TE0006_4': 'Fourth formal name of targets',
            'TE0006_5': 'Fifth formal name of targets',
            'TE0006_6': 'Sixth formal name of targets',
            'TE0006_7': 'Seventh formal name of targets',
            'TE0006_8': 'Eighth formal name of targets',
            'TE0006_9': 'Ninth formal name of targets',
            'TE0006_10': 'Tenth formal name of targets',
            'TE0006_11': 'Eleventh formal name of targets',
            'TE0006_12': 'Twelfth formal name of targets',
            'TE0006_13': 'Thirteenth formal name of targets',
            'TE0006_14': 'Fourteenth formal name of targets',
            'TE0007_1': 'First named group for targets',
            'TE0007_2': 'Second named group for targets',
            'TE0007_3': 'Third named group for targets',
            'TE0007_4': 'Fourth named group for targets',
            'TE0007_5': 'Fifth named group for targets',
            'TE0008': 'Type of impact on human victims/targets',
            'TE0009': 'Type of egregious violence',
            'TE0010': 'High number of persons killed',
            'TE0011': 'Low number of persons killed',
            'TE0012': 'High number of persons injured',
            'TE0013': 'Low number of persons injured',
            'TE0014': 'High number of persons kidnapped/held hostage',
            'TE0015': 'Low number of persons kidnapped/held hostage',
            'TE0016': 'High number of persons arrested',
            'TE0017': 'Low number of persons arrested',
            'TE0018': 'Geopolitical entity targeted',
            'TE0019': 'Property targeted or damaged',
            'O0001': 'The attributed origins of the event',
            'DUPE0001': 'Is this event part of a duplicate event cluster?',
            'DUPE0002': 'Duplicate event cluster number'
}

# Intensity-related variables
INTENSITY_VARS = {'SYMBOLIC',
                  '#_OF_INITIATORS_DISC',
                  'LENGTH_DISC',
                  'VIOLENCE_ADV',
                  '#_PARTICIPATING',
                  'WEAP/INJURY',
                  'ATTACK_PER',
                  'WEAPON_GRD',
                  'ATTACK_EXT',
                  '#_INJURED',
                  '#_KILLED',
                  'COERCIVE_ACT',
                  'ARREST',
                  'SOLDIER',
                  'EXTRAORDINARY'}


def load_data():
    '''
    load the full speed dataset
    '''
    data = pd.read_csv(DATA_PATH, low_memory=False)

    # turn date variables into datetime objects
    data['DL0002'] = pd.to_datetime(data['DL0002'])
    data['DL0003'] = pd.to_datetime(data['DL0003'])
    data['DL0004'] = pd.to_datetime(data['DL0004'])
    return data


def filter_data(data, country_name, location=False):
    if location:
        data_country = data.loc[(data.DL0006 == country_name)
                                & (data.DL0007 != country_name)]
    else:
        data_country = data.loc[(data.DL0006 == country_name)]

    return data_country


def get_time_bins(time_first, time_last, interval='year'):
    if interval == 'year':
        times = np.arange(time_first.year, time_last.year + 2)
    if interval == 'day':
        time_diff = time_last-time_first
        times = np.arange(time_first, time_first + time_diff, datetime.timedelta(days=1))
    return times


def filter_times(data_filter, times, date_var='DL0002', cumulative=True):
    data_times = []
    for i in range(1, len(times)):
        if cumulative:
            data_time = data_filter.loc[(data_filter[date_var].dt.year >= times[0])
                                        & (data_filter[date_var].dt.year < times[i])]
        else:
            data_time = data_filter.loc[(data_filter[date_var].dt.year >= times[i-1])
                                        & (data_filter[date_var].dt.year < times[i])]
        data_times.append(data_time)
    return data_times


def plot_event_map(data_filter,
                   kde=True,
                   clip_border=True,
                   fill=True,
                   cmap='Reds',
                   color='red',
                   levels=100,
                   thresh=0.01,
                   gridsize=200,
                   marker='.',
                   projection=gplt.crs.AlbersEqualArea(),
                   linewidth=0.5,
                   marker_linewidth=0,
                   edgecolor=[0.5, 0.5, 0.5],
                   s=7,
                   alpha=0.05,
                   ax=None,
                   weights_var=None
                   ):

    latitude = data_filter['DL0010']
    longitude = data_filter['DL0011']
    country = data_filter['DL0006'].iloc[0]
    geoseries = gpd.points_from_xy(longitude, latitude)

    if country == 'Philippines':
        shape_path = SHAPE_PATH_PHL
    if country == 'Sierra Leone':
        shape_path = SHAPE_PATH_SLE
    if country == 'Liberia':
        shape_path = SHAPE_PATH_LBR

    country_shape = gpd.read_file(shape_path)
    geoseries = gpd.points_from_xy(longitude, latitude)
    coords = gpd.GeoDataFrame(geometry=geoseries)
    if clip_border:
        clip = country_shape.geometry
    else:
        clip = None

    if ax is None:
        ax = plt.gca()
    else:
        plt.sca(ax)

    if weights_var is None:
        weights = None
    else:
        weights = np.array(data_filter[weights_var].fillna(0))

    if kde:
        gplt.kdeplot(coords,
                     clip=clip,
                     fill=fill,
                     cmap=cmap,
                     levels=levels,
                     thresh=thresh,
                     gridsize=gridsize,
                     ax=ax,
                     weights=weights)
        gplt.polyplot(country_shape,
                      ax=ax,
                      zorder=2,
                      linewidth=linewidth,
                      edgecolor=edgecolor)
    else:
        gplt.pointplot(coords,
                       ax=ax,
                       s=s,
                       color=color,
                       marker=marker,
                       linewidths=marker_linewidth,
                       alpha=alpha)
        gplt.polyplot(country_shape,
                      linewidth=linewidth,
                      edgecolor=edgecolor,
                      ax=ax)


def plot_map_series(data_times, titles=None, kde=False, s=3, alpha=0.02, 
                    save=False):
    if len(data_times) >= 10:
        ncols = 10
    else:
        ncols = len(data_times)
    nrows = int(np.floor(len(data_times)/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4))
    for i, ax in enumerate(axes.flatten()):
        if data_times[i].shape[0] == 0:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', which='both', length=0,
                           labelleft=False, labelbottom=False)
            continue
        if data_times[i].shape[0] < 100:
            levels = data_times[i].shape[0]
        else:
            levels = 100
        if titles is not None:
            ax.set_title(str(titles[i]), fontsize=8)
        plot_event_map(data_times[i],
                       kde=kde,
                       ax=ax,
                       s=s,
                       alpha=alpha)
        if save:
            plt.savefig('grid.jpeg', dpi=300)
    plt.show()


def calc_var_subseq(var, data_filter):
    data_filter = data_filter.sort_values(var)
    if data_filter[var].dtype == 'datetime64[ns]':
        var_diff_subseq_ev = np.array(data_filter[var].diff().dt.days)[1:]
    if var == 'DL0010' or var == 'DL0011':
        print('Calculating distance using DL0010 and DL0011')
        lat1 = np.pi/180 * np.array(data_filter['DL0010'])[0:-1]
        lat2 = np.pi/180 * np.array(data_filter['DL0010'])[1:]
        lon1 = np.pi/180 * np.array(data_filter['DL0011'])[0:-1]
        lon2 = np.pi/180 * np.array(data_filter['DL0011'])[1:]
        lat_diffs = np.pi/180*(lat2-lat1)
        lon_diffs = np.pi/180*(lon2-lon1)
        R = 6371  # Radius of the earth in kilometers
        a = np.sin(lat_diffs/2) * np.sin(lat_diffs/2) + np.cos(lat1) * np.cos(lat2) * np.sin(lon_diffs/2) * np.sin(lon_diffs/2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        var_diff_subseq_ev = R * c
    elif data_filter[var].dtype == 'float' or data_filter[var].dtype == 'int':
        var_diff_subseq_ev = np.array(data_filter[var].diff())
    #else:
    #    print('This variable type not yet implemented')
    return var_diff_subseq_ev


def calc_stat_vs_time(data_filter, var, window_size, date_var='DL0002', step=1,
                      calc_type='mean'):

    var_vals = data_filter[var]

    # make a dataframe of the variable values indexed by date
    var_date_df = pd.DataFrame(np.array(var_vals),
                               index=np.array(data_filter[date_var]),
                               columns=[var])

    # sort by ascending date
    var_date_df = var_date_df.sort_index()
    var_date_df = var_date_df.dropna()
    var_date_df = var_date_df.groupby(level=0).mean()

    # calculate the full list of consecutive days in the date range
    date_range = pd.date_range(var_date_df.index.min(),
                               var_date_df.index.max(),
                               freq='D')

    # re-index, filling dates with no value with nan
    var_date_df = var_date_df.reindex(index=date_range)
    var_df = pd.DataFrame(np.array(var_date_df[var]))

    # calculate the rolling mean with the set window and step size
    # automatically skips nans
    if calc_type == 'mean':
        stat_vs_time = var_df.rolling(window=window_size, min_periods=1,
                                      step=step).mean()
    if calc_type == 'count':
        stat_vs_time = var_df.rolling(window=window_size, min_periods=1,
                                      step=step).count()

    return stat_vs_time
