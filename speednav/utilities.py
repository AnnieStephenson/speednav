import os
import numpy as np
import pandas as pd
import geoplot as gplt
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
import math
import itertools
import networkx as nx

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

INTENSITY_VARS = {'symbolic': '0 or 1, describing whether or not the act was symbolic',
                  'num_initiators_disc': 'total number of initiators, includeing known, suspected, and ambigious',
                  'num_hostage': 'total number of hostages',
                  'length_disc': 'length of event in days',
                  'violence_adv': '0 or 1 to indicate whether violence was advocated',
                  'weap_injury': '0 or 1 to indicate whether there were weapons involved or persons injured',
                  'attack_per': '0 or 1 to indicate whether it was an attack targeting persons',
                  'weapon_grd': 'reduced weapon type of known or suspected initiators',
                  'attack_ext': '0 or 1 to indicate whether an attack was extraordinary in nature',
                  'num_injured': 'number of persons injured',
                  'num_killed': 'numer of persons killed',
                  'coercive_act': '0 or 1 to indicate whether event was a coercive act',
                  'num_arrest': 'number of persons arrested',
                  'soldier': '0 or 1 to indicate whether soldiers were involved',
                  'extraordinary': '0 or 1 to indicate whether the event was an extraordinary state act',
                  'num_participating': 'NUM_INJURED + NUM_OF_INITIATORS_DISC + NUM_KILLED + NUM_HOSTAGE'}

def calc_intensity_vars(data_filter):
    '''
    '''

    # convert nans to 0. Needed for numeric values
    data_nan0 = data_filter.fillna(0)

    # SYMBOLIC
    data_filter['symbolic'] = np.array(data_nan0.ER0004 == 2).astype('int')

    # 'NUM_OF_INITIATORS_DISC'
    data_filter['num_initiators_disc'] = (data_nan0.INI0012 + data_nan0.INI0013)/2 + (data_nan0.INI0024 + data_nan0.INI0025)/2

    # 'NUM_HOSTAGE'
    data_filter['num_hostage'] = (data_nan0.TE0014 + data_nan0.TE0015)/2

    # 'LENGTH_DISC'
    data_filter['length_disc'] = (data_nan0.DL0003 - data_nan0.DL0002).dt.days  # convert from datetime to number

    # 'VIOLENCE_ADV'
    data_filter['violence_adv'] = data_nan0.ER0006

    # 'WEAP/INJURY'
    data_filter['weap_injury'] = (~((data_nan0.INI0009 == 0) * (data_nan0.TE0012 == 0))).astype('int')

    # 'ATTACK_PER'
    data_filter['attack_per'] = (data_nan0.ER0007 == 7).astype('int')

    # 'WEAPON_GRD'
    data_filter['weapon_grd'] = data_nan0.INI0010

    # 'ATTACK_EXT'
    data_filter['attack_ext'] = (data_nan0.ER0007 <= 5)*(data_nan0.ER0007 >= 2).astype('int')

    # 'NUM_INJURED'
    data_filter['num_injured'] = (data_nan0.TE0012 + data_nan0.TE0013)/2

    # 'NUM_KILLED'
    data_filter['num_killed'] = (data_nan0.TE0010 + data_nan0.TE0011)/2

    # 'COERCIVE_ACT'
    data_filter['coercive_act'] = (data_nan0.ER0009 == 3).astype('int')

    # 'NUM_ARREST'
    data_filter['num_arrest'] = (data_nan0.TE0016 + data_nan0.TE0017)/2.

    # 'SOLDIER'
    data_filter['soldier'] = (data_nan0.INI0006 == 31).astype('int')

    # 'EXTRAORDINARY'
    data_filter['extraordinary'] = (data_nan0.ER0009 == 4).astype('int')

    # 'NUM_PARTICIPATING'
    data_filter['num_participating'] = data_filter.num_injured + data_filter.num_initiators_disc + data_filter.num_killed + data_filter.num_hostage


# composite intensity variables
INTENSITY_VARS_COMP = {'EXPRESSION_SG',
                       'EXPRESSION_MASS',
                       'VIOLENCE_NS',
                       'REPRESSION_ST',
                       'DISRUPTIVE_ST',
                       'VIOLENCE_ST'}


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
    if calc_type == 'sum':
        stat_vs_time = var_df.rolling(window=window_size, min_periods=1,
                                      step=step).sum()

    return stat_vs_time


def construct_edge_list(data_filter, node_name, edge_name, return_df=False):

    # first make an array or df of each node var and edge variables
    if node_name == 'index':
        nodes = data_filter.index
    else:
        nodes = data_filter[node_name]

    if edge_name == 'index':
        edges = data_filter.index
    else:
        edges = data_filter[edge_name]
    node_edge_data = np.vstack((nodes, edges))
    sorted_edge_data = np.sort(node_edge_data.astype(str), axis=1)

    # if two nodes have the same edge variable, then they share an edge
    # sort the dataframe based on the edge variable
    # use numpy unique with return_inverse true
    edge_types, inds = np.unique(sorted_edge_data[1, :].astype(str), return_index=True)

    # get rid of nans, which are always last (TODO: check to make sure this is always true)
    if edge_types[-1] == 'nan':
        edge_types = edge_types[:-1]
        inds = inds[:-1]

    # then will have to loop through those indices, as they'll be the first ones
    edge_list_tot = np.zeros((1, 2))
    for i in range(len(inds)):
        nodes_edge_type = sorted_edge_data[0, i-1:inds[i]].astype(int)
        edge_list = np.array(list(itertools.combinations(nodes_edge_type, 2)))
        if not edge_list.size == 0:
            edge_list_tot = np.vstack((edge_list_tot, edge_list))
    edge_list_tot = np.delete(edge_list_tot, 0, 0)

    # then make a pandas df from the edge list array where first column is source and second is target
    if return_df:
        edge_df = pd.DataFrame({'source': edge_list_tot[:, 0], 'target': edge_list_tot[:, 1]})
        result = edge_df
    else:
        result = edge_list_tot

    return result


def construct_edge_list_multivar(data_filter, node_name, edge_name_list, return_df=False):
    if return_df:
        edge_lists = []
    else:
        edge_comb = np.zeros((1, 2))
    for i in range(len(edge_name_list)):
        edge_list = construct_edge_list(data_filter, node_name, edge_name_list[i], return_df=return_df)
        if return_df:
            edge_lists.append(edge_list)
        else:
            edge_comb = np.vstack((edge_comb, edge_list))
    if return_df:
        edge_comb = pd.concat(edge_lists)
    else:
        edge_comb = np.delete(edge_comb, 0, 0)

    return edge_comb