import pandas as pd
import os
import numpy as np

# set working directory
os.chdir("../../")

import geopandas as gpd
import matplotlib.pyplot as plt

from cartopy import config
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

forest_to_ag        = pd.read_csv("./outputs/deathsperkm.csv")

# Put everything in millions of dollars
onemil              = 1000000.0
#Conversion:
forest_to_ag['conv'] = forest_to_ag['conv'] * -1.0 / onemil
# Ag deaths:
forest_to_ag['agdeaths'] = forest_to_ag['agdeaths'] / onemil
# Forest deaths
forest_to_ag['total'] = forest_to_ag['total'] / onemil

vmin, vmax          = -1.0,1.0

# CRS
central_lat = 23
central_lon = -96
crs         = ccrs.AlbersEqualArea(central_lon, central_lat)
crs_proj4   = crs.proj4_init

# Axes
ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))

ax.coastlines('10m')
ax.add_feature(cartopy.feature.BORDERS, edgecolor='black', linestyle='-')

import cartopy.feature as cfeature
ax.add_feature(cfeature.NaturalEarthFeature(
    'cultural', 'admin_1_states_provinces_lines', '10m',
    edgecolor='grey', linestyle=':', facecolor='none'))

#ax.set_title('Deaths from changing an acre of forest to cropland', pad=35, fontdict={'fontsize': '10', 'fontweight' : '3'})

# Colour map
from colormap import Colormap
c                   = Colormap()
mycmap              = c.cmap_linear('blue', 'white', 'green')
sm                  = plt.cm.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
cbar                = plt.colorbar(sm, fraction=0.03, pad=0.04)
cbar.set_label('Air quality benefit (millions $/km²)', rotation=270, labelpad=10)

geo_df              = gpd.read_file("./plots/tl_2014_us_county/tl_2014_us_county.shp")
geo_df              = geo_df.rename(columns={'GEOID': 'FIPS'})

otherstates         = ["02","15","60","66","69","72","78"]
geo_df              = geo_df[~geo_df.STATEFP.isin(otherstates)]

geo_df.FIPS = geo_df.FIPS.astype(int)
geo_df              = pd.merge(geo_df, forest_to_ag, on='FIPS')

# GDF to correct CRS
df_ae               = geo_df.to_crs(crs_proj4)

#fig                 = geo_df.plot(column='total', vmin=vmin, vmax=vmax, legend=True)
df_ae.plot(column='conv', cmap=mycmap, linewidth=0.8, ax=ax, edgecolor='None', norm=plt.Normalize(vmin=vmin, vmax=vmax))

plt.savefig('./plots/convdeaths.png', bbox_inches='tight',dpi=300)
