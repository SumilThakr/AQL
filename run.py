import pandas as pd
import os
import numpy as np

# set working directory
os.chdir(".")


#########################################################################
########################### LAND USE CHANGES ############################
#########################################################################

# Pixel counts are from Lawler et al. (2014) for each scenario and land
# use type. The headers for pixel counts are of the format
# scenario#LANDCOVERCODE, where:
#
# the scenarios are:
# nlcd              2001 baseline
# forest            Forest incentives (subsidies to convert to forests)
#                   policy scenario
# native            Natural habitats (taxed conversion of forests)
#                   policy scenario
# proag             High crop demand trend scenario
# ref               1990s trend scenario
# urban             urban containment policy scenario
#
# and the land cover codes are:
#
# 1                 crops
# 2                 pasture
# 3                 forest
# 4                 urban
# 5                 rangeland

# This data was extracted from raw inputs available at:
# http://silvis.forest.wisc.edu/future-us-land-cover-2050/

pixels              = pd.read_csv('./inputs/lu/pixel-count.csv')

# Each pixel is 100m x 100m. Convert to km²:
cols                = list(pixels.columns.values.tolist())
for col in cols[9:40]:
    pixels[col]     = pixels[col] / 100.0

#########################################################################
######################## AGRICULTURAL  EMISSIONS ########################
#########################################################################

# 1. Ammonia emissions from fertilizer.
# Load county-level synthetic fertilizer fractions
fertfrac            = pd.read_csv('./inputs/ag/fertfrac.csv')
fertmat             = fertfrac.iloc[:,[4,5,6,7,8,9,10]].to_numpy()
# Load emission factors for each type of fertilizer.
# N.B. These are from Diaz Goebes et al. (2003).
fertef              = pd.read_csv('./inputs/efs/fertef.csv')
# Drop manure:
manureef            = fertef.tail(1)['ef'][:]
fertef.drop(fertef.tail(1).index,inplace=True)
# Multiply the emission factors by the fertilizer fraction, and then by
# 17/14 (to convert from kg-N to kg-NH3). The result is the county-level
# emission factor for applied synthetic N in kg-NH3/kg-N-applied:
efs                 = np.dot(fertmat,fertef['ef'].to_numpy().T) * 17.0/14.0
fdf                 = pd.DataFrame(efs, columns = ['fert-ef'])
fdf['FIPS']         = fertfrac['FIPS']
# Load fertilizer data from NUGIS
nugis               = pd.read_csv('./inputs/ag/nugis.csv')
# Recovered N takes the manure emission factor manureef = 0.305.
# Farm N takes the county-level fertilizer emission factors from fdf (above).
# These need to be joined on FIPS:
m                   = pd.merge(nugis, fdf, on='FIPS',how='left')
# Some states (e.g., Connecticut) don't have fertilizer fraction data, so
# use a national, production-weighted average EF of 0.078
m['fert-ef']        = m['fert-ef'].replace(0, 0.078)
m['fert-ef']        = m['fert-ef'].replace(np.nan, 0.078)
# Convert lbs/acre to kg/km² when you multiply.
# 1 lb              = 0.453592 kg
# 1 acre            = 0.00404686 km²
# 1 lb/acre         = 0.453592 / 0.00404686
c                   = 0.453592/0.00404686
# Get NH3 emissions in kg-NH3/km² from fertilizer and manure by county:
m['emis']           = (m['FarmLbsN_perAc'] * m['fert-ef'] * c) + ((m['RecLbsN_perAc']) * 0.305 * c) 
# m['emis'] is ready to be multiplied by crop areas in km² for each
# scenario to get emissions.

# Pasture also takes manure emissions.
# 0.59 is from FAO: manure left on pasture as a fraction of total manure
# (left on pasture + applied to soils)
m['pastureemis']           = ((m['RecLbsN_perAc']) * 0.305 * c * 0.59)

# 2. Dust PM2.5 emissions
# Agricultural dust emissions are largely from tilling and harvest
# activities, which vary spatially based on ERS resource regions.
# Load ERS resource regions
ers                 = pd.read_csv('./inputs/ag/ers.csv')
#ERS Resource Region key: 
#    1:             Heartland
#    2:             Northern Crescent
#    3:             Northern Great Plains
#    4:             Prairie Gateway
#    5:             Eastern Uplands
#    6:             Southern Seaboard
#    7:             Fruitful Rim
#    8:             Basin and Range
#    9:             Mississippi Portal
# Wade, Claassen, and Wallander (2015) give survey information on the
# type of tilling practices by region:
till                = pd.read_csv('./inputs/ag/ersTill.csv')
ers                 = pd.merge(ers, till, on='ERS',how='left')
# Zhang, Heath, Carpenter,and Fisher (2015) give PM10 emissions by
# tilling and harvest behavior:
tillemis            = pd.read_csv('./inputs/ag/zhang.csv')
tillpm10            = tillemis['PM10_lb_acre']
# Following Hill et al. (2019), we assume a crosswalk as follows:
# Full adoption:                    No till emissions
# Partial adoption, No till/strip:  Reduced emissions
# Partial, other and Non-adoption:  Conventional emissions
# We also convert from lb/acre and assume that mass of PM2.5/PM10 = 0.2
fineratio           = 0.2
data = {'FullAdopt':        [(tillpm10[4] + tillpm10[5]) * c * fineratio],
        'PartialNoStrip':   [(tillpm10[2] + tillpm10[3]) * c * fineratio],
        'PartialOther':     [(tillpm10[0] + tillpm10[1]) * c * fineratio],
        'NonAdopt':         [(tillpm10[0] + tillpm10[1]) * c * fineratio]}
tems                = pd.DataFrame(data)
# fertfrac.csv also has dust transport fractions from Pace et al. (2005).
# These are county averages of local, within-grid-cell deposition of
# dust emissions from agricultural sources.
tfs                 = fertfrac[['FIPS', 'Dust Transport Fraction']].copy()
ers                 = pd.merge(ers, tfs, on='FIPS',how='left')
# Handle no data
ers['Dust Transport Fraction']     = ers['Dust Transport Fraction'].replace(np.nan, ers['Dust Transport Fraction'].mean())

# Get the tilling PM2.5 emissions per km² of cropland by county.
ers['emis']         = ers['Dust Transport Fraction'] * (ers['FullAdopt'] * tems['FullAdopt'][0] + ers['PartialNoStrip'] * tems['PartialNoStrip'][0] + ers['PartialOther'] * tems['PartialOther'][0] + ers['NonAdopt'] * tems['NonAdopt'][0])
# ers['emis'] is ready to be multiplied by crop areas in km² for each
# scenario to get emissions.

#########################################################################
########################## SOIL  NO  EMISSIONS ##########################
#########################################################################

# Williams, Guenther, and Fehsenfield (1992) give total United States
# soil NOx emissions by land use type (from Table 4):
# Source        NO (MMton-N/yr)
# -----------------------------
# Grassland      89.3
# Forests        16.3
# Wetlands        0.053
#
# Corn          136
# Wheat          41.5
# Soybeans       13.0
# Cotton         18.0
# Also, Table 3 gives areas by land use type:
# Category      Area (10^6 km²)
# -----------------------------
# Grassland       0.937
# Forests         2.344
# Wetlands        0.352
#
# Corn            0.300
# Wheat           0.214
# Soybeans        0.554
# Cotton          0.047
# From these, we estimate average NO emissions (kg-NO/km²) for each land
# use type.
# Convert MMton-N to kg-NO:
# 1 MMton           = 1000000 kg
# 1 kg-N            = (14.0067 + 15.999) / 14.0067 kg-NO
cmass               = 1000000 * (14.0067 + 15.999) / 14.0067
# The average NO emissions per agricultural land for all crops listed (kg-NO/km²):
cropnox             = cmass * (136 + 41.5 + 13.0 + 18.0) / (1000000 * (0.3 + 0.214 + 0.554 + 0.047))
# The average NO emissions per land area for grasslands (kg-NO/km²):
grassnox            = cmass * 89.3 / (1000000 * 0.937)
# The average NO emissions per land area for forests (kg-NO/km²):
forestnox           = cmass * 16.3 / (1000000 * 2.344)

#########################################################################
####################### WINDBLOWN DUST  EMISSIONS #######################
#########################################################################

# Windblown dust emissions are derived from the following study:
# Mansell, G., et al. "Fugitive Wind Blown Dust Emissions and Model
# Performance Evaluation–Phase II." ENVIRON International Corporation,
# Novato California and University of California at Riverside. May.
# http://www.wrapair.org/forums/dejf/documents/WRAP_WBD_PhaseII_Final_Report_05
# 506 (2006).

# We use a PM2.5/PM10 ratio of 0.15, as recommended by EPA in the following
# source:
# https://www.epa.gov/sites/default/files/2020-10/documents/
#background_document_for_revisions_to_fine_fraction_ratios_used_for_ap-42_fugitive_dust_emission.pdf

# We convert to kg-PM2.5/ha for each state for each land-use class. We assume
# that "Agricultural" land corresponds to cropland, and "Grasslands"
# corresponds to pasture and rangeland.

wbddf               = pixels[['GEOID']]
wbddf               = wbddf.rename(columns={'GEOID': 'FIPS'})

wbdust              = pd.read_csv("inputs/efs/wbdustef.csv")
# Get averages of emission factors:
wb_ag_avg           = wbdust['Agricultural'].mean()
wb_grass_avg        = wbdust['Grasslands'].mean()
wbdust['SFIPS']     = wbdust['SFIPS'].astype(int) * 1000
wbddf['SFIPS']     = (np.floor(wbddf['FIPS']/1000)*1000).astype(int)
wbdust              = pd.merge(wbdust,wbddf,on='SFIPS', how='right')
wbdust['Agricultural'].replace(to_replace = np.nan, value = wb_ag_avg, inplace=True)
wbdust['Grasslands'].replace(to_replace = np.nan, value = wb_grass_avg, inplace=True)

#########################################################################
######################## NEW BIOGENIC  EMISSIONS ########################
#########################################################################

# We use global offline MEGAN 2.1 emissions by land use type (kg km-2 yr-1)
# MEGAN offline biogenic VOC emissions are available from:
# http://geoschemdata.wustl.edu/ExtData/HEMCO/OFFLINE_BIOVOC/v2021-12/0.25x0.3125/2020/
# The underlying Plant Functional Type (PFT) distributions are available at:
# https://bai.ess.uci.edu/megan/data-and-code/megan21
# The PFT map was regridded to the 0.25x0.3125 annual emissions. The
# emissions in each grid cell were then divided by a weighted sum of the
# PFTs within that grid cell, where the weights were given by the emission
# factors from Table 2 of Guenther et al. (2012) by PFT for isoprene, MTPO
# and MTPA (defined here:
# http://wiki.seas.harvard.edu/geos-chem/index.php/Species_in_GEOS-Chem
# This gave a spatial weighting factor that accounts for LAI, temperature,
# etc., that was multiplied to the emission factor and PFT distribution
# to get emissions by PFT. The PFTs were assigned land use types, where
# PFTs 1-8 were forest, PFTs 9-14 were assigned rangeland or pasture, and
# PFT 15 was assigned to agriculture. The results were then regridded to US
# counties.
# Guenther, A. B., et al. "The Model of Emissions of Gases and Aerosols from
# Nature version 2.1 (MEGAN2. 1): an extended and updated framework for
# modeling biogenic emissions." Geoscientific Model Development 5.6 (2012):
# 1471-1492.
bvoc                = pd.read_csv("./inputs/bvoc/bvoc-county.csv")

#########################################################################
############# COMPARISON WITH EXISTING EMISSION INVENTORIES #############
#########################################################################

# Below, we find the total land use related emissions burdens for the
# default land use in the National Land Cover Dataset (NLCD 2001), and
# compare the results to existing literature.

# County-specific emission factors are merged by FIPS with pixels['GEOID'].
# Rename columns
pixels              = pixels.rename(columns={'GEOID': 'FIPS'})
m                   = m.rename(columns={'emis': 'nh3-emis'})
ers                 = ers.rename(columns={'emis': 'dust-emis'})

# Drop everything but FIPS and the areas for each scenario
pixels              = pixels.iloc[:,[4,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38]]
pixels              = pd.merge(pixels, m[['FIPS','nh3-emis','pastureemis']], on='FIPS', how='left')
pixels              = pd.merge(pixels, ers[['FIPS','dust-emis']], on='FIPS', how='left')

# Handle counties with no data
pixels['nh3-emis']  = pixels['nh3-emis'].replace(np.nan, pixels['nh3-emis'].mean())
pixels['dust-emis'] = pixels['dust-emis'].replace(np.nan, pixels['dust-emis'].mean())

# Getting total emissions for NLCD
# 1. NH3
nh3                 = (pixels['nlcd1'] * pixels['nh3-emis'])
#print("Total NH3 emissions:", nh3.sum() /1000000000, "Tg/yr")
totlanduse          = pixels['nlcd1'] + pixels['nlcd2'] + pixels['nlcd3'] + pixels['nlcd4'] + pixels['nlcd5']
nh3perkm2           = nh3 / totlanduse 
nh3perkm2.to_csv("./outputs/NH3_emis.csv")
# 1.075 Tg/year
# For comparison: 0.93 Tg-NH3/yr from NEI 2017.

# 1b. NH3 from pasture
pasturen            = (pixels['nlcd2'] * pixels['pastureemis'])
#print("Total NH3 emissions from pasture:", pasturen.sum() /1000000000, "Tg/yr")
#nh3perkm2           = nh3 / totlanduse 

# 2. PM2.5
pm25                = (pixels['nlcd1'] * pixels['dust-emis'])
#print("Total PM2.5 emissions:", pm25.sum() /1000000000, "Tg/yr")
pm25perkm2          = pm25 / totlanduse
pm25perkm2.to_csv("./outputs/PM25_emis.csv")
# 0.17 Tg/yr
# For comparison: 0.79363790758 Tg/yr from NEI 2017
# Including 0.45x transport fraction, this is 0.36 Tg/yr.

# 3. NOx
nox                 = (pixels['nlcd1'] * cropnox) + ((pixels['nlcd2'] + pixels['nlcd5']) * grassnox) + (pixels['nlcd3'] * forestnox)
#print("Total NOx emissions", nox.sum() /1000000000, "Tg/yr")
noxperkm2           = nox / totlanduse
noxperkm2.to_csv("./outputs/NOx_emis.csv")
# 1.23 Tg/yr
# For comparison, 1.11 Tg/yr across the US from Hudman et al. (2012):
# https://acp.copernicus.org/articles/12/7779/2012/acp-12-7779-2012.html
# as processed in Thakrar et al. (2022).
# 1113620513.0860746 kg/yr = ~1.11 Tg/yr.

# 4. VOC
voct                = pd.merge(pixels, bvoc, on='FIPS', how='left')
vocemis             = (voct['nlcd1'] * voct['Ag']) + ((voct['nlcd2'] + voct['nlcd5']) * voct['Grass']) + (voct['nlcd3'] * voct['Forest'])
vocemis             = (voct['nlcd3'] * voct['Forest'])
#print("VOC from forest:", vocemis.sum() /1000000000, "Tg/yr")
vocemis             = (voct['nlcd1'] * voct['Ag'])
#print("VOC from agriculture:", vocemis.sum() /1000000000, "Tg/yr")
vocemis             = (voct['nlcd2'] + voct['nlcd5']) * voct['Grass']
#print("VOC from grasslands:", vocemis.sum() /1000000000, "Tg/yr")
vocemis             = (voct['nlcd1'] * voct['Ag']) + ((voct['nlcd2'] + voct['nlcd5']) * voct['Grass']) + (voct['nlcd3'] * voct['Forest'])
#print("Total VOC emissions:", vocemis.sum() /1000000000, "Tg/yr")
vocemisperkm2       = vocemis / totlanduse
vocemisperkm2.to_csv("./outputs/VOC_emis.csv")
pixels['FIPS'].to_csv("./outputs/FIPS_emis.csv")
#voct['nlcd3'].to_csv("./outputs/nlcd3.csv")
# 22.8 Tg/yr
# For comparison: 24.8 Tg/yr from MEGAN2.1 as processed in
# Thakrar et al. (2022).

#########################################################################
######################### UNCERTAINTY PARAMETERS ########################
#########################################################################

# We can consider uncertainties from many sources:
# 1. Model:             here, we use results for 2 RCMs (InMAP or AP2)
# 2. C-R function:      here, we use GEMM, ACS and H6S C-R functions
# 3. Deposition:        here, we use avg, max and min values for each county
# 5. VSL:               for mortality risk, we use a range of VSL estimates
# 6. SCC:               for carbon sequestration, we use a range of estimates

# 1. Model (choose between 'InMAP' or 'AP2')
aqm                 = 'InMAP'

# 2. C-R function (choose between 'gemm', 'acs', or 'h6s')
cr                  = 'gemm'

# 3. Deposition (choose between 'mean', 'min', or 'max')
dep_param           = 'mean'

# 4. VSL (choose between vsl_central, vsl_central - vsl_stdev, and vsl_central + vsl_stdev)
# https://www.epa.gov/sites/default/files/2017-09/documents/ee-0568-22.pdf
vsl_central         = 7.4
vsl_stdev           = 4.7
epa_vsl             = vsl_central

# 5. SCC (choose between '5avg', '3avg', '2.5avg', '3high)
scc_param = '3avg'

#########################################################################
######################### MORTALITY  ESTIMATION #########################
#########################################################################

# To estimate changes in mortality from changes in emissions, we use
# marginal damage estimates from Choma et al. (2021):
# Choma, Ernani F., et al. "Health benefits of decreases in on-road
# transportation emissions in the United States from 2008 to 2017."
# Proceedings of the National Academy of Sciences 118.51 (2021): e2107402118.
# Choma et al. uses a source receptor (SR) matrix from the InMAP Reduced
# Complexity Model (RCM). The SR matrix is described in:
#Goodkind, Andrew L., et al. "Fine-scale damage estimates of particulate
# matter air pollution reveal opportunities for location-specific
# mitigation of emissions." Proceedings of the National Academy of
# Sciences 116.18 (2019): 8775-8780.
# The InMAP model is described in:
# Tessum, Christopher W., Jason D. Hill, and Julian D. Marshall.
# "InMAP: A model for air pollution interventions." PloS one 12.4
# (2017): e0176131.
# The SR matrix data is available in:
# https://www.caces.us/data
# For different RCMs and concentration-response relationships, for both
# elevated and ground-level emissions.
# Choma et al. (2021) use more recent, non-linear concentration-response
# relationships from Burnett et al. (2018) (i.e., "GEMM").
# Here, we only consider ground-level emissions. We only use the
# InMAP source-receptor matrix using the GEMM concentration-response
# relationship.

isrm                = pd.read_csv("./inputs/rcm/rcm-gemm.csv")
# The mortality column is mortality per short ton of pollutant, so it
# needs to be converted to kg:
isrm['mortality']   = isrm['mortality'] /  907.185
isrm                = isrm.rename(columns={'fips': 'FIPS'})
# Note that the Choma et al. GEMM results have been pre-converted
# from 'per tonne' to 'per short ton'.

# Getting deaths by land use type:
def agdeaths(scenario, model, crf):
    # From PM2.5
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    pmd                 = pd.merge(pixels, pmd, on='FIPS',how='left')
    resultpm            = pmd[scenario+'1'] * pmd['dust-emis'] * pmd['mortality']
    # From Windblown dust
    wbd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    wbd                 = pd.merge(pixels, wbd, on='FIPS',how='left')
    resultwb            = wbd[scenario+'1'] * wbdust['Agricultural'] * wbd['mortality']
    # From NOx
    noxd                = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    noxd                = pd.merge(pixels, noxd, on='FIPS',how='left')
    resultnox           = (noxd[scenario+'1'] * cropnox) * noxd['mortality']
    # From NH3
    nh3d                = isrm[(isrm.pollutant == 'nh3') & (isrm.model == model) & (isrm.crf == crf)]
    nh3d                = pd.merge(pixels, nh3d, on='FIPS',how='left')
    resultnh3           = (nh3d[scenario+'1'] * nh3d['nh3-emis']) * nh3d['mortality']
    # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS', how='left')
    bvoc['Ag']          = bvoc['Ag'].replace(np.nan, bvoc['Ag'].mean())
    vocd                = pd.merge(vocd, bvoc, on='FIPS', how='left')
    resultvoc           =  vocd['Ag'] * vocd[scenario+'1'] * vocd['mortality']
    return resultpm.sum() + resultnox.sum() + resultnh3.sum() + resultvoc.sum() + resultwb.sum()

def pasturedeaths(scenario, model, crf):
    # From NOx
    noxd                = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    noxd                = pd.merge(pixels, noxd, on='FIPS',how='left')
    resultnox          = ((noxd[scenario+'2'] + noxd[scenario+'5']) * grassnox) * noxd['mortality']
    # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS')
    bvoc['Grass']       = bvoc['Grass'].replace(np.nan, bvoc['Grass'].mean())
    vocd                = pd.merge(vocd, bvoc, on='FIPS', how='left')
    resultvoc           = vocd['Grass'] * (vocd[scenario+'5'] + vocd[scenario+'2']) * vocd['mortality']
    # From NH3
    nh3d                = isrm[(isrm.pollutant == 'nh3') & (isrm.model == model) & (isrm.crf == crf)]
    nh3d                = pd.merge(pixels, nh3d, on='FIPS',how='left')
    resultnh3           = (nh3d[scenario+'2'] * nh3d['pastureemis']) * nh3d['mortality']
    # From Windblown dust
    wbd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    wbd                 = pd.merge(pixels, wbd, on='FIPS',how='left')
    resultwb            = (wbd[scenario+'5'] + wbd[scenario+'2']) * wbdust['Grasslands'] * wbd['mortality']
    return resultnox.sum() + resultvoc.sum() + resultnh3.sum() + resultwb.sum()

def forestdeaths(scenario, model, crf):
   # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS', how='left')
    bvoc['Forest']      = bvoc['Forest'].replace(np.nan, bvoc['Forest'].mean())
    vocd                 = pd.merge(vocd, bvoc, on='FIPS',how='left')
    resultvoc           =  vocd['Forest'] * vocd[scenario+'3'] * vocd['mortality']
    return resultvoc.sum()

def deaths(lu, scenario):
    switch = {
            'pasture':  pasturedeaths,
            'forest':   forestdeaths,
            'ag':       agdeaths
    }
    func = switch.get(lu, lambda: "")
    return func(scenario, aqm, cr)

ag_base                 = deaths('ag','nlcd')
forest_base             = deaths('forest','nlcd')
pasture_base            = deaths('pasture','nlcd')
tot_base                = ag_base + forest_base + pasture_base

# Change in deaths relative to 2001
scenarios               = ["forest", "native", "proag", "ref", "urban"]
'''
for x in scenarios:
    change              = deaths('ag', x) + deaths('forest', x) + deaths('pasture', x) - tot_base
    print(x, change)
'''
# change in deaths from each land use type:
'''
for x in scenarios:
    change              = deaths('ag', x) - ag_base
    print("deaths from ag land for scenario:", x, change)
for x in scenarios:
    change              = deaths('forest', x) - forest_base
    print("deaths from forest land for scenario:", x, change)
for x in scenarios:
    change              = deaths('pasture', x) - pasture_base
    print("deaths from pasture land for scenario:", x, change)
'''

#########################################################################
############################## DEPOSITION ###############################
#########################################################################

# We consider land-use related pollutant removal from forests, grasslands,
# and agricultural land.
# We consider removal of ambient PM2.5, NO2, and SO2.
# The pollutant removal per tree area for each county in g/m2 is given by
# the i-Tree Landscape Pollutant Ranges, available from:
# https://www.itreetools.org/support/resources-overview/i-tree-methods-and-files
dep                     = pd.read_csv("./inputs/dep/Landscape_air_pollutant_removal_ranges.csv")
gdep                    = pd.read_csv("./inputs/dep/Shrubland_Landscape_air_pollutant_removal_ranges.csv")
pdep                    = pd.read_csv("./inputs/dep/Grassland_Landscape_air_pollutant_removal_ranges.csv")
# For each scenario, we derive the change in pollutant removal (kg) and
# the associated health impacts if that pollution was emitted at ground level
# in the county where it was deposited.
def getpmdep(scenario, model, crf):
    depp                = dep[['FIPS','PM2.5'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    depp                = pd.merge(depp, pmd, on='FIPS',how='left')
    depp['PM2.5mean'].fillna(value=depp['PM2.5'+dep_param].mean(), inplace=True)
    df                  = pixels[['FIPS',scenario+'3','nlcd3']]
    df                  = pd.merge(df, depp, on='FIPS',how='left')
    df['PM2.5mean'].fillna(value=df['PM2.5'+dep_param].mean(), inplace=True)
    result1             =  (df[scenario+'3'] - df['nlcd3']) * df['PM2.5'+dep_param] * 1000 * df['mortality']
    result1.fillna(value=0.0, inplace=True)
    gdepp               = gdep[['FIPS','PM2.5'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    gdepp               = pd.merge(gdepp, pmd, on='FIPS',how='left')
    gdepp['PM2.5'+dep_param].fillna(value=gdepp['PM2.5'+dep_param].mean(), inplace=True)
    gdf                 = pixels[['FIPS',scenario+'1','nlcd1',scenario+'2','nlcd2',scenario+'5','nlcd5']]
    gdf                  = pd.merge(gdf, gdepp, on='FIPS',how='left')
    gdf['PM2.5'+dep_param].fillna(value=gdf['PM2.5'+dep_param].mean(), inplace=True)
    result2             =  (gdf[scenario+'5'] - gdf['nlcd5']) * gdf['PM2.5'+dep_param] * 1000 * gdf['mortality']
    result2.fillna(value=0.0, inplace=True)
    pdepp               = pdep[['FIPS','PM2.5'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    pdepp               = pd.merge(pdepp, pmd, on='FIPS',how='left')
    pdepp['PM2.5'+dep_param].fillna(value=pdepp['PM2.5'+dep_param].mean(), inplace=True)
    pdf                 = pixels[['FIPS',scenario+'1','nlcd1',scenario+'2','nlcd2',scenario+'5','nlcd5']]
    pdf                 = pd.merge(pdf, pdepp, on='FIPS',how='left')
    pdf['PM2.5'+dep_param].fillna(value=pdf['PM2.5'+dep_param].mean(), inplace=True)
    result3             =  (pdf[scenario+'2'] - pdf['nlcd2']) * pdf['PM2.5'+dep_param] * 1000 * pdf['mortality']
    result3.fillna(value=0.0, inplace=True)
    return (result1 + result2 + result3)

def getnoxdep(scenario, model, crf):
    depp                = dep[['FIPS','NO2'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    depp                = pd.merge(depp, pmd, on='FIPS',how='left')
    depp['NO2'+dep_param].fillna(value=depp['NO2'+dep_param].mean(), inplace=True)
    df                  = pixels[['FIPS',scenario+'3','nlcd3']]
    df                  = pd.merge(df, depp, on='FIPS',how='left')
    df['NO2'+dep_param].fillna(value=df['NO2'+dep_param].mean(), inplace=True)
    result1             = (df[scenario+'3'] - df['nlcd3']) * df['NO2'+dep_param] * 1000 * df['mortality']
    result1.fillna(value=0.0, inplace=True)
    gdepp               = gdep[['FIPS','NO2'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    gdepp               = pd.merge(gdepp, pmd, on='FIPS',how='left')
    gdepp['NO2'+dep_param].fillna(value=gdepp['NO2'+dep_param].mean(), inplace=True)
    gdf                 = pixels[['FIPS',scenario+'1','nlcd1',scenario+'2','nlcd2',scenario+'5','nlcd5']]
    gdf                 = pd.merge(gdf, gdepp, on='FIPS',how='left')
    gdf['NO2'+dep_param].fillna(value=gdf['NO2'+dep_param].mean(), inplace=True)
    result2             =  (gdf[scenario+'5'] - gdf['nlcd5']) * gdf['NO2'+dep_param] * 1000 * gdf['mortality']
    result2.fillna(value=0.0, inplace=True)
    pdepp               = pdep[['FIPS','NO2'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    pdepp               = pd.merge(pdepp, pmd, on='FIPS',how='left')
    pdepp['NO2'+dep_param].fillna(value=pdepp['NO2'+dep_param].mean(), inplace=True)
    pdf                 = pixels[['FIPS',scenario+'1','nlcd1',scenario+'2','nlcd2',scenario+'5','nlcd5']]
    pdf                 = pd.merge(pdf, pdepp, on='FIPS',how='left')
    pdf['NO2'+dep_param].fillna(value=pdf['NO2'+dep_param].mean(), inplace=True)
    result3             =  (pdf[scenario+'2'] - pdf['nlcd2']) * pdf['NO2'+dep_param] * 1000 * pdf['mortality']
    result3.fillna(value=0.0, inplace=True)
    return (result1 + result2 + result3)

def getsoxdep(scenario, model, crf):
    depp                = dep[['FIPS','SO2'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'so2') & (isrm.model == model) & (isrm.crf == crf)]
    depp                = pd.merge(depp, pmd, on='FIPS',how='left')
    depp['SO2'+dep_param].fillna(value=depp['SO2'+dep_param].mean(), inplace=True)
    df                  = pixels[['FIPS',scenario+'3','nlcd3']]
    df                  = pd.merge(df, depp, on='FIPS',how='left')
    df['SO2'+dep_param].fillna(value=df['SO2'+dep_param].mean(), inplace=True)
    result1             = (df[scenario+'3'] - df['nlcd3']) * df['SO2'+dep_param] * 1000 * df['mortality']
    result1.fillna(value=0.0, inplace=True)
    gdepp               = gdep[['FIPS','SO2'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'so2') & (isrm.model == model) & (isrm.crf == crf)]
    gdepp               = pd.merge(gdepp, pmd, on='FIPS',how='left')
    gdepp['SO2'+dep_param].fillna(value=gdepp['SO2'+dep_param].mean(), inplace=True)
    gdf                 = pixels[['FIPS',scenario+'1','nlcd1',scenario+'2','nlcd2',scenario+'5','nlcd5']]
    gdf                 = pd.merge(gdf, gdepp, on='FIPS',how='left')
    gdf['SO2'+dep_param].fillna(value=gdf['SO2'+dep_param].mean(), inplace=True)
    result2             = (gdf[scenario+'5'] - gdf['nlcd5']) * gdf['SO2'+dep_param] * 1000 * gdf['mortality']
    result2.fillna(value=0.0, inplace=True)
    pdepp               = pdep[['FIPS','SO2'+dep_param]]
    pmd                 = isrm[(isrm.pollutant == 'so2') & (isrm.model == model) & (isrm.crf == crf)]
    pdepp               = pd.merge(pdepp, pmd, on='FIPS',how='left')
    pdepp['SO2'+dep_param].fillna(value=pdepp['SO2'+dep_param].mean(), inplace=True)
    pdf                 = pixels[['FIPS',scenario+'1','nlcd1',scenario+'2','nlcd2',scenario+'5','nlcd5']]
    pdf                 = pd.merge(pdf, pdepp, on='FIPS',how='left')
    pdf['SO2'+dep_param].fillna(value=pdf['SO2'+dep_param].mean(), inplace=True)
    result3             =  (pdf[scenario+'2'] - pdf['nlcd2']) * pdf['SO2'+dep_param] * 1000 * pdf['mortality']
    result3.fillna(value=0.0, inplace=True)
    return (result1 + result2 + result3)

def deposition(pol, scenario):
    switch = {
            'pm25':     getpmdep,
            'nox':      getnoxdep,
            'sox':      getsoxdep
    }
    func = switch.get(pol, lambda: "")
    return func(scenario, aqm, cr)

#print((pixels['proag3'] - pixels['nlcd3']).sum())
'''
print("change in deaths from deposition (+ is fewer deaths, - is more deaths)")
for x in scenarios:
    change               = (deposition('pm25', x)).sum() + (deposition('nox', x)).sum() + (deposition('sox', x)).sum()
    print(x, change)
'''

#########################################################################
############################### VALUATION ###############################
#########################################################################

# 1. Valuation of change in mortality risk attributable to PM2.5 exposure
# We use the EPA central estimate of $7.4 million ($2006) +/- one standard
# deviation.
# We use the BLS inflation calculator https://data.bls.gov/cgi-bin/cpicalc.pl
# to convert $2006 to $2021
vsl                     = epa_vsl * 1.40 * 1000000

# 2. Valuation of economic returns to land
# We use the following data from http://quickstats.nass.usda.gov/
# SURVEY > ECONOMICS > FARMS & LAND & ASSETS > AG LAND > ASSET VALUE >
# AG LAND, CROPLAND - ASSET VALUE, MEASURED IN $ / ACRE > TOTAL > STATE >
# 2001 > ANNUAL > YEAR
cropval                 = pd.read_csv("./inputs/val/croplandassetvalue.csv")
# NOTE: For the high crop demand scenario (proag), we want to use the following values instead:
#cropval                 = pd.read_csv("./inputs/val/croplandassetvalue_highcropdemand.csv")

# For several states, the value is "(D)" because there isn't enough data to
# anonymously report statistics. We replace these with the average.
cropval.replace(to_replace =" (D)", value =np.nan, inplace=True)
cropval.replace(to_replace =",", value ="", regex=True, inplace=True)
cropval.Value = cropval.Value.astype(float)
cropval['Value'].fillna(value=cropval['Value'].mean(), inplace=True)
# Get states (NaN is other states)
cropval['State ANSI'].replace(to_replace =np.nan, value ="0", inplace=True)
cropval['State ANSI']   = cropval['State ANSI'].astype(int) * 1000
# The values are in $ / acre, rather than $ / km2.
# Also, convert from $2001 to $2021 Using BLS inflation calculator:
# $1 (2001) = $1.49 (2021) 
cropval['Value']        = cropval['Value'] * 1.49 / 0.00404686
def getcropvals(scen):
    pixels['State ANSI']= (np.floor(pixels['FIPS']/1000)*1000).astype(int)
    df                  = pd.merge(cropval,pixels,on='State ANSI', how='left')
    df['cropchange']    = df[scen+'1'] - df['nlcd1']
#    df['Value'].replace({0: othercropstates}, inplace=True)
    return (df['cropchange'] * df['Value']).sum()

# Pasture
# SURVEY > ECONOMICS > FARMS & LAND & ASSETS > AG LAND > ASSET VALUE >
# AG LAND, PASTURELAND - ASSET VALUE, MEASURED IN $ / ACRE > TOTAL > STATE >
# 2021 > ANNUAL > YEAR
pastureval              = pd.read_csv("./inputs/val/pasturelandassetvalue.csv")
# For several states, the value is "(D)" because there isn't enough data to
# anonymously report statistics. We replace these with the average.
pastureval.replace(to_replace =" (D)", value =np.nan, inplace=True)
pastureval.replace(to_replace =",", value ="", regex=True, inplace=True)
pastureval.Value = pastureval.Value.astype(float)
pastureval['Value'].fillna(value=pastureval['Value'].mean(), inplace=True)

# Get states (NaN is other states)
pastureval['State ANSI'].replace(to_replace =np.nan, value ="0", inplace=True)
pastureval['State ANSI'] = pastureval['State ANSI'].astype(int) * 1000
otherpasturestates      = pastureval[pastureval['State ANSI'] == 0].Value
# The values are in $ / acre, rather than $ / km2.
# Also, convert from $2001 to $2021 Using BLS inflation calculator:
# $1 (2001) = $1.49 (2021) 
pastureval['Value']     = pastureval['Value'] / 0.00404686
def getpasturevals(scen):
    pixels['State ANSI']= (np.floor(pixels['FIPS']/1000)*1000).astype(int)
    df                  = pd.merge(pastureval,pixels,on='State ANSI', how='left')
    df['pasturechange']    = (df[scen+'2'] - df['nlcd2'])
    return (df['pasturechange'] * df['Value']).sum()


# 3. For other ecosystem services, read the data underlying Lawler et al. (2014)
# figures 2 and 4. This data was recorded using Web Plot Digitizer:
# https://automeris.io/WebPlotDigitizer/
lawler                  = pd.read_csv("./inputs/val/lawler-figs.csv")
# Here, the units are:
# food production:          10^14 kcal
# timber production         10^7 cubic feet
# carbon storage            10^8 Mg C

# Also, the policy scenarios (forest, urban, native) are with reference to
# the 'ref' scenario (1990s trends). The trend scenarios (ref, proag) are
# with reference to the 'nlcd' baseline.
# For clarity, here we make all the values with reference to nlcd:

forestlawler            = lawler.loc[[0, 3]].sum()
urbanlawler             = lawler.loc[[1, 3]].sum()
nativelawler            = lawler.loc[[2, 3]].sum()
reflawler               = lawler.loc[3]
proaglawler             = lawler.loc[4]

def lawler(scen):
    lawlerfigs = {
            'forest':   forestlawler,
            'urban':    urbanlawler,
            'native':   nativelawler,
            'ref':      reflawler,
            'proag':    proaglawler
    }
    obj = lawlerfigs.get(scen, lambda: "")
    return obj

# 3. Valuation of carbon sequestration
# Assuming the sequestered carbon would otherwise be CO2, we use the Social
# Cost of Carbon taken from the Technical Support Document: Social Cost of
# Carbon, Methane, and Nitrous Oxide Interim Estimates under Executive Order
# 13990, February 2021, Table ES-1. Available here:
# https://www.whitehouse.gov/wp-content/uploads/2021/02/TechnicalSupportDocument_SocialCostofCarbonMethaneNitrousOxide.pdf

def getscc(discount):
    scc                     = pd.read_csv("./inputs/val/scc.csv")
    # Select scenario/discount rate from '5avg', '3avg', '2.5avg', '3high'
    scc3                    = scc[discount][0]
    # Convert from $2020 to $2021 by x1.08
    scc3                    = scc3 * 1.08
    # This is the social cost of CO2 emission per metric ton. Convert to kg:
    scc3                    = scc3 / 1000
    # Convert to per mass of carbon sequestered, rather than per CO2 emitted:
    scc3                    = scc3 * 44.01/12.01
    return scc3

# 4. Valuation of timber production

# Nielsen, Plantinga, and Alig (2014) give county-level land prices for
# crop, pasture, and range conversion to forest.
nielsen                 = pd.read_csv("./inputs/val/nielsen2013.csv")

# Assume that the increase in forest land is all from managed forests.
# For each scenario, we find the change in forest land for each county:
#for scen in scenarios:
def forestval(scen):
    forest              = pixels[scen+'3'] - pixels['nlcd3']
    # Only find counties where there is an increase in forest:
    forest              = np.maximum(forest, np.zeros(len(forest)))
    # We only care about other land where there is a decrease:
    ag                  = pixels['nlcd1'] - pixels[scen+'1']
    ag                  = np.maximum(ag, np.zeros(len(ag)))
    pasture             = pixels[scen+'2'] - pixels['nlcd2']
    pasture             = np.maximum(pasture, np.zeros(len(pasture)))
    rangeland           = pixels[scen+'5'] - pixels['nlcd5']
    rangeland           = np.maximum(rangeland, np.zeros(len(rangeland)))
    totaldecrease       = ag + pasture + rangeland
    # This gives the proportion of land decrease that is from each land use type
    ag                  = ag / totaldecrease
    pasture             = pasture / totaldecrease
    rangeland           = rangeland / totaldecrease
    ag.replace(to_replace =np.nan, value =0.0, inplace=True)
    pasture.replace(to_replace =np.nan, value =0.0, inplace=True)
    rangeland.replace(to_replace =np.nan, value =0.0, inplace=True)

    frame               = { 'FIPS': pixels['FIPS'], 'ag': ag, 'pasture': pasture, 'rangeland': rangeland, 'forest': forest}
    convs               = pd.DataFrame(frame)
    # Merge with nielsen
    convs               = pd.merge(convs, nielsen, on='FIPS',how='left')
    value               = (convs['croptoforest'] * convs['ag'] * convs['forest']) + (convs['pasturetoforest'] * convs['pasture'] * convs['forest']) + (convs['rangetoforest'] * convs['rangeland'] * convs['forest'])
    # 1 acre            = 0.00404686 km²
    # From CPI Inflation Calculator, $1 in 1997 is $1.75 in 2021 (BLS?)
    return abs(value.sum()) * (1.75 /0.00404686)
#    print(scen, abs(value.sum()) * (1.75 /0.00404686) / 1000000000)

# Values from timber land in 2021$USBillion
# forest 108.95656496281748
# native 51.63965423692296
# proag 22.2603984888095
# ref 47.78297595808377
# urban 53.61512858868377

#########################################################################
################################ RESULTS ################################
#########################################################################

# 1. We want to know the change in ecosystem services for each scenario.
# Our time horizon is 50 years
horizon                 = 50
onebillion              = 1000000000.0
# our discount rate is 3%:
discountf               = 0.97

# This incorporates the 20-year cessation lag in deaths
cessation = [0.3, 0.425 ,0.55, 0.675, 0.8, 0.813333333, 0.826666667, 0.84, 0.853333333,
        0.866666667, 0.88, 0.893333333, 0.906666667, 0.92, 0.933333333,
        0.946666667, 0.96, 0.973333333, 0.986666667, 0.999999997, 0.999999993,
        0.99999999, 0.999999987, 0.999999983, 0.99999998, 0.999999977,
        0.999999973, 0.99999997, 0.999999967,0.999999963,0.99999996,
        0.999999957,0.999999953,0.99999995,0.99999995, 0.99999995, 0.99999995,
        0.99999995, 0.99999995, 0.99999995, 0.99999995, 0.99999995, 0.99999995,
        0.99999995, 0.99999995, 0.99999995,0.99999995,0.99999995,0.99999995,0.99999995]

import math
def get_sensitivity(perc):
    # The 7 refers to the number of years between 2005 and 2012 (Holt et al., 2015)
    ff = math.exp((math.log(1-perc))/7)
    i = [1.0]
    for j in range(0,50):
        i.append(i[-1]*ff)
    return i

nh3_sensitivity = get_sensitivity(0.38)
nox_sensitivity = get_sensitivity(0.29)
sox_sensitivity = get_sensitivity(-0.23)

#nh3_sensitivity = [1.0, 0.933988803, 0.872335085, 0.814751202, 0.7609685, 0.710736059,
#                   0.663819521, 0.62, 0.579073058, 0.540847753, 0.505145745, 0.47180047,
#                   0.440656356, 0.411568103, 0.3844, 0.359025296, 0.335325607, 0.313190362,
#                   0.292516291, 0.273206941, 0.255172224, 0.238328, 0.222595684, 0.207901876,
#                   0.194178024, 0.181360101, 0.169388303, 0.158206779, 0.14776336, 0.138009324,
#                   0.128899163, 0.120390375, 0.112443262, 0.105020748, 0.098088203, 0.091613283,
#                   0.085565781, 0.079917481, 0.074642033, 0.069714823, 0.065112864, 0.060814686,
#                   0.056800236, 0.053050784, 0.049548838, 0.04627806, 0.04322319, 0.040369976,
#                   0.037705105, 0.035216146, 0.032891486]


# so, totfactor gives the multiplier based on the horizon, discount rate, and cessation lag.
totfactor = 0.0
count = 0
while (count < horizon):
    totfactor = totfactor + pow(discountf,count) * cessation[count]
    count = count + 1
#print(totfactor)

rows                    = []
for x in scenarios:
    # air quality cost
    change              = deaths('ag', x) + deaths('forest', x) + deaths('pasture', x) - tot_base - ((deposition('pm25', x)).sum() + (deposition('nox', x)).sum() + (deposition('sox', x)).sum())
    # climate cost
    # We want the model average using the 3% discount rate
    # Convert 10^8 MgC to kgC
    carboncost          = lawler(x)['carbon'] * getscc(scc_param) * 100000000000
    # economic returns to land
    forestv             = forestval(x)
    cropv               = getcropvals(x)
    pasturev            = getpasturevals(x)
    # append data
    rows.append([x, change * -1.0 * vsl * totfactor / onebillion, carboncost / onebillion, forestv / onebillion, cropv / onebillion, pasturev/onebillion])

results                 = pd.DataFrame(rows, columns=["scenario", "aq $", "carbon $", "forest $", "crop $", "pasture $"])
print(results)
#results.to_csv("./outputs/fig1.csv")


#########################################################################
############################## SENSITIVITY ##############################
#########################################################################
# Because of changes in background pollutant concentrations over time, the sensitivity
# of secondary PM2.5 formation to NH3 or NOx emissions changes over time.
# We estimate the effects of this on a very conservative scenario, where:
# Deposition goes to zero
# NH3 gets 38% smaller every 7 years
# NOx gets 29% smaller every 7 years

def get_tot_factor_sens(sens):
    totfactor = 0.0
    count = 0
    while (count < horizon):
        totfactor = totfactor + pow(discountf,count) * cessation[count] * sens[count]
        count = count + 1
    return totfactor

totfactor_nh3 = get_tot_factor_sens(nh3_sensitivity)
totfactor_nox = get_tot_factor_sens(nox_sensitivity)
totfactor_sox = get_tot_factor_sens(sox_sensitivity)

def agotherdeaths(scenario, model, crf):
    # From PM2.5
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    pmd                 = pd.merge(pixels, pmd, on='FIPS',how='left')
    resultpm            = pmd[scenario+'1'] * pmd['dust-emis'] * pmd['mortality']
    # From windblown dust
    wbd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    wbd                 = pd.merge(pixels, wbd, on='FIPS',how='left')
    resultwb            = wbd[scenario+'1'] * wbdust['Agricultural'] * wbd['mortality']
    # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS', how='left')
    vocd                = pd.merge(vocd, bvoc, on='FIPS', how='left')
    bvoc['Ag']          = bvoc['Ag'].replace(np.nan, bvoc['Ag'].mean())
    resultvoc           =  vocd['Ag'] * vocd[scenario+'1'] * vocd['mortality']
    return resultpm.sum() + resultvoc.sum() + resultwb.sum()

def agpmdeaths(scenario, model, crf):
    # From PM2.5
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    pmd                 = pd.merge(pixels, pmd, on='FIPS',how='left')
    resultpm            = pmd[scenario+'1'] * pmd['dust-emis'] * pmd['mortality']
    # From windblown dust
    wbd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    wbd                 = pd.merge(pixels, wbd, on='FIPS',how='left')
    resultwb            = wbd[scenario+'1'] * wbdust['Agricultural'] * wbd['mortality']
    return resultpm.sum() + resultwb.sum()

def agvocdeaths(scenario, model, crf):
    # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS', how='left')
    vocd                = pd.merge(vocd, bvoc, on='FIPS', how='left')
    bvoc['Ag']          = bvoc['Ag'].replace(np.nan, bvoc['Ag'].mean())
    resultvoc           =  vocd['Ag'] * vocd[scenario+'1'] * vocd['mortality']
    return resultvoc.sum()

def agnoxdeaths(scenario, model, crf):
    # From NOx
    noxd                = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    noxd                = pd.merge(pixels, noxd, on='FIPS',how='left')
    resultnox           = (noxd[scenario+'1'] * cropnox) * noxd['mortality']
    return resultnox.sum()

def agnh3deaths(scenario, model, crf):
    # From NH3
    nh3d                = isrm[(isrm.pollutant == 'nh3') & (isrm.model == model) & (isrm.crf == crf)]
    nh3d                = pd.merge(pixels, nh3d, on='FIPS',how='left')
    resultnh3           = (nh3d[scenario+'1'] * nh3d['nh3-emis']) * nh3d['mortality']
    return resultnh3.sum()

def pasturenoxdeaths(scenario, model, crf):
    # From NOx
    noxd                = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    noxd                = pd.merge(pixels, noxd, on='FIPS',how='left')
    resultnox          = ((noxd[scenario+'2'] + noxd[scenario+'5']) * grassnox) * noxd['mortality']
    return resultnox.sum()

def pasturenh3deaths(scenario, model, crf):
    # From NH3
    nh3d                = isrm[(isrm.pollutant == 'nh3') & (isrm.model == model) & (isrm.crf == crf)]
    nh3d                = pd.merge(pixels, nh3d, on='FIPS',how='left')
    resultnh3           = (nh3d[scenario+'2'] * nh3d['pastureemis']) * nh3d['mortality']
    return resultnh3.sum()

def pasturepmdeaths(scenario, model, crf):
    # From Windblown dust
    wbd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    wbd                 = pd.merge(pixels, wbd, on='FIPS',how='left')
    resultwb            = (wbd[scenario+'5'] + wbd[scenario+'2']) * wbdust['Grasslands'] * wbd['mortality']
    return resultwb.sum()

def pastureotherdeaths(scenario, model, crf):
    # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS')
    vocd                = pd.merge(vocd, bvoc, on='FIPS', how='left')
    bvoc['Grass']       = bvoc['Grass'].replace(np.nan, bvoc['Grass'].mean())
    resultvoc           =  vocd['Grass'] * (vocd[scenario+'5'] + vocd[scenario+'2']) * vocd['mortality']
    return resultvoc.sum()

def forestdeaths(scenario, model, crf):
   # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS', how='left')
    vocd                 = pd.merge(vocd, bvoc, on='FIPS',how='left')
    bvoc['Forest']      = bvoc['Forest'].replace(np.nan, bvoc['Forest'].mean())
    resultvoc           =  vocd['Forest'] * vocd[scenario+'3'] * vocd['mortality']
    return resultvoc.sum()

def deaths_sens(lu, scenario):
    switch = {
            'forest':           forestdeaths,
            'ag_nox':           agnoxdeaths,
            'ag_nh3':           agnh3deaths,
            'ag_other':         agotherdeaths,
            'ag_pm':            agpmdeaths,
            'ag_voc':           agvocdeaths,
            'pasture_nox':      pasturenoxdeaths,
            'pasture_nh3':      pasturenh3deaths,
            'pasture_other':    pastureotherdeaths,
            'pasture_pm':       pasturepmdeaths,
    }
    func = switch.get(lu, lambda: "")
    return func(scenario, aqm, cr)

ag_nox_base                 = deaths_sens('ag_nox','nlcd')
ag_nh3_base                 = deaths_sens('ag_nh3','nlcd')
ag_other_base               = deaths_sens('ag_other','nlcd')
ag_pm_base                  = deaths_sens('ag_pm', 'nlcd')
ag_voc_base                 = deaths_sens('ag_voc', 'nlcd')
forest_base                 = deaths_sens('forest','nlcd')
pasture_nox_base            = deaths_sens('pasture_nox','nlcd')
pasture_nh3_base            = deaths_sens('pasture_nh3','nlcd')
pasture_other_base          = deaths_sens('pasture_other','nlcd')
pasture_pm_base             = deaths_sens('pasture_pm','nlcd')

tot_nox_base                = ag_nox_base + pasture_nox_base
tot_nh3_base                = ag_nh3_base + pasture_nh3_base
tot_other_base              = ag_pm_base + ag_voc_base + forest_base + pasture_other_base + pasture_pm_base

# Change in deaths relative to 2001
scenarios               = ["forest", "native", "proag", "ref", "urban"]
#for x in scenarios:
#    change              = deaths('ag', x) + deaths('forest', x) + deaths('pasture', x) - tot_base
#    print(x, change)

#print(totfactor)        # 22.850534376808614
#print(totfactor_nox)    # 10.148278341501431
#print(totfactor_nh3)    # 8.004808365566706

# We also want to take into account the trend in changes in background pollution:

def get_tot_factor_dep(sens,tf):
    totfactor = 0.0
    count = 0
    while (count < horizon):
        totfactor = totfactor + pow(discountf,count) * cessation[count] * sens[count] * tf[count]
        count = count + 1
    return totfactor

pm_dep_profile          = get_sensitivity(0.0246306407730926)
dep_totfactor_pm        = get_tot_factor_sens(pm_dep_profile)
nox_dep_profile         = get_sensitivity(0.0228851659674423)
dep_totfactor_nox       = get_tot_factor_dep(nox_dep_profile, nox_sensitivity)
sox_dep_profile         = get_sensitivity(0.0918885856004334)
dep_totfactor_sox       = get_tot_factor_dep(sox_dep_profile, sox_sensitivity)
#dep_totfactor_sox       = get_tot_factor_sens(sox_dep_profile)


#print("Sensitivity: dep_totfactors")
#print(dep_totfactor_pm)
#print(dep_totfactor_nox)
#print(dep_totfactor_sox)

'''
print("Sensitivity: Changing deposition via sensitivity and reduction in background pollution")
rows                    = []
for x in scenarios:
    # air quality cost
    change_nox              = (deaths_sens('ag_nox', x) + deaths_sens('pasture_nox', x) - tot_nox_base) * (-1.0 * vsl * totfactor_nox / onebillion) - ((deposition('nox', x)).sum() * dep_totfactor_nox * -1.0 * vsl / onebillion)
    change_nh3              = (deaths_sens('ag_nh3', x) + deaths_sens('pasture_nh3', x) - tot_nh3_base) * -1.0 * vsl * totfactor_nh3 / onebillion
    change_sox              =  (deposition('sox', x)).sum()  * vsl * dep_totfactor_sox / onebillion
    change_other              = (deaths_sens('pasture_other', x) + deaths_sens('ag_other', x) + deaths_sens('forest', x) - tot_other_base) * (-1.0 * vsl * totfactor / onebillion) - ((deposition('pm25', x)).sum() * (-1.0 * vsl * dep_totfactor_pm / onebillion))
#    change = change_nox
    change = change_nox + change_nh3 + change_other + change_sox
    rows.append([x,change])
print(rows)
'''
'''
print("Sensitivity: Changing deposition only via sensitivity")
rows                    = []
for x in scenarios:
    # air quality cost
    change_nox              = (deaths_sens('ag_nox', x) + deaths_sens('pasture_nox', x) - tot_nox_base) * (-1.0 * vsl * totfactor_nox / onebillion) - ((deposition('nox', x)).sum() * totfactor_nox * -1.0 * vsl / onebillion)
    change_nh3              = (deaths_sens('ag_nh3', x) + deaths_sens('pasture_nh3', x) - tot_nh3_base) * -1.0 * vsl * totfactor_nh3 / onebillion
    change_sox              =  (deposition('sox', x)).sum()  * vsl * totfactor_sox / onebillion
    change_other              = (deaths_sens('pasture_other', x) + deaths_sens('ag_other', x) + deaths_sens('forest', x) - tot_other_base) * (-1.0 * vsl * totfactor / onebillion) - ((deposition('pm25', x)).sum() * (-1.0 * vsl * totfactor / onebillion))
#    change = change_nox
    change = change_nox + change_nh3 + change_other + change_sox
    rows.append([x,change])
print(rows)
'''

#########################################################################
########################## DEATHS BY POLLUTANT ##########################
#########################################################################

# For each scenario, we also want to print deaths by pollutant, for both
# the main results and the sensitivity. 

#1. Sensitivity results by pollutant
'''
print("Sensitivity results by pollutant (including deposition change)")
print("PM2.5")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_pm', x) + deaths_sens('pasture_pm', x) - ag_pm_base - pasture_pm_base) * (-1.0 * vsl * totfactor / onebillion) - (deposition('pm25', x).sum() * (-1.0 * vsl * dep_totfactor_pm / onebillion))
    rows.append([x,change])
print(rows)

print("VOC")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_voc', x) + deaths_sens('pasture_other', x) + deaths_sens('forest', x) - ag_voc_base - pasture_other_base - forest_base) * (-1.0 * vsl * totfactor / onebillion)
    rows.append([x,change])
print(rows)

print("NOx")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_nox', x) + deaths_sens('pasture_nox', x) - ag_nox_base - pasture_nox_base) * (-1.0 * vsl * totfactor_nox / onebillion) - (deposition('nox', x).sum() * (-1.0 * vsl * dep_totfactor_nox / onebillion))
    rows.append([x,change])
print(rows)

print("NH3")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_nh3', x) + deaths_sens('pasture_nh3', x) - ag_nh3_base - pasture_nh3_base) * (-1.0 * vsl * totfactor_nh3 / onebillion)
    rows.append([x,change])
print(rows)

print("SOx")
rows                    = []
for x in scenarios:
    change = (deposition('sox', x).sum() * (vsl * dep_totfactor_sox / onebillion))
    rows.append([x,change])
print(rows)
'''
#2. Main results by pollutant
'''
print("Main results by pollutant")
print("PM2.5")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_pm', x) + deaths_sens('pasture_pm', x) - ag_pm_base - pasture_pm_base) * (-1.0 * vsl * totfactor / onebillion) - (deposition('pm25', x).sum() * (-1.0 * vsl * totfactor / onebillion))
    rows.append([x,change])
print(rows)

print("VOC")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_voc', x) + deaths_sens('pasture_other', x) + deaths_sens('forest', x) - ag_voc_base - pasture_other_base - forest_base) * (-1.0 * vsl * totfactor / onebillion)
    rows.append([x,change])
print(rows)

print("NOx")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_nox', x) + deaths_sens('pasture_nox', x) - ag_nox_base - pasture_nox_base) * (-1.0 * vsl * totfactor / onebillion) - (deposition('nox', x).sum() * (-1.0 * vsl * totfactor / onebillion))
    rows.append([x,change])
print(rows)

print("NH3")
rows                    = []
for x in scenarios:
    change = (deaths_sens('ag_nh3', x) + deaths_sens('pasture_nh3', x) - ag_nh3_base - pasture_nh3_base) * (-1.0 * vsl * totfactor / onebillion)
    rows.append([x,change])
print(rows)

print("SOx")
rows                    = []
for x in scenarios:
    change = (deposition('sox', x).sum() * (vsl * totfactor / onebillion))
    rows.append([x,change])
print(rows)
'''
#########################################################################
################################ OUTPUTS ################################
#########################################################################
#import geopandas as gpd
#import matplotlib.pyplot as plt

# Figure 1 shows the ecosystem services for each scenario, with the correct
# baseline (policies should be relative to ref rather than nlcd):
#results_toplot          = results

# Maps
# Forest deaths by FIPS (including deposition)
def forestdeaths_perkm(model, crf):
    # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                 = pd.merge(bvoc, vocd, on='FIPS',how='left')
    resultvoc            = vocd['Forest'] * vocd['mortality']
    return resultvoc

fdpkm                   = pd.DataFrame(forestdeaths_perkm(aqm, cr), columns = ['forestdeaths'])
fdpkm['FIPS']           = bvoc['FIPS']
#fdpkm.to_csv("forestdeaths.csv")

# Also, you want deposition
def getpmdep_perkm(model, crf):
    depp                = dep[['FIPS','PM2.5mean']]
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    depp                = pd.merge(depp, pmd, on='FIPS',how='right')
    return depp['PM2.5mean'] * 1000 * depp['mortality']

def getnoxdep_perkm(model, crf):
    depp                = dep[['FIPS','NO2mean']]
    pmd                 = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)]
    depp                = pd.merge(depp, pmd, on='FIPS',how='right')
    return depp['NO2mean'] * 1000 * depp['mortality']

def getsoxdep_perkm(model, crf):
    depp                = dep[['FIPS','SO2mean']]
    pmd                 = isrm[(isrm.pollutant == 'so2') & (isrm.model == model) & (isrm.crf == crf)]
    depp                = pd.merge(pmd, depp, on='FIPS',how='left')
#    depp['SO2mean']  = depp['SO2mean'].replace(np.nan, depp['SO2mean'].mean())
    return depp['SO2mean'] * 1000 * depp['mortality']

def deposition_perkm(pol):
    switch = {
            'pm25':  getpmdep_perkm,
            'nox':    getnoxdep_perkm,
            'sox':   getsoxdep_perkm
    }
    func = switch.get(pol, lambda: "")
    return func(aqm, cr)

totdep                  = deposition_perkm('pm25') + deposition_perkm('nox') + deposition_perkm('sox')
totdepdf                = pd.DataFrame(totdep, columns = ['totdep'])
pmd                     = isrm[(isrm.pollutant == 'so2') & (isrm.model == aqm) & (isrm.crf == cr)].reset_index()
pmd                     = pd.DataFrame(pmd['FIPS'], columns = ['FIPS'])
totdepdf['FIPS']        = pmd['FIPS']

# Deaths minus deposition:
forest_aq               = pd.merge(totdepdf, fdpkm, on='FIPS', how='left')
forest_aq['total']      = forest_aq['forestdeaths'] - forest_aq['totdep']

# Ag deaths by FIPS
def agdeaths_perkm(model, crf):
    # From PM2.5
    pmd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)].reset_index()
    pmd                 = pd.merge(pixels, pmd, on='FIPS', how='left')
    resultpm            = pmd['dust-emis'] * pmd['mortality']
    # From Windblownust
    wbd                 = isrm[(isrm.pollutant == 'pm25') & (isrm.model == model) & (isrm.crf == crf)]
    wbd                 = pd.merge(pixels, wbd, on='FIPS',how='left')
    resultwb            = wbdust['Agricultural'] * wbd['mortality']
    # From NOx
    noxd                = isrm[(isrm.pollutant == 'nox') & (isrm.model == model) & (isrm.crf == crf)].reset_index()
    noxd                = pd.merge(pixels, noxd, on='FIPS', how='left')
    resultnox           = cropnox * noxd['mortality']
    # From NH3
    nh3d                = isrm[(isrm.pollutant == 'nh3') & (isrm.model == model) & (isrm.crf == crf)].reset_index()
    nh3d                = pd.merge(pixels, nh3d, on='FIPS', how='left')
    resultnh3           = nh3d['nh3-emis'] * nh3d['mortality']
    # From VOC
    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
    vocd                = pd.merge(pixels, vocd, on='FIPS', how='left')
    bvoc['Ag']          = bvoc['Ag'].replace(np.nan, bvoc['Ag'].mean())
    vocd                = pd.merge(vocd, bvoc, on='FIPS', how='left')
    resultvoc           =  vocd['Ag'] * vocd['mortality']
#    vocd                = isrm[(isrm.pollutant == 'voc') & (isrm.model == model) & (isrm.crf == crf)]
#    vocd                = pd.merge(pixels, vocd, on='FIPS')
#    resultvoc           = cropvoc * vocd['mortality']
    return resultpm + resultnox + resultnh3 + resultwb + resultvoc

agd                     = pd.DataFrame(agdeaths_perkm(aqm, cr), columns = ['agdeaths'])
agd['FIPS']             = pixels['FIPS']

# Deaths from 1km conversion from forest to ag
forest_to_ag            = pd.merge(agd,forest_aq, on='FIPS',how='left')
# Convert everything to $
forest_to_ag['agdeaths'] = forest_to_ag['agdeaths'] * -1.0 * vsl
forest_to_ag['total']   = forest_to_ag['total'] * -1.0 * vsl
forest_to_ag['conv']    = forest_to_ag['agdeaths'] - forest_to_ag['total']

forest_to_ag.to_csv("./outputs/deathsperkm.csv")
