from IPython.display import Markdown
from IPython.display import display

import os
import warnings
import gc


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import datetime as dt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit, train_test_split
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.svm import LinearSVR

def test():
    print(45)
    
    
COL_NAMES = {
    'parcelid': 'parcel_id',  # un identifiant unique pour chaque maison
    'airconditioningtypeid': 'cooling_id',  # type de ventilation 1~13
    'architecturalstyletypeid': 'architecture_style_id',  # type d'architecture 1~27
    'basementsqft': 'basement_sqft',  # surface de l'espace de vie
    'bathroomcnt': 'bathroom_cnt',  # nombre de salles de bain
    'bedroomcnt': 'bedroom_cnt',  # nombre de chambre
    'buildingclasstypeid': 'framing_id',  # type de structure solide = 1 , 1~5
    'buildingqualitytypeid': 'quality_id',  # état global du batiment (lowest..highest)
    'calculatedbathnbr': 'bathroom_cnt_calc',  # Same meaning as 'bathroom_cnt'?
    'decktypeid': 'deck_id',  # type de pont
    'finishedfloor1squarefeet': 'floor1_sqft',  # surface du rez-de-chaussée
    'calculatedfinishedsquarefeet': 'finished_area_sqft_calc',  # surface totale habitable
    'finishedsquarefeet12': 'finished_area_sqft',  # surface totale habitable
    'finishedsquarefeet13': 'perimeter_area',  # perimetre surface habitanle
    'finishedsquarefeet15': 'total_area',  # surface totale
    'finishedsquarefeet50': 'floor1_sqft_unk',  # Same meaning as 'floor1_sqft'?
    'finishedsquarefeet6': 'base_total_area',  # surface totale
    'fips': 'fips',  # Federal Information Processing Standard code https://en.wikipedia.org/wiki/FIPS_county_code 
    'fireplacecnt': 'fireplace_cnt',  # Nombre de cheminée
    'fullbathcnt': 'bathroom_full_cnt',  # Nombre de SDB avec baignoire
    'garagecarcnt': 'garage_cnt',  # Nombre de garages
    'garagetotalsqft': 'garage_sqft',  # surface totale des garages
    'hashottuborspa': 'spa_flag',  # spa ou jacuzzi
    'heatingorsystemtypeid': 'heating_id',  # type de chauffage, 1~25
    'latitude': 'latitude',  # latitude of the middle of the parcel multiplied by 1e6
    'longitude': 'longitude',  # longitude of the middle of the parcel multiplied by 1e6
    'lotsizesquarefeet': 'lot_sqft',  # surface du terrain
    'poolcnt': 'pool_cnt', # Nombre de piscines
    'poolsizesum': 'pool_total_size',  # surface totale des piscines
    'pooltypeid10': 'pool_unk_1', #spa ou jacuzzi
    'pooltypeid2': 'pool_unk_2', # piscine + spa ou jacuzzi
    'pooltypeid7': 'pool_unk_3', # piscine sans spa ou jacuzzi
    'propertycountylandusecode': 'county_landuse_code', #code du comté
    'propertylandusetypeid': 'landuse_type_id' ,  # Type de terres, 25 categories
    'propertyzoningdesc': 'zoning_description',  # Usages des terres autorisés
    'rawcensustractandblock': 'census_1',
    'regionidcity': 'city_id',  # ville
    'regionidcounty': 'county_id',  # comté
    'regionidneighborhood': 'neighborhood_id',  # quartier
    'regionidzip': 'region_zip', # code postal
    'roomcnt': 'room_cnt',  # nombre totale de pièces
    'storytypeid': 'story_id',  # type d'étages, 1~35
    'threequarterbathnbr': 'bathroom_small_cnt',  # Nombre de salle de bains réduites (douche+lavabo+wc)
    'typeconstructiontypeid': 'construction_id',  # materiaux de base, 1~18
    'unitcnt': 'unit_cnt',  # Number of units the structure is built into (2=duplex, 3=triplex, etc)
    'yardbuildingsqft17': 'patio_sqft',  # Patio in yard
    'yardbuildingsqft26': 'storage_sqft',  # Storage shed/building in yard
    'yearbuilt': 'year_built',  # The year the principal residence was built
    'numberofstories': 'story_cnt',  # nombre d'étage
    'fireplaceflag': 'fireplace_flag',  # présence d'une cheminée
    'structuretaxvaluedollarcnt': 'tax_structure',
    'taxvaluedollarcnt': 'tax_parcel',
    'assessmentyear': 'tax_year',  # The year of the property tax assessment (2015 for 2016 data)
    'landtaxvaluedollarcnt': 'tax_land',
    'taxamount': 'tax_property',
    'taxdelinquencyflag': 'tax_overdue_flag',  # Property taxes are past due as of 2015
    'taxdelinquencyyear': 'tax_overdue_year',  # Year for which the unpaid propert taxes were due
    'censustractandblock': 'census_2'
}


def rename_columns(df):
    df.rename(columns=COL_NAMES, inplace=True)

def md(input):
    display(Markdown(input))

def step(input):
    return md(f"✅ *{input}*")

def kv(key, value):
    return md("**{}** : {}".format(key, value))

#function to get all info in one go
def full_info(df):
    df_column=[]
    df_dtype=[]
    df_null=[]
    df_nullc=[]
    df_mean=[]
    df_median=[]
    df_std=[]
    df_min=[]
    df_max=[]
    df_uniq=[]
    for col in df.columns: 
        df_column.append( col)
        df_dtype.append( df[col].dtype)
        df_null.append( round(100 * df[col].isnull().sum(axis=0)/len(df[col]),2))
        df_nullc.append( df[col].isnull().sum(axis=0))
        df_uniq.append( df[col].nunique()) if df[col].dtype == 'object' else df_uniq.append( NaN)
        df_mean.append(  '{0:.2f}'.format(df[col].mean())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_mean.append( NaN)
        df_median.append( '{0:.2f}'.format(df[col].median())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_median.append( NaN)
        df_std.append( '{0:.2f}'.format(df[col].std())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_std.append( NaN)
        df_max.append( '{0:.2f}'.format(df[col].max())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_max.append( NaN)
        df_min.append( '{0:.2f}'.format(df[col].min())) if df[col].dtype == 'int64' or df[col].dtype == 'float64' else df_min.append( NaN)
    return pd.DataFrame(data = {'ColName': df_column, 'ColType': df_dtype, 'NullCnt': df_nullc, 'NullCntPrcntg': df_null,  'Min': df_min, 'Max': df_max, 'Mean': df_mean, 'Med': df_median, 'Std': df_std, 'UniqCnt': df_uniq})

def float_64_to_32(df):
    for c, dtype in zip(df.columns, df.dtypes):
        if dtype == np.float64:
            df[c] = df[c].astype(np.float32)
            
def explode_date(df, col_name, prefix, drop = True):
    df["{}_year".format(prefix)] = df[col_name].dt.year
    df["{}_month".format(prefix)] = df[col_name].dt.month
    df["{}_day".format(prefix)] = df[col_name].dt.day
    df["{}_quarter".format(prefix)] = df[col_name].dt.quarter
    if drop:
        df.drop(col_name, inplace=True, axis=1)
    return df

def drop_features(df, useless, missing):
    unused_feature_list = useless + missing
    return df.drop(unused_feature_list, axis=1, errors='ignore')

def cat_to_code(df):
    object_type = df.select_dtypes(include=['object']).columns.values
    df[object_type] = df[object_type].astype('category')
    for column in object_type:
        df[column] = df[column].cat.codes
        

def linear_regressor(X_train, Y_train, X_test, Y_test):
    regressor = LinearRegression(fit_intercept=True)
    model = regressor.fit(X_train, Y_train)
    pred_LR = regressor.predict(X_test)
    get_results(Y_test, pred_LR, "Linear Regressor")
    # Returns the trained model
    return model

def get_results(Y_test, predictions, model_name):
    mae_RFR = mean_absolute_error(predictions, Y_test)
    print ("MAE: {}".format(mae_RFR))
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(ncols=2, sharey=False, figsize=(15,5))
    sns.residplot(predictions, Y_test, color="g", ax=axs[0]).set_title("Residuals plot of " + model_name)
    sns.scatterplot(x=Y_test, y=predictions, ax=axs[1]).set_title("Model Error")
    axs[1].set(xlabel='True Values', ylabel='Predicted Values')
    
def reduce(df, mask):
    return df[df.eval(mask)]