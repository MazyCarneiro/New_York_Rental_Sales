
import pandas as pd  #Data Analysis and structure library 
import csv           #Implements classes to read and write tabular data in csv format
import numpy as np   #Scientific Library
import scipy as sp   #Scientific and technical Computing library
import scipy.stats as stats

filename = "NewYork_RentalSales.csv"
#https://public.enigma.com/datasets/new-york-city-property-sales/fd6efa37-2dcd-4294-8795-a0e6044f15b4

#Load dataset
#By opening the csv file in this format using pandas all the empty values become NaN.
df = pd.read_csv(filename, index_col=None) 
#The first and the last two columns have been elimiated
df.head(10)
#Filling all the missing values with 0
df = df.fillna(0)

print("Data Dimensions are: ", df.shape, '\n')
#Summary of thr dataframe
df.info()
#Detect any missing value
df.isnull().sum()

len(df)

df.describe()

#skewness and kurtosis`
print("Skewness:%f"% df.sale_price.skew())
print("Kurtosis %f" % df.sale_price.kurt())

table = pd.crosstab( df.borough_code_definition,df.sale_price,
                    margins=True )
table = table.sort_values(by='All', ascending=False)
table.drop('All', inplace=True)
table

A = df.loc[df['borough_code_definition'] == 'Brooklyn']
A.describe()

df.sale_price.median()
sum(df.duplicated(df.columns))

df.borough_code_definition.value_counts()

df.neighborhood.value_counts()

df.building_class_at_present_code_definition.value_counts()

#SALE DATE is object but should be datetime
df.sale_date = pd.to_datetime(df.sale_date , errors='coerce')
#Both TAX CLASS attributes should be categorical
df.tax_class_at_time_of_sale_code_definition = df.tax_class_at_time_of_sale_code_definition.astype('category')
df.tax_class_at_present_code_definition = df.tax_class_at_present_code_definition.astype('category')
df.zipcode = df.zipcode.astype('category')

df.info()

df.describe(include='all')

df.year_built.mean()

df.year_built.describe()

# NEW COLUMN ADDED TO THE DATAFRAME
df['building_age'] = 2017 - df.year_built
df.building_age.describe()

#SALE DATE is object but should be datetime
df['sale_date']    = pd.to_datetime(df['sale_date'], errors='coerce')
df['sale_year']    = df['sale_date'].dt.year
df['sale_month']   = df['sale_date'].dt.month
df['sale_day']     = df['sale_date'].dt.day

df.sale_month.value_counts()

df.sale_day.value_counts()

df.zipcode.value_counts()

df.zipcode.value_counts()

df.residential_units.value_counts()

df.residential_units.describe()

df.borough_code_definition.describe()

df.borough_code.describe()

df[df['total_units'] == df['commercial_units'] + df['residential_units']]

df.residential_units.describe()

df.commercial_units.describe()

df.exempt_value_total.describe()

df['Above_Below_median'] = np.where((df['sale_price'] > df['sale_price'].median()), "1", "0")


A_B = df[['borough_code_definition','sale_price', 'Above_Below_median','sale_month']]

A_B.sale_month.mode()

A_B.Above_Below_median.value_counts()

A_B.mode()

A_B.sale_month.value_counts(ascending=False)

A_B.borough_code_definition.value_counts()

Mhttan = df.loc[df['borough_code_definition'] == 'Manhattan']
Mhttan.sale_price.median() 

Bklyn = df.loc[df['borough_code_definition'] == 'Brooklyn']
Bklyn.sale_price.median()

Qn = df.loc[df['borough_code_definition'] == 'Queens']
Qn.sale_price.median()

Bx = df.loc[df['borough_code_definition'] == 'Bronx']
Bx.sale_price.median()

SI = df.loc[df['borough_code_definition'] == 'Staten Island']
SI.sale_price.median()


#Making columns that were objects/Strings into categorical columns

df['Borough'] = df['borough_code_definition'].astype('category')
df["Borough"] = df["Borough"].cat.codes

df['Neighborhood'] = df['neighborhood'].astype('category')
df["Neighborhood"] = df["Neighborhood"].cat.codes


df['Building_class_category'] = df['building_class_category_code_definition'].astype('category')
df["Building_class_category"] = df["Building_class_category"].cat.codes


df['Tax_class_at_present'] = df['tax_class_at_present_code_definition'].astype('category')
df["Tax_class_at_present"] = df["Tax_class_at_present"].cat.codes


df['Building_class_at_present'] = df['building_class_at_present_code_definition'].astype('category')
df["Building_class_at_present"] = df["Building_class_at_present"].cat.codes


df['Community_district'] = df['community_district_definition'].astype('category')
df["Community_district"] = df["Community_district"].cat.codes


df['School_district'] = df['school_district_definition'].astype('category')
df["School_district"] = df["School_district"].cat.codes


df['City_council_district'] = df['city_council_district_definition'].astype('category')
df["City_council_district"] = df["City_council_district"].cat.codes


df['Fire_company'] = df['fire_company_definition'].astype('category')
df["Fire_company"] = df["Fire_company"].cat.codes


df['Police_precinct'] = df['police_precinct_definition'].astype('category')
df["Police_precinct"] = df["Police_precinct"].cat.codes


df['Health_center_district'] = df['health_center_district'].astype('category')
df["Health_center_district"] = df["Health_center_district"].cat.codes


df['Owner_type'] = df['owner_type_definition'].astype('category')
df["Owner_type"] = df["Owner_type"].cat.codes

df['Above_Below_median'] = df['Above_Below_median'].astype('int')

df['Year_built'] = df['year_built'].astype('str')
df['Year_altered_one'] = df['year_altered_one'].astype('str')
df['Year_altered_two'] = df['year_altered_two'].astype('str')

df['sale_day'] = df['sale_day'].astype('str')
df['sale_month'] = df['sale_month'].astype('str')
df['sale_year'] = df['sale_year'].astype('str')

drop_columns = ['borough_code','building_class_category_code','borough_code_definition','tax_class_at_present_code',
                'neighborhood','building_class_category_code_definition','tax_class_at_present_code_definition',
                'year_built','community_district_definition','school_district_definition','city_council_district_definition',
                'fire_company_definition','police_precinct_definition','health_center_district','owner_type_definition',
                'year_altered_one','year_altered_two','building_class_at_present_code_definition','building_class_at_present_code',
                 'community_district','school_district','city_council_district',
                'x_coordinate','y_coordinate','maximum_allowable_facility_far','maximum_allowable_commercial_far',
                'maximum_allowable_residential_far','historical_district_name','owner_type','owner_type_definition',
                'fire_company','fire_company_definition','tax_class_at_time_of_sale_code','tax_class_at_time_of_sale_code_definition',
                'building_class_at_time_of_sale_code','building_class_at_time_of_sale_code_definition','address','apartment_number',
                'zipcode','sale_date','police_precinct']

new_df = df.drop(drop_columns, axis=1)

new_df.dtypes

new_df.head(10)

#Creating a new dataset with only the columns that will be needed for Part 2 of the assignment
NewYork_RentalSales = pd.DataFrame(new_df[['Year_built',
                                 'Borough',
                                 'Neighborhood',
                                 'Building_class_category',
                                 'Building_class_at_present',
                                 'Tax_class_at_present',
                                 'Community_district',
                                 'School_district',
                                 'City_council_district',
                                 'Fire_company',
                                 'Health_center_district',
                                 'Owner_type',
                                 'Year_altered_one',
                                 'Year_altered_two',
                                 'residential_units',               
                                 'commercial_units',                
                                 'total_units',                     
                                 'land_square_feet',           
                                 'gross_square_feet',               
                                 'sale_price',                    
                                 'Police_precinct',                 
                                 'floor_area_total_building',       
                                 'floor_area_commercial',           
                                 'floor_area_residential',          
                                 'floor_area_office',               
                                 'floor_area_retail',               
                                 'floor_area_garage',               
                                 'floor_area_storage',              
                                 'floor_area_factory',              
                                 'floor_area_other',                
                                 'buildings_number_of',             
                                 'floors_number_of',                
                                 'units_residential_number_of',       
                                 'units_total_number_of',             
                                 'assessed_value_land',             
                                 'assessed_value_total',            
                                 'exempt_value_land',               
                                 'exempt_value_total',              
                                 'building_age',                    
                                 'sale_year',                        
                                 'sale_month',                       
                                 'sale_day',
                                 'Above_Below_median']])
NewYork_RentalSales.head(10)

cols=[i for i in NewYork_RentalSales.columns if i not in ["Above_Below_median"]]
for col in cols:
    NewYork_RentalSales[col]=pd.to_numeric(NewYork_RentalSales[col])

cdf = NewYork_RentalSales.copy()

#Centering all columns except the predictor column (the last column)
for feature_name in NewYork_RentalSales.columns[0:-1]:
    mean_ = NewYork_RentalSales[feature_name].mean()
    sd_   = NewYork_RentalSales[feature_name].std()
    cdf[feature_name] = (NewYork_RentalSales[feature_name] - mean_) / (sd_)
cdf.head(10)

cdf.to_csv('NewYork_RentalSales_MAIN.csv',index=False)


cdf.transpose()

cdf.Above_Below_median.value_counts()

cdf.columns


########################## END ###########################
