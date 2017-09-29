import pandas as pd
import numpy as np
import time

start = time.time()

# shape = (580179, 79)
df = pd.read_pickle('data/opa.pkl')

''' -- List of columns in original Philadelphia Database -- '''

'the_geom', 'the_geom_webmercator', 'assessment_date', 'basements',
'beginning_point', 'book_and_page', 'building_code',
'building_code_description', 'category_code',
'category_code_description', 'census_tract', 'central_air',
'cross_reference', 'date_exterior_condition', 'depth',
'exempt_building', 'exempt_land', 'exterior_condition', 'fireplaces',
'frontage', 'fuel', 'garage_spaces', 'garage_type',
'general_construction', 'geographic_ward', 'homestead_exemption',
'house_extension', 'house_number', 'interior_condition', 'location',
'mailing_address_1', 'mailing_address_2', 'mailing_care_of',
'mailing_city_state', 'mailing_street', 'mailing_zip', 'market_value',
'market_value_date', 'number_of_bathrooms', 'number_of_bedrooms',
'number_of_rooms', 'number_stories', 'off_street_open',
'other_building', 'owner_1', 'owner_2', 'parcel_number', 'parcel_shape',
'quality_grade', 'recording_date', 'registry_number', 'sale_date',
'sale_price', 'separate_utilities', 'sewer', 'site_type', 'state_code',
'street_code', 'street_designation', 'street_direction', 'street_name',
'suffix', 'taxable_building', 'taxable_land', 'topography',
'total_area', 'total_livable_area', 'type_heater', 'unfinished', 'unit',
'utility', 'view_type', 'year_built', 'year_built_estimate', 'zip_code',
'zoning', 'objectid', 'lat', 'lng'

''' -------------- E -- N -- D -------------- '''

# Create lot size from columns 'frontage' and 'depth'
df['lot_size'] = round(df['depth'] * df['frontage'],2)

# Narrow columns to relevant columns
df = df[['location','number_of_bathrooms','number_of_bedrooms','number_of_rooms',
'number_stories','sale_price','sale_date','total_area','total_livable_area','year_built','zip_code',
'garage_spaces','depth','frontage','exterior_condition','lat','lng','zoning']]

# Mean exterior_condition = 3.85, median = 4
# Fill in Nans with median condition of 4
df['exterior_condition'].fillna(4,inplace=True)

# Fill lat and lng with 99.999999
df['lat'].fillna(99.999999,inplace=True)
df['lng'].fillna(99.999999,inplace=True)

# 34 rows missing sale date
# 3 - 4 rows in other columns
# df.shape = (579695, 18)
df.dropna(axis=0,how='any',inplace=True)

#get only 5 digit zip and convert to int
df.zip_code = df.zip_code.apply(lambda row: int(row[0:5]))

#convert '196Y',201S value to 1960 and 2010
def convert_to_int(value):
    try:
        int(value)
    except ValueError:
        return int(value[0:3] + '0')
    else:
        return int(value)

df.year_built = df.year_built.map(lambda row: convert_to_int(row))


# create dataframe that only has residential buildings

# zones = [zone for zone in df.zoning.unique().tolist() if zone[0] == "R"]

# df = df[df['zoning'].isin(zones)]
# df.reset_index(inplace=True,drop=True)

# create full address
def create_address(x):
    address = '{} philadelphia pa {}'.format(x[0],x[10]).lower()
    return ' '.join([word for word in address.split(' ') if word != ''])

df['address'] = df.apply(create_address, axis=1)

# df.shape = (514902, 19)

# create date-time sale date
df['sale_date'] = pd.to_datetime(df['sale_date'])

#create year column
df['sale_year'] = df['sale_date'].apply(lambda x: x.year)
df['sale_month'] = df['sale_date'].apply(lambda x: x.month)


print(time.time() - start)

def get_neighborhood_streets(lat_min,lat_max,lng_min,lng_max):
    # newbold neighborhood boundaries
    neighborhood = df[(df['lat'] > lat_min) & (df['lat'] < lat_max) & (df['lng'] > lng_min) & (df['lng'] < lng_max)]

    # create new column called block
    neighborhood['block'] = neighborhood.address.apply(lambda x: ' '.join(x.split(' ')[1:-3]))
    # create new block nums
    def create_house_nums(row):
        num = row[-3].split(' ')[0]
        go = 1
        while go == 1:
            try:
                int(num)
            except ValueError:
                pass
            else:
                return int(num)
            num = num[:-1]

    neighborhood['house_num'] = neighborhood.apply(create_house_nums,axis=1)

    # else int(x.split(' ')[0].split('-')[0][0:1]))
    # create blocks as hundreds
    neighborhood['block_num'] = neighborhood.house_num.apply(lambda x: x - (x%100) if x > 99 else x - (x%10))

    # find max for block
    maxy = neighborhood.groupby('block',as_index=False)['block_num'].max()
    maxy = maxy.rename(columns={'block_num':'max_num'})

    # add 100 to max block, e.g want 1905 to round to 2000 instead of 1900
    maxy['max_num'] = maxy.max_num.apply(lambda x: x + 100)

    # find min for block
    miny = neighborhood.groupby('block',as_index=False)['block_num'].min()
    miny = miny.rename(columns={'block_num':'min_num'})

    # merge together
    max_min = maxy.merge(miny)

    max_min_dict = {}
    def to_dict(x):
        max_min_dict[x[0]] = (x[2],x[1])

    max_min.apply(to_dict,axis=1)
    print(time.time() - start)
    print(max_min_dict)

def get_vars_for_regression():
    pass


if __name__ == '__main__':
    get_neighborhood_streets(lat_min=39.947805,
                            lat_max=39.957015,
                            lng_min=-75.225538,
                            lng_max=-75.199470)
