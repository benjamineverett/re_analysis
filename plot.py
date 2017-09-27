'''
-------------- Plot --------------
This class will plot points onto a map

Much thanks to google's metadata API
----- https://developers.google.com/maps/documentation/streetview/metadata -----

Much thanks to Domino's tutorial
----- https://blog.dominodatalab.com/creating-interactive-crime-maps-with-folium/ -----
'''

import pandas as pd
import numpy as np
import folium
import os
from fetch_images import FetchImages
import requests
import time

class Plot(object):

    def __init__(self):
        # initialize FetchImage class to use various features
        self.FI = FetchImages(which_API='GSV',save_to=None)



    def plot_point(self,file_path_of_labeled):
        '''
        -- Plot point onto map --

            PARAMETERS
            ----------
                address: str -> '1234 n 26th st philadelphia pa 19121'
                    address of location for lat and lng


        '''

        brewerytown = 39.976726, -75.178243
        # create empty map for folium
        mappy = folium.Map(location=brewerytown, zoom_start=20)
        addresses = self._get_trees_from_dataframe(filepath=file_path_of_labeled)
        counter = 1
        count_of_addresses = len(addresses)
        for address in addresses:
            print('Plotting {}\nThis is point {}/{}'.format(address,counter,count_of_addresses))
            latitude, longitude = self._get_lat_and_lng(address=address)
            if latitude != 0:
                folium.features.Circle(location=[latitude,longitude],
                                radius=5,
                                popup='<i>{}</i>'.format(address),
                                color='#86cc31',
                                fill=True,
                                fill_color='#86cc31',
                                fill_opacity = 0.5
                                ).add_to(mappy)
                if counter % 100 == 0 or counter == 1:
                    mappy.save(outfile='map.html')
            counter += 1
        mappy.save(outfile='map.html')

    def _get_lat_and_lng(self,address):
        '''
        -- Get latitude and longitude given address --

            PARAMETERS
            ----------
                address: str -> '1234 n 26th st philadelphia pa 19121'
                    address of location for lat and lng

            RETURNS
            -------
                lat: float
                    latitude in form 39.974986
                lng: float
                    longitude in form -75.178770
        '''
        meta_data = self.FI._get_meta_data(address=address)
        counter = 1
        tries = 1
        while counter == 1:
            if meta_data.status_code == 200 and meta_data.json()['status'] == 'OK':
                lat = meta_data.json()['location']['lat']
                lng = meta_data.json()['location']['lng']
                return lat,lng
            else:
                print('\n\n\nError. Retrying in {} seconds\n\n\n'.format(5*tries))
                time.sleep(tries * 5)
                tries += 1
                if tries > 5:
                    return 0,0

    def _get_trees_from_dataframe(self,filepath):
        df = pd.read_pickle(filepath)
            trees = df[df['label'] == 1]
            trees = trees['filename'].apply(lambda x: x.split('.')[0].replace('_',' '))
            trees.drop_duplicates(inplace=True)
        return trees.tolist()

if __name__ == '__main__':
    btown = Plot()
    # btown.get_trees_from_dataframe()
    btown.plot_point(file_path_of_labeled='data/labeled.pkl')
