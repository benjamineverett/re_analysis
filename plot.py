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
from folium import IFrame
import base64

class Plot(object):

    def __init__(self,
                    starting_loc_map,
                    zoom,
                    html_save_to
                    ):
        self.mappy = self._initialize_map(starting_loc_map,zoom)
        self.df_metadata, self.df_labeled, self.df_predicted = self._get_data()
        self.save_to = '{}.html'.format(html_save_to)

    def plot_predicted(self):
        green = '#228b22'
        gray = '#D3D3D3'
        # true/false, column, fill/not fill, color, radius, opacity
        labels = [(1,'predicted',True,green,5,0.5),(0,'predicted',False,gray,2,0.3)]
        for label in labels:
            column = label[1]
            labeled_points, points_count = self._get_labeled(label=column,prediction=label[0],dataframe=self.df_predicted)
            counter = 1
            for address in labeled_points:
                lat_lng_address = '{}.jpg'.format(' '.join(address.split('.')[0].split('_')[:-1]).replace(' ','_'))
                readable_address = ' '.join(address.split('_')[:-1])
                print('Plotting {}\nThis is point {}/{}'.format(readable_address,counter,points_count))
                latitude, longitude = self._get_lat_and_lng(address=lat_lng_address)
                if latitude != 0:
                    ''' Credit to for excellent tutorial
                    http://nbviewer.jupyter.org/gist/ocefpaf/0ec5c93138744e5072847822818b4362
                    '''
                    # encoded = base64.b64encode(open('pics/resized/{}.jpg'.format(address), 'rb').read()).decode()
                    # html = '<img src="data:image/jpeg;base64,{}">'.format
                    # iframe = IFrame(html(encoded), width=300, height=600)
                    # popup = folium.Popup(iframe, max_width=1000)
                    folium.features.Circle(location=[latitude,longitude],
                                    radius=label[4],
                                    # popup=popup,
                                    # popup='<i>{}</i>'.format(readable_address),
                                    color=label[3],
                                    fill=label[2],
                                    fill_color=label[3],
                                    fill_opacity = label[5]
                                    # weight=2.5,
                                    # opacity=0.3
                                    ).add_to(self.mappy)
                    if counter % 25000 == 0 or counter == 1:
                        print('--------------- SAVING MAP ---------------')
                        self.mappy.save(outfile=self.save_to)
                counter += 1
        print('--------------- FINISHED -- SAVING MAP ---------------')
        self.mappy.save(outfile=self.save_to)


    def plot_point(self):
        '''
        -- Plot point onto map --

            PARAMETERS
            ----------
                address: str -> '1234 n 26th st philadelphia pa 19121'
                    address of location for lat and lng
        '''
        green = '#228b22'
        gray = '#0000FF' # blue
        labels = [('tp',green,True,green,1),
                    ('fp',green,True,gray,1),
                    ('fn',gray,True,green,1),
                    ('tn',gray,True,gray,1)]
        for label in labels:
            labeled_points, points_count = self._get_labeled(dataframe=self.df_labeled,prediction=label[4],label=label[0])
            counter = 1
            for address in labeled_points:
                lat_lng_address = '{}.jpg'.format(' '.join(address.split('.')[0].split('_')[:-1]).replace(' ','_'))
                readable_address = ' '.join(address.split('_')[:-1])
                print('Plotting {}\nThis is point {}/{} ~ Group {}'.format(readable_address,counter,points_count,label[0].upper()))
                latitude, longitude = self._get_lat_and_lng(address=lat_lng_address)
                if latitude != 0:
                    ''' Credit to for excellent tutorial
                    http://nbviewer.jupyter.org/gist/ocefpaf/0ec5c93138744e5072847822818b4362
                    '''
                    if label [0] != 'tn':
                        encoded = base64.b64encode(open('pics/labeled_resized/{}.jpg'.format(address), 'rb').read()).decode()
                        html = '<img src="data:image/jpeg;base64,{}">'.format
                        iframe = IFrame(html(encoded), width=200, height=400)
                        popup = folium.Popup(iframe, max_width=1000)
                        folium.features.Circle(location=[latitude,longitude],
                                        radius=5,
                                        popup=popup,
                                        # popup='<i>{}</i>'.format(readable_address),
                                        color=label[1],
                                        fill=label[2],
                                        fill_color=label[3],
                                        fill_opacity = 0.5
                                        ).add_to(self.mappy)
                    else:
                        folium.features.Circle(location=[latitude,longitude],
                                        radius=2,
                                        # popup=popup,
                                        # popup='<i>{}</i>'.format(readable_address),
                                        color=label[1],
                                        fill=label[2],
                                        fill_color=label[3],
                                        fill_opacity = 0.2
                                        ).add_to(self.mappy)
                    if counter % 10000 == 0 or counter == 1:
                        print('--------------- SAVING MAP ---------------')
                        self.mappy.save(outfile=self.save_to)
                counter += 1
        print('--------------- FINISHED -- SAVING MAP ---------------')
        self.mappy.save(outfile=self.save_to)

    def _get_data(self):

        return pd.read_pickle('data/meta_data.pkl'), pd.read_pickle('data/all_labels.pkl'), pd.read_pickle('data/all_predicted.pkl')

    def _initialize_map(self,starting_loc_map,zoom):
        # create empty map for folium
        return folium.Map(location=starting_loc_map, zoom_start=zoom)

    def _get_labeled(self,label,dataframe,prediction):
        labeled = dataframe[dataframe[label] == prediction]['filename']
        # labeled = self.df_labeled['filename'].apply(lambda x: '{}.jpg'.format(' '.join(x.split('.')[0].split('_')[:-1]).replace(' ','_')))
        # labeled.drop_duplicates(inplace=True)
        lst = labeled.tolist()
        return lst, len(lst)

    def _get_lat_and_lng(self,address):
        lat = self.df_metadata[self.df_metadata.filename == '{}'.format(address)]['lat_lng'].tolist()[0][0]
        lng = self.df_metadata[self.df_metadata.filename == '{}'.format(address)]['lat_lng'].tolist()[0][1]
        return lat,lng


class GSVMetaData(object):

    def __init__(self,data_address,df_filepath,df_backup_filepath,df_backup_name):
        self.link = self._set_link()
        self.API = self._get_api()
        self.df = self._load_data(data_address)
        self.df_filepath = df_filepath
        self.df_backup_filepath = df_backup_filepath
        self.df_backup_name = df_backup_name

    def get_meta_data(self,address):
        payload = self._set_payload(address)
        return self._get_meta_data(payload)

    def run_main(self):
        self.address_lst = self._get_addresses()
        address_count = len(self.address_lst)
        for_df = []
        counter = 1
        for address in self.address_lst:
            print('fetching {} -- {} of {}'.format(address,counter,address_count))
            meta_data = self.get_meta_data(address.split('.')[0][:-6].replace('_',' '))
            lat, lng, date, pano_id = self._check_meta_data(meta_data)
            for_df.append([address,(lat,lng),date,pano_id])
            counter += 1
            if counter % 100 == 0:
                self._save_df(for_df)
                for_df = []
        self._save_df(for_df)

    def _check_meta_data(self,meta_data):
        if meta_data.status_code == 200 and meta_data.json()['status'] == 'OK':
            try:
                lat = meta_data.json()['location']['lat']
                lng = meta_data.json()['location']['lng']
                date = meta_data.json()['date']
                pano_id = meta_data.json()['pano_id']
            except KeyError:
                print('-------------- RESOLVING KEY ERRORS --------------')
                try:
                    meta_data.json()['location']['lat']
                except KeyError:
                    lat = 0
                else:
                    lat = meta_data.json()['location']['lat']
                try:
                    meta_data.json()['location']['lng']
                except KeyError:
                    lng = 0
                else:
                    lng = meta_data.json()['location']['lng']
                try:
                    meta_data.json()['date']
                except KeyError:
                    date = 0
                else:
                    date = meta_data.json()['date']
                try:
                    pano_id = meta_data.json()['pano_id']
                except KeyError:
                    pano_id = 0
                else:
                    pano_id = meta_data.json()['pano_id']
                return lat,lng,date,pano_id
            else:
                lat = meta_data.json()['location']['lat']
                lng = meta_data.json()['location']['lng']
                date = meta_data.json()['date']
                pano_id = meta_data.json()['pano_id']
                return lat, lng, date, pano_id
        else:
            return 0,0,0,0

    def _get_meta_data(self,payload):
        return requests.get(self.link,params=payload)

    def _set_payload(self,address):
        return {'location':address,
                    'key':self.API}

    def _get_api(self):
        return os.environ.get('GOOGLE_STREET_VIEW_KEY')

    def _set_link(self):
        return 'https://maps.googleapis.com/maps/api/streetview/metadata?parameters'

    def _load_data(self,data_address):
        return pd.read_pickle(data_address)

    def _get_addresses(self):
        return self.df['filename'].tolist()

    def _save_df(self,data):
        print('--------------- SAVING DF ----------------')
        if os.path.isfile(self.df_filepath):
            # read and save old df
            old_df = pd.read_pickle(self.df_filepath)
            old_df.to_pickle('{}/{}_old_{}.pkl'.format(self.df_backup_filepath,
                                                        self.df_backup_name,
                                                        time.ctime().lower().replace(' ','_')))
            # create and back up new df
            df = pd.DataFrame(data,columns=['filename','lat_lng','pic_date','pano_id'])
            df.to_pickle('{}/{}_new_{}.pkl'.format(self.df_backup_filepath,
                                                        self.df_backup_name,
                                                        time.ctime().lower().replace(' ','_')))
            # combine two dfs
            new_df = old_df.append(df)
            new_df.reset_index(inplace=True,drop=True)
            # overwrite old instance of resized.pkl
            new_df.to_pickle(self.df_filepath)
        else:
            # create new df and save as pickle
            self.df = pd.DataFrame(data,columns=['filename','lat_lng','pic_date','pano_id'])
            self.df.to_pickle(self.df_filepath)



if __name__ == '__main__':
    btown = Plot(starting_loc_map=(39.967254, -75.172137),
                    zoom=16,
                    html_save_to = 'labeled_small')
    btown.plot_point()
    # btown = Plot(starting_loc_map=(39.967254, -75.172137),
    #                 zoom=16,
    #                 html_save_to = 'predicted_small_2')
    # btown.plot_predicted()
    # meta = GSVMetaData(data_address='data/all_labels.pkl',
                        # df_filepath='data/meta_data.pkl',
                        # df_backup_filepath='data/backups',
                        # df_backup_name='backup_metadata')
    # meta.run_main()
