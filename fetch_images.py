'''
This class fetches images from Google Street View(GSV)
And saves images in a folder at a specified location
Examples formatted as follows:
      variable -> <'what is inside carrots is what should be entered'>
      e.g. street_name -> <'W Master St'>
Much thanks to google's excellet documentation
----- https://developers.google.com/maps/documentation/streetview/intro -----
'''

import requests # used to execute API calls
import os
import time
import pandas as pd
import cv2
import numpy as np
import math

class FetchImages(object):

    def __init__(self,which_API=None,save_to=None):
        '''
        -- Initializes class --

            PARAMETERS
            ----------
                which_API: str
                    Specify <'GSV'> for google street view. Goal is to build out this class later with more API calls
                save_to : str
                    file path to save fetched data to
            RETURNS
            ------
                None
        '''

        # set API and link
        self.API, self.link = self._links(which_API)
        # set file path for saving data
        # if path does not exist, _create_save_to_directory will create directory
        self.save_to = save_to
        # checks for API and save_to folder
        # returns message and exits if either are empty
        self._check_initialization()
        # set array for checking pics
        self.last_pic = np.array([[0],[0]])
        # start repeat counter
        self.repeats = 0

    def fit_info(self,
                    street_name,
                    zip_code,
                    blk_st,
                    blk_end,
                    city='Philadelphia',
                    state='PA'):
        '''
        -- Set parameters for search --

            PARAMETERS
            -----------
                zip_code: int -> <19121>
                    Specify 5 digit zip code
                street_name: str -> <'W Master St' or 'N 26th St'>
                    Specify street name
                blk_st: int -> <1200>
                    Specify lower bound to begin search
                blk_end: int -> <1400>
                    Specify upper bound to begin search
                city: str -> <'Philadelphia'>
                    Specify city
                state: str -> <'PA'>
                    Specify state abbreviation

            INITIALIZED
            -----------
                self.zip_code:
                    Sets zip code attribute
                self.street_name, self.city, self.state:
                    Lower cases each respecitvely and sets attribute
                self.house_nums:
                    Initializes house numbers as range from blk_st to blk_end
                self._create_addresses():
                    Calls function

            RETURNS
            -------
                None
        '''

        self.zip_code = zip_code
        self.street_name = street_name.lower()
        self.house_nums = (blk_st,blk_end)
        self.city = city.lower()
        self.state = state.lower()
        self.blocks = self._set_blocks()
        self.street_city_state_zip = '{} {} {} {}'.format(self.street_name,
                                                        self.city,
                                                        self.state,
                                                        self.zip_code)


    def set_payload(self,
                pic_size='640x640',
                address=None,
                heading=0,
                fov=80,
                API_key=None):
        '''
        -- Set payload parameters for API --

            NOTE
            ----
                All parameters required by GSV API to fetech picture
                See https://developers.google.com/maps/documentation/streetview/intro for full documentation

            PARAMETERS
            ----------
                pic_size: str -> '640x640'
                    Set size of picture, max 640x640 for free API 1000x1000 for paid
                address: str -> '2501 w master street philadelphia pa 19121'
                    Address of picture to fetch
                heading: int -> 0
                    Heading of camera, e.g. 0 = North, 180 = South
                fov: int -> 80
                    Field of view of camera, less = narrower pic
                API_key:
                    API key from .bash_profile

            INITIALIZED
            -----------
                self.payload:
                    Payload for requests library per GSV specifications

            RETURNS
            -------
                None
        '''

        # set parameters per GSV requirements
        self.payload = {'size': pic_size,
          'location': '{}'.format(address),
          'heading': '{}'.format(heading),
          'fov': fov,
          'key': API_key}

    def fetch_pictures(self,even_heading,print_num_fetches=True):
        '''
        -- Fetch pictures --

            PARAMETERS
            ----------
                even_heading: int
                    Set heading for the camera if the street number is odd
                odd_heading: int
                    Set heading for the camera if the street number is even
                print_num_fetches: bool
                    default=True


            INITIALIZED
            -----------
                self.directory = directory to save to

            RETURNS
            -------
                Saves pictures to specified directory
        '''

        # set self.directory, function creates if it does not exist
        self._create_save_to_directory()
        # start counter for _fetches_for_day
        counter = 0
        # begin time for _info_for_day
        overall_start = time.time()
        # pull even or odd numbered streets, starts with even
        for block in self.blocks.keys():
            start = '{} {}'.format(block, self.street_city_state_zip)
            end = '{} {}'.format(block+100, self.street_city_state_zip)
            self.heading = self._get_heading(blk_st=start,blk_end=end)

            for even_or_odd_block in self.blocks[block].keys():
                self.bearing = self._get_bearing(heading=self.heading,
                                            side_of_st=even_heading,
                                            blk=even_or_odd_block)

                for house_num in self.blocks[block][even_or_odd_block]:
                    address = '{} {}'.format(house_num, self.street_city_state_zip)
                    start = time.time()
                    print('Fetching {}'.format(address))
                    # set params for this picture fetch
                    self.set_payload(address=address,heading=self.bearing,API_key=self.API)
                    # get picture object from google
                    self._fetch_picture()
                    # save picture object
                    self._save_pic(address)
                    counter += 1
                    # display time to fetch picture
                    print('Time to fetch: {}\n'.format(time.time() - start))
        # print helpful info
        self._info_for_day(num_fetches=counter,overall_start_time=overall_start)

    def _get_bearing(self,heading,side_of_st,blk):
        if side_of_st == 'right':
            even_pic = (heading + 90) % 360
            odd_pic = (heading + 270) % 360
        elif side_of_st == 'left':
            even_pic = (heading + 270) % 360
            odd_pic = (heading + 90) % 360
        else:
            return 'You need to specify "right" or "left" for the variable "even_heading"'
        if blk == 'even':
            return even_pic
        elif blk == 'odd':
            return odd_pic
        else:
            return 'I need a bearing'

    def _get_heading(self,blk_st,blk_end):
        block_start = self._get_meta_data(blk_st)
        block_end = self._get_meta_data(blk_end)
        lat_start = math.radians(block_start.json()['location']['lat'])
        lng_start = math.radians(block_start.json()['location']['lng'])
        lat_end = math.radians(block_end.json()['location']['lat'])
        lng_end = math.radians(block_end.json()['location']['lng'])

        dLong = lng_end - lng_start

        dPhi = math.log(math.tan(lat_end/2.0+math.pi/4.0)/math.tan(lat_start/2.0+math.pi/4.0))
        if abs(dLong) > math.pi:
            if dLong > 0.0:
                 dLong = -(2.0 * math.pi - dLong)
            else:
                dLong = (2.0 * math.pi + dLong)

        bearing = (math.degrees(math.atan2(dLong, dPhi)) + 360.0) % 360.0;

        return bearing

    def _get_meta_data(self,address):
        link = 'https://maps.googleapis.com/maps/api/streetview/metadata?parameters'
        payload = {'location':address,
                    'key':self.API}
        return requests.get(link,params=payload)


    def _info_for_day(self,num_fetches,overall_start_time):
        '''
        -- Print helpful info from latest fetch sequence --

            PARAMETERS
            ----------
                num_fetches: int
                    number of fetches from latest fetch_pictures function
                overall_start_time: time
                    time when fetch_pictures function started
            PRINTS
            ------
                How many pictures where fetched in most recent fetch
                How much time it took to fetch pictures in most recent fetch
                How many total fetches there have been for the day (GSV allows 25,000 free pics/day)
        '''

        # count total fetches for the day, 25,000 free/day for GSV
        self._fetches_for_day(num_fetches)
        # calculate total time to fetch all pics
        overall_finish = time.time() - overall_start_time
        # print info related to fetches for the day
        print('\nI fetched a total of {} pictures in {} seconds'.format(num_fetches, round(overall_finish,2)))
        print('{} of {} photos were repeats'.format(self.repeats,num_fetches))
        print('Your TOTAL fetches TODAY are: {}\n'.format(self.total_fetches_for_the_day))

    def _links(self,which_API):
        '''
        -- Sets API key and link for API --
            Called in __init__ function

            TODO
            ----
                Expand class to include other API calls

            PARAMETERS
            ----------
                which_API: str -> 'GSV'
                    passed from initalization

            RETURNS
            -------
                API key from .bash_profile
                Link for API
        '''

        if which_API == 'GSV':
            # gets API key from bash_profile and str of API link
            return os.environ.get('GOOGLE_STREET_VIEW_KEY'), 'https://maps.googleapis.com/maps/api/streetview?'
        else:
            # return message and break loop if API not found
            return "I don't have that API"

    def _set_blocks(self):
        blocks = {}

        if self.house_nums[0] > 99:
            remainder = self.house_nums[0] % 100
            block = self.house_nums[0] - remainder

        elif self.house_nums[0] < 100:
            block = 0

        while block < self.house_nums[-1]:
            # blocks[block] = list(range(block, block + 100))
            blocks[block] = {'even': [x for x in list(range(block, block + 100)) if x % 2 == 0]}
            blocks[block].update({'odd': [x for x in list(range(block, block + 100)) if x % 2 != 0]})
            block += 100

        return blocks

    def _fetch_picture(self):
        '''
        Fetch pictures from GSV

            INITIALIZED
            -----------
                self.response: object
                    initializes object fetched from google
            RETURNS
            -------
                None
        '''

        self.response = requests.get(self.link,params=self.payload)

    def _create_save_to_directory(self):
        '''
        -- Create directory to save fetched pictures to --

            PARAMETERS
            ----------
                None

            INITIALIZED
            -----------
                self.directory: str -> 'users/ben/Desktop/n_25th_st'
                    Initialize directory to save to

            RETURNS
            -------
                None
        '''

        # create directory str
        self.directory = '{}'.format(self.save_to)
        # self.directory = '{}/{}'.format(self.save_to,self.street_name.lower().replace(' ','_'))
        # if directory to save to does not exist, then create it
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def _save_pic(self,address):
        '''
        -- Save fetched picture --

            PARAMETERS
            ----------
                address: str
                    string of address of picture object
            RETURNS
            -------
                Saves picture to specified file path
        '''

        street = address.replace(' ','_').lower()

        if self.response.status_code == 200:
            self._save_file(name='temp')
            if self._check_for_repeats() == False:
                # since not a repeat picture, save to file
                self._save_file(name=street)
                # set new picture to be last picture to compare in next loop
                self.last_pic = self.new_pic
            else:
                # update number of repeats for _info_for_day
                self.repeats += 1
                print('picture is repeat')
                pass
        else:
            print('No picture fetched for this address')
            pass

    def _check_for_repeats(self):
        '''
        -- Checks if picture is a repeat of last picture --
            Called in _save_pic function

            PARAMETERS
            ----------
                None

            RETURNS
            -------
                Boolean value
        '''

        # create numpy array of new picture
        self.new_pic = cv2.imread('{}/{}'.format(self.directory,'temp.jpg'))
        # compare new picture to last picture
        return np.array_equal(self.new_pic,self.last_pic)

    def _save_file(self,name):
        '''
        -- Saves file to hard drive ---
            Called in _save_pic function

            PARAMETERS
            ----------
                name: str -> 'temp'
                    name of file

            RETURNS
            -------
                writes fetched picture to file
                format self.directory/name
        '''

        with open("{}/{}.jpg".format(self.directory,name), 'wb') as f:
            f.write(self.response.content)


    def _fetches_for_day(self,
                        num_fetches,
                        fetch_pickle_from='data/fetches_by_date.pkl',
                        save_pickle_to='data/fetches_by_date.pkl'):
        '''
        -- Return total number of fetches for the day --

        PARAMETERS
        ----------
            num_fetches: int
                number of fetches passed from counter in fetch pictures
            fetch_pickle_from: str
                file path history of fetches exists
            save_pickle_to: str
                file path to save updated pickle to

        INITIALIZED
        -----------
            self.df: pandas dataframe
                initialize pandas dataframe which can be viewed to see history of fetches
                dataframe used to total fetches for the day
                25,000 free fetches per day for GSV
            self.total_fetches_for_the_day: int
                set fetches for the day as attribute

        RETURNS
        -------
            None
            self.df: saved as pickled dataframe
        '''

        # get today's date
        date = time.strftime("%Y-%m-%d")
        # get today's time
        time_of_day = time.strftime("%H:%M:%S")
        # retrieve pickle with history of fetches for GSV API
        self.df = pd.read_pickle(fetch_pickle_from)
        self.df = self.df.append({'date':date,'time':time_of_day,'fetches':num_fetches},ignore_index=True)
        self.df.to_pickle(save_pickle_to)
        # total up the fetches for today
        self.total_fetches_for_the_day = self.df[self.df['date']==date]['fetches'].sum()

    def _check_initialization(self):
        '''
        Called in __init__ function to check for initalizers
        '''

        if self.save_to == None:
            return "Please specify a location to save to"
        if self.API == None:
            return "Please specify an API to use"
        else:
            pass
