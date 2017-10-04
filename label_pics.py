'''
This class file contains two classes:
    Labler: used to label pics
    Resizer: used to resize pics
'''
import os # for file operations
import cv2 # imaging library
import numpy as np # numpy!!
import pandas as pd
import pdb
import time
from block_creator import Randomizer

class Labeler(Randomizer):
    '''
    Labeler, a summary:

        1) Labeler inherits from Randomizer
            Randomizer chooses blocks at random from specified
                neighborhood and does bookkeeping

        2) Fetch files from neighborhood up to specified number of pics

        3) Label pics and save

        4) Create panda's data frame containing filename, label

        To label pics, on mac:
            Fn + left arrow key = 0
            Fn + right arrow key = 1
    '''

    def __init__(self,
                neighborhood,
                num_pics,
                zip_code):
        '''
        -- Initializes class --

        PARAMETERS
        ----------
            neighborhood: str
                name of file where pics are stored
            num_pics: int
                specify number of pics to pull from neighborhood
            zip_code: int
                specify zip code of neighborhood
        '''
        #Randomizer.__init__(neighborhood,num_pics)
        super(Labeler,self).__init__(neighborhood,num_pics)
        self._initializer(neighborhood)
        # set zip code
        self.zip_code = zip_code

    def label_pics(self):
        '''
        -- Main block to run code for labeling --

            PARAMETERS
            ----------

                NONE: uses all self initialized in self._initializer()

            RETURNS
            -------
                saves to pkl: filename, label
        '''
        pic_counter = 1
        list_of_labels = []
        for pict in self.pics_to_label:
            # get pictures
            pic1, pic2, flip1, flip2 = self._create_all_pics(p=pict[0])
            # loop through pics and enumerate for filenaming purposes
            for i,pic in enumerate([(pic1,flip1),(pic2,flip2)]):
                # display picture
                self._show_pic(picture=pic[0],counter=pic_counter,address=pict[0])
                # save picture and flip of picture
                cv2.imwrite('{}/{}_{}_pic{}.jpg'.format(self.labeled_pics_folder_path,pict[0].replace(' ','_'),self.zip_code,i+1),pic[0])
                cv2.imwrite('{}/{}_{}_flip{}.jpg'.format(self.labeled_pics_folder_path,pict[0].replace(' ','_'),self.zip_code,i+1),pic[1])
                # append label and label for flip pic
                list_of_labels.append(['{}_{}_pic{}.jpg'.format(pict[0].replace(' ','_'),self.zip_code,i+1),self.label,pict[1]])
                list_of_labels.append(['{}_{}_flip{}.jpg'.format(pict[0].replace(' ','_'),self.zip_code,i+1),self.label,pict[1]])
                # save every 100 labels
                if pic_counter % 100 == 0:
                    self._save_labels(list_of_labels)
                    list_of_labels = []
                pic_counter += 1
        # close all windows
        cv2.destroyAllWindows()
        # save labels to file
        self._save_labels(list_of_labels)
        self._save_labeled_blocks()

    ''' --------------- BEGIN HIDDEN METHODS --------------- '''

    def _initializer(self,neighborhood):
        ''' Run these functions to load class '''
        # Initializes self.pics_to_label
        self.get_random_pics()
        # get count of number of files in folder
        self.file_count = len(self.pics_to_label)
        # set folder path
        self.folder_path = 'pics/{}'.format(neighborhood)
        # set labeled pics folder path
        self.labeled_pics_folder_path = 'pics/labeled'.format(neighborhood)

    def _save_labeled_blocks(self):
        ''' Save sampled block to text so it is not used again '''

        with open('data/sampled_blocks.txt','a') as f:
            for block in self.labeled_blocks:
                 f.write('{},\n'.format(block))
            f.close()

    def _create_all_pics(self,p):
        '''
        -- Take 1 photo, 600x600, splits vertically and mirros both images --

            PARAMETERS
            ----------
                p: str
                    filepath to picture
            RETURNS
            -------
                pic1, pic2: original picture split in half vertically
                flip1, flip2: mirrored image of pic1 and pic2 respectivly
        '''
        pic1, pic2 = self._read_and_split_pic(picture=p)
        flip1, flip2 = self._mirror(picture1=pic1,picture2=pic2)
        return pic1, pic2, flip1, flip2

    def _mirror(self,picture1, picture2):
        # return mirror image of pics
        return np.flip(picture1,1), np.flip(picture2,1)

    def _read_and_split_pic(self,picture):
        # read pic, returns np array
        pic = cv2.imread('{}/{}_{}.jpg'.format(self.folder_path,picture.replace(' ','_'),self.zip_code))
        # split picture vertically
        pic1 = pic[0:600,0:300]
        pic2 = pic[0:600,301:600]
        return pic1, pic2

    def _show_pic(self,picture,counter,address):
        '''
        -- Displays picture and returns boolean --

            PARAMETERS
            ----------
                picture: np.array
                    cv2.imread np.array of picture

                counter: int
                    count of number of pics labeled

                address: str
                    address in readable formate for display purposes

            INSTANTIATED
            ------------
                self.label: bool value
                    used to append to list of lists for panda's data frame
        '''

        # set title on image displayer
        pic_title = '{} of {} ~ {}'.format(counter,self.file_count*2,address)
        # display the picture
        cv2.imshow(pic_title,picture)
        # move the picture to the relative center of my screen
        cv2.moveWindow(pic_title, 420, 50)
        # openCV2 specification to set a wait key
        k = cv2.waitKey()
        # if fn + right key struck, return 1 = TRUE == YES
        if k == 119:
            # close window
            cv2.destroyWindow(pic_title)
            # update label counter
            self.label = 1
        # if fn + left key struck, return 0 = FALSE == NO
        if k == 115:
            cv2.destroyWindow(pic_title)
            self.label = 0

    def _save_labels(self,labeled_pics):
        # check if data frame exists, if not, create new data frame
        if os.path.isfile('data/labeled.pkl'):
            old_df = pd.read_pickle('data/labeled.pkl')
            # save old df as backup
            old_df.to_pickle('data/backups/backup_labeled_old_{}.pkl'.format(time.ctime().lower().replace(' ','_')))
            self.df = pd.DataFrame(labeled_pics,columns=['filename', 'label','block'])
            self.df.to_pickle('data/backups/backup_labeled_new_{}.pkl'.format(time.ctime().lower().replace(' ','_')))
            # create updated data frame
            new_df = old_df.append(self.df)
            new_df.reset_index(inplace=True,drop=True)
            new_df.to_pickle('data/labeled.pkl')
        else:
            self.df = pd.DataFrame(labeled_pics,columns=['filename', 'label','block'])
            self.df.to_pickle('data/labeled.pkl')

    ''' --------------- END HIDDEN METHODS --------------- '''

class Resizer(object):
    '''
    -- Initialize class --

        PARAMETERS
        ----------
            filepath: str
                filepath to folder where images to resize reside

            num_pixels: int
                number of pixels to scale larger height to
                    note: perspective maintained
                e.g. original image = 600x300
                        num_pixels = 100 -> rescaled image 100x50

            resized_file_path: str
                filepath to folder to place resized images

            move: bool
                whether to move files to new folder after resizing

            df_filepath: str
                files saved into panadas data frame as filename, np.array
                specify filepath to save data frame to
                new data frame will be created if none exists in filepath

            df_backup_filepath: str
                specify filepath to save backup data frame to
                all data frames are saved as a backup before merging as fail safe

            df_backup_name: str
                specify name for backup data frame

            show_resized_pic: bool
                default = False
                    display resized photos at end of resizing
    '''

    def __init__(self,
                filepath,
                num_pixels,
                resized_file_path,
                move,
                df_filepath,
                df_backup_filepath,
                df_backup_name,
                show_resized_pic=False,):
        self.filepath = filepath
        self.resized_file_path = resized_file_path
        self.num_pixels = num_pixels
        self.show_resized_pic = show_resized_pic
        self.move = move
        self.df_filepath = df_filepath
        self.df_backup_filepath = df_backup_filepath
        self.df_backup_name = df_backup_name

    def resize_pics(self,to_np_array=True):
        '''
        -- Main block to run Resizer --

            PARAMETERS
            ----------
                to_np_array: bool
                    default = True
                    if False -> resize and save pics only
                    if True -> resize pics and save in data frame as numpy array
            RETURNS
            -------
                if to_np_array = True:
                    saves resized photos
                    saves pickled data frame containing filename,numpy array
                else:
                    saves reised photos ONLY
        '''
        # create array for data frame
        if to_np_array:
            array = []
        # get list of files
        files = [os.fsdecode(file) for file in os.listdir(self.filepath)]
        counter = 1
        for picture in files:
            # if os.fsdecode is used more than once in folder, it places .DS_Store file
            # it will be read as the first file in folder
            # this 'if' statement circumvents problem
            if picture == '.DS_Store':
                pass
            else:
                # give a little feedback, make me feel like the computer is working
                print('Resizing {}. Picture {} of {}'.format(picture,counter,len(files)))
                # resize picture
                pic1, pic2 = self._resize_pic(picture)
                # if True, show picture
                if self.show_resized_pic:
                    self._show_pic(picture)
                # write picture to file
                for i, half_pic in enumerate([pic1,pic2]):
                    cv2.imwrite('{}/{}_pic{}.jpg'.format(self.resized_file_path,picture.split('.')[0],i+1),half_pic)
                    counter += 1
                    # append to list as [filename, np.array]
                    if to_np_array:
                        array.append(['{}_pic{}'.format(picture.split('.')[0],i+1), np.array(half_pic)])
                        #np.array(cv2.imread('{}/{}_pic{}'.format(self.resized_file_path,picture,i)))])
                        if counter % 1000 == 0:
                            if to_np_array:
                                # give me some feed back that everything is working
                                print('\n-------------------- SAVING DATA ---------------------\n')
                                self._save_df(data=array)
                                # reset to empty array after saving
                                array=[]
        # append list to data frame
        if to_np_array:
            self._save_df(data=array)

        if self.move:
            for picture in files:
                print('Moving {}'.format(picture))
                os.rename('{}/{}'.format(self.filepath,picture), 'pics/labeled_resized/{}'.format(picture))

    ''' --------------- BEGIN HIDDEN METHODS --------------- '''

    def _resize_pic(self,pic):
        '''
        -- Resize picture and return np.array --

            PARAMETERS
            ----------
                pic: str
                    filepath of picture
            RETURNS
            -------
                np array of resized image
        '''
        # read image from folder/filename
        image = cv2.imread('{}/{}'.format(self.filepath,pic))
        #split image
        pic1 = image[0:600,0:300]
        pic2 = image[0:600,301:600]
        image_size = pic1
        # resize according to height
        r = self.num_pixels / image_size.shape[1]
        # set dimensions of new image
        dim = (self.num_pixels, int(image_size.shape[0] * r))
        # resize image and return as numpy array
        pic1 = cv2.resize(pic1, dim, interpolation = cv2.INTER_AREA)
        pic2 = cv2.resize(pic2, dim, interpolation = cv2.INTER_AREA)
        return pic1, pic2

    def _show_pic(self,pic):
        '''
        -- Method to show picture on the screen --- '

            PARAMETERS
            ----------
                pic: numpy array

            RETURNS
            -------
                show resized picture on screen
                called at end of resize pics method
        '''

        cv2.imshow('resized',resized)
        cv2.waitKey()

    def _save_df(self,data):
        '''
        -- Saves info to data frame

            PARAMETERS
            ----------
                IF to_np_array = True

                data: list of lists -> [['filename1.jpg',np.array],['filename2.jpg',np.array]]

            RETURNS
            -------
                Saves data as pickled panda's data frame
        '''
        # check if data frame exists, if not, create new data frame
        if os.path.isfile(self.df_filepath):
            # read and save old df
            old_df = pd.read_pickle(self.df_filepath)
            old_df.to_pickle('{}/{}_old_{}.pkl'.format(self.df_backup_filepath,
                                                        self.df_backup_name,
                                                        time.ctime().lower().replace(' ','_')))
            # create and back up new df
            self.df = pd.DataFrame(data,columns=['filename', 'np_array'])
            self.df.to_pickle('{}/{}_new_{}.pkl'.format(self.df_backup_filepath,
                                                        self.df_backup_name,
                                                        time.ctime().lower().replace(' ','_')))
            # combine two dfs
            new_df = old_df.append(self.df)
            new_df.reset_index(inplace=True,drop=True)
            # overwrite old instance of resized.pkl
            new_df.to_pickle(self.df_filepath)
        else:
            # create new df and save as pickle
            self.df = pd.DataFrame(data,columns=['filename', 'np_array'])
            self.df.to_pickle(self.df_filepath)



    ''' --------------- END HIDDEN METHODS --------------- '''

if __name__ == '__main__':
    # fairmount = Labeler('fairmount',800,19130)
    # fairmount.label_pics()
    # resize = Resizer(num_pixels=50)
    # resize.resize_pics()
    # brewerytown = Labeler('brewerytown',900,19121)
    # brewerytown.label_pics()
    # resize = Resizer(num_pixels=50)
    # resize.resize_pics()
    # pennsport = Labeler('pennsport',900,19147)
    # pennsport.label_pics()
    # resize = Resizer(num_pixels=50)
    # resize.resize_pics()
    # west_philly_north = Labeler('west_philly_north',250,19143)
    # west_philly_north.label_pics()
    # resize = Resizer(filepath='pics/labeled',
    #                 num_pixels=50,
    #                 resized_file_path='pics/resized',
    #                 move=True,
    #                 df_filepath='data/resized.pkl',
    #                 df_backup_filepath='data/backups',
    #                 df_backup_name='backup_resized'
    #                 )
    # resize.resize_pics()
    for neighborhood in ['brewerytown','fairmount','newbold','pennsport','west_philly_north']:
        resize = Resizer(filepath='pics/{}'.format(neighborhood),
                        num_pixels=50,
                        resized_file_path='pics/resized',
                        move=False,
                        df_filepath='data/all_resized.pkl',
                        df_backup_filepath='data/backups',
                        df_backup_name='backup_all_resized'
                        )
        resize.resize_pics()



#
#
#
#
# def _create_list_files(self):
#     '''
#     -- Creates attribute containing list of files in directory --
#
#         RETURNS
#         -----------
#         list of files from specified directory
#     '''
#     return [os.fsdecode(file) for file in os.listdir(self.folder_path)]
