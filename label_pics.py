'''
This class file contains two classes:
    Labler: used to label pics
    Resizer: used to resize pics
'''
import os # for file operations
import cv2 # imaging library
import numpy as np # numpy!!
# from load_image import ViewPhoto
import pandas as pd
import pdb

class Labeler(object):
    '''
    Labeler will:
        1) Fetch files from specified folder and display
        the pictures on screen for labeling

        2) Save files to new folder for accurate record keeping
        and to resize later

        3) Create panda's data frame containing filename, label

        On mac:
            Fn + left arrow key = 0
            Fn + right arrow key = 1
    '''

    def __init__(self,folder_path,label_pics_folder_path):
        '''
        -- Initializes class --

        PARAMETERS
        ----------
            folder_path: str
                file path of folder containing pics to label
            label_pics_folder_path: str
                file path of folder to save label pics to
        '''
        self.folder_path = folder_path
        # get list of files in folder
        self.list_of_files = self._create_list_files()
        # get count of number of files in folder
        self.file_count = len(self.list_of_files)
        self.label_pics_folder_path = label_pics_folder_path

    def label_pics(self,save_label_file_path):
        '''
        -- Main block to run code for labeling --

            PARAMETERS
            ----------
                save_label_file_path: str
                    specify name of file to save labeled data as

            RETURNS
            -------
                saves to pkl: filename, label
        '''
        pic_counter = 1
        list_of_labels = []
        for pict in self.list_of_files:
            if pict == '.DS_Store':
                pass
            else:
                # get pictures
                pic1, pic2, flip1, flip2 = self._create_all_pics(p=pict)
                # loop through pics and enumerate for filenaming purposes
                for i,pic in enumerate([(pic1,flip1),(pic2,flip2)]):
                    # display picture
                    self._show_pic(picture=pic[0],counter=pic_counter)
                    # save picture and flip of picture
                    cv2.imwrite('{}/{}_pic{}.jpg'.format(self.label_pics_folder_path,pict,i+1),pic[0])
                    cv2.imwrite('{}/{}_flip{}.jpg'.format(self.label_pics_folder_path,pict,i+1),pic[1])
                    # append label and label for flip pic
                    list_of_labels.append(['{}_pic{}.jpg'.format(pict,i+1),self.label])
                    list_of_labels.append(['{}_flip{}.jpg'.format(pict,i+1),self.label])
                    # save every 100 labels
                    if pic_counter == 100:
                        self._save_labels(list_of_labels)
            pic_counter += 1
        # close all windows
        cv2.destroyAllWindows()
        # save labels to file
        self._save_labels(list_of_labels)

    ''' --------------- BEGIN HIDDEN METHODS --------------- '''

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
        pic = cv2.imread('{}/{}'.format(self.folder_path,picture))
        # split picture vertically
        pic1 = pic[0:600,0:300]
        pic2 = pic[0:600,301:600]
        return pic1, pic2

    def _show_pic(self,picture,counter):
        '''
        -- Displays picture and returns boolean --

            PARAMETERS
            ----------
                picture: np.array
                    cv2.imread np.array of picture

                counter: int
                    count of number of pics labeled

            INSTANTIATED
            ------------
                self.label:bool value
                    used to append to list of lists for panda's data frame
        '''

        # set title on image displayer
        pic_title = '{} of {}'.format(counter,self.file_count)
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
        # check if dataframe exists, if not, create new dataframe
        if os.path.isfile('data/labeled.pkl'):
            old_df = pd.read_pickle('data/labeled.pkl')
            # save old df as backup
            old_df.to_pickle('data/backup_old_{}_labeled.pkl'.format(time.ctime().lower().replace(' ','_')))
            self.df = pd.DataFrame(labeled_pics,columns=['filename', 'label'])
            self.df.to_pickle('data/backup_new_{}_labeled.pkl'.format(time.ctime().lower().replace(' ','_')))
            new_df = old_df.append(self.df)
            new_df.reset_index(inplace=True,drop=True)
            new_df.to_pickle('data/labeled.pkl')
        else:
            self.df = pd.DataFrame(labeled_pics,columns=['filename', 'label'])
            self.df.to_pickle('data/labeled.pkl')

    def _create_list_files(self):
        '''
        -- Creates attribute containing list of files in directory --

            RETURNS
            -----------
            list of files from specified directory
        '''
        return [os.fsdecode(file) for file in os.listdir(self.folder_path)]

    ''' --------------- END HIDDEN METHODS --------------- '''

class Resizer(object):
    '''
    -- Initialize class --

        PARAMETERS
        ----------
            filepath: str
                filepath to folder where images to resize reside
            resized_file_path: str
                filepath to folder to place resized images
            dataframe_name: str
                files saved into panadas datframe as filename, np.array
                specify name to give to data frame
            num_pixels: int
                number of pixels to scale larger height to
                    note: perspective maintained
                e.g. original image = 600x300
                        num_pixels = 100 -> rescaled image 100x50
            show_resized_pic: bool
                default = False
                display resized photos at end
    '''

    def __init__(self,
                filepath,
                resized_file_path,
                num_pixels,
                show_resized_pic=False):
        self.filepath = filepath
        self.resized_file_path = resized_file_path
        self.num_pixels = num_pixels
        self.show_resized_pic = show_resized_pic


    def resize_pics(self,to_np_array=True):
        '''
        -- Main block to run Resizer --

            PARAMETERS
            ----------
                to_np_array: bool
                    default = True
                    if False -> resize and save pics only
                    if True -> resize pics and save in dataframe as numpy array
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
                # give a little feedback, make us feel like the computer is working
                print('Resizing {}. Picture {} of {}'.format(picture,counter,len(files)))
                # resize picture
                resized = self._resize_pic(picture)
                # if True, show picture
                if self.show_resized_pic:
                    self._show_pic()
                # write picture to file
                cv2.imwrite('{}/{}'.format(self.resized_file_path,picture),resized)
                counter += 1
                # append to list as [filename, np.array]
                if to_np_array:
                    array.append([picture, np.array(cv2.imread('{}/{}'.format(self.resized_file_path,picture)))])
        # append list to dataframe
        if to_np_array:
            self._save_df(data=array)

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
        # resize according to height
        r = self.num_pixels / image.shape[1]
        # set dimensions of new image
        dim = (self.num_pixels, int(image.shape[0] * r))
        # resize image and return as numpy array
        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    def _show_pic(self,pic):
        '''
        -- Method to show picture on the screen --- '

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
                Called if to_np_array = True
                data: list of lists -> [['filename1.jpg',np.array],['filename2.jpg',np.array]]

            RETURNS
            -------
                Saves data as pickeled panda's data frame
        '''
        # check if dataframe exists, if not, create new dataframe
        if os.path.isfile('data/resized.pkl'):
            # read and save old df
            old_df = pd.read_pickle('data/resized.pkl')
            old_df.to_pickle('data/backup_old_{}_resized.pkl'.format(time.ctime().lower().replace(' ','_')))
            # create and back up new df
            self.df = pd.DataFrame(labeled_pics,columns=['filename', 'np_array'])
            self.df.to_pickle('data/backup_new_{}_resized.pkl'.format(time.ctime().lower().replace(' ','_')))
            # combine two dfs
            new_df = old_df.append(self.df)
            new_df.reset_index(inplace=True,drop=True)
            # overwrite old instance of resized.pkl
            new_df.to_pickle('data/resized.pkl')
        else:
            # create new df and save as pickle
            self.df = pd.DataFrame(labeled_pics,columns=['filename', 'np_array'])
            self.df.to_pickle('data/resized.pkl')

    ''' --------------- END HIDDEN METHODS --------------- '''

















''' --------------- BEGIN REMOVED FUNCTIONS --------------- '''

def check_for_repeats(self):
    '''
    -- Checks pics in folder for repeats --
        INPUT: None
        SELF: list_of_files
        REVISE:
            self.list_of_unique_pics = list of tuples containing (file name, vector) of unquie pics in folder
            self.num_unique_pics = number of unique picture
        RETURN: None
    '''

    list_of_unique_pics = []
    list_of_unique_vectors = []
    counter = 0
    num_of_files = len(self.list_of_files)
    for picture in self.list_of_files:
        print('Checking {}. File {} of {}'.format(picture,counter+1,num_of_files))
        # imread returns np array of photo
        pic = cv2.imread('{}/{}'.format(self.folder_path,picture))
        # append to lists if picture is unique
        if pic.tolist() not in list_of_unique_vectors:
            list_of_unique_pics.append(picture)
            list_of_unique_vectors.append(pic.tolist())
        counter += 1
    # update attributes
    self.num_unique_pics = len(list_of_unique_pics)
    # currently list_of_unique_vectors is list of lists, return to list of np arrays
    unique_vectors = np.array(list_of_unique_vectors)
    # update self.list_of_unique_pics = [(file name, np.array of vector)]
    self.list_of_unique_pics = list(zip(list_of_unique_pics,unique_vectors))
    print('\nTotal pictures processed = {}\nTotal unique pictures = {}'.format(counter,self.num_unique_pics))
