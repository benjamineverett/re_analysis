

''' ----------------  Code for FETCHING pictures --------------- '''

def get_neighborhood_dict(neighborhood):
    # get dict containing: {neighborhood:{st_direction:{st_name, (block start, block end)}}}
    #                      {'fairmount':{'n_s': {'n 22nd st', (800,000)}},'brewerytown'..........}}}
    with open('data/{}.txt'.format(neighborhood),'r') as f:
        return eval(f.read())

def get_neighborhood_streets(neighborhood):
    neighborhoods = get_neighborhood_dict(neighborhood)
    # FetchImages will save to folder specified by neighborhood
    # get dictionary specific to neighborhood, get n_s and e_w variables
    return neighborhoods[neighborhood], tuple(neighborhoods[neighborhood].keys())

def get_them_pics(neighborhood,zip_code):
    from fetch_images import FetchImages
    # instaniate FetchImages class
    instantiated_class = FetchImages(which_API='GSV',
                                    save_to='pics/{}'.format(neighborhood))
    streets, directions = get_neighborhood_streets(neighborhood)
    # get n_s, e_w
    for direction in directions:
        # get street names
        for street in streets[direction].keys():
            # get block start and block end for neighborhood
            blk_st, blk_end = streets[direction][street]
            # fit info to class
            instantiated_class.fit_info(street_name=street,
                                        zip_code=zip_code,
                                        blk_st=blk_st,
                                        blk_end=blk_end)
            # set heading for picture to be taken and fetch pic
            # either right or left
            if direction == 'n_s':
                instantiated_class.fetch_pictures(even_heading='left')
            if direction == 'e_w':
                instantiated_class.fetch_pictures(even_heading='left')

def test_pics_for_small_sample(street_name,zip_code,blk_st,blk_end,even_heading):
    from fetch_images import FetchImages
    instantiated_class = FetchImages(which_API='GSV',save_to='/Users/benjaminreverett/Desktop/test')
    instantiated_class.fit_info(street_name,zip_code,blk_st,blk_end)
    instantiated_class.fetch_pictures(even_heading)


''' -----------------  Code for LABELING pics  ----------------- '''

def label_pics():
    from label_pics import Labeler
    fairmount = Labeler('fairmount',800,19130)
    fairmount.label_pics()

    brewerytown = Labeler('brewerytown',900,19121)
    brewerytown.label_pics()
    resize = Resizer(num_pixels=50)
    resize.resize_pics()
    pennsport = Labeler('pennsport',900,19147)
    pennsport.label_pics()

''' --------------- Code for RESIZING pics --------------- '''
from fetch_images import Resizer
def resize_pics():
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


''' ---------------- Code for TRAINING neural network ---------------'''

    ''' Use cnn.py for tweaking neural network and running, use this function when model is
    returning results that should run through the pipeline
    '''
def run_NN():
    from cnn import NeuralNetwork
    NN = NeuralNetwork()
    NN.import_data(label_file_path='data/labeled.pkl',
                    array_file_path='data/resized.pkl',
                    merge_on='filename')

    NN.train_test_split(X_col_name='np_array',
                        y_col_name='label',
                        train_split=0.8)

    NN.set_parameters(random_seed=17,
                        batch_size=10,
                        classes=10,
                        epochs=10,
                        image_dims=(100,50),
                        num_filters=5,
                        pool = (3, 3),
                        kern_size = (3, 3),
                        colors=3)
    NN.run_models()
    NN.output_predictions()

    # push throughn DFOps to get data
    # into correct formate for TreeData
    data = DFOps(filepath_to_df=NN.filename)
    data.perform_all_ops()
    data.df.to_pickle('test')

    # Get back RMSE, precision and recall
    trees = TreeData('test','data/labeled_edited.pkl')
    trees.get_all()
    print(trees.tfRMSE())


''' ---------------- IF NAME MAIN --------------- '''

if __name__ == '__main__':
    # get_them_pics('west_philly_north',19143)
    # get_them_pics('fairmount',19130)
    # get_them_pics('brewerytown',19121)
    # get_them_pics('newbold',19145)
    # get_them_pics('pennsport',19148)
    # get_them_pics('pennsport',19147)
    # label_pics(file_path_to_label='/pics/brewerytown/batch1',
            #    file_path_of_labeled='/pics/to_resize')
    # resize_labeled_pics(filepath_to_resize='/Users/benjaminreverett/Desktop/test/labeled',
    #                     filepath_store_resized='/Users/benjaminreverett/Desktop/test/resized',
    #                     num_pixels=50,
    #                     show_resized_pic=False
    #                     )
    # run_NN()
    # test_pics_for_small_sample(street_name='fairmount ave',
    #                             zip_code = 19130,
    #                             blk_st=1800,
    #                             blk_end=2300,
    #                             even_heading='left')

    # resize_pics()
