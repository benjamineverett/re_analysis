

''' ----------------  Code for FETCHING pictures --------------- '''

def get_neighborhood_dict():
    # get dict containing: {neighborhood:{st_direction:{st_name, (block start, block end)}}}
    #                      {'fairmount':{'n_s': {'n 22nd st', (800,000)}},'brewerytown'..........}}}
    with open('/Users/benjaminreverett/Desktop/philadelphia_streets_by_neighborhood.txt','r') as f:
        return eval(f.read())

def get_neighborhood_streets(neighborhood):
    neighborhoods = get_neighborhood_dict()
    # FetchImages will save to folder specified by neighborhood
    # get dictionary specific to neighborhood, get n_s and e_w variables
    return neighborhoods[neighborhood], tuple(neighborhoods[neighborhood].keys())

def get_them_pics(neighborhood,zip_code):
    from fetch_images import FetchImages
    # instaniate FetchImages class
    instantiated_class = FetchImages(which_API='GSV',save_to='/Users/benjaminreverett/Desktop/Pics/{}'.format(neighborhood))
    streets, directions = get_neighborhood_streets(neighborhood)
    # get n_s, e_w
    for direction in directions:
        # get street names
        for street in streets[direction].keys():
            # get block start and block end for neighborhood
            blk_st, blk_end = streets[direction][street]
            # fit info to class
            instantiated_class.fit_info(street,zip_code,blk_st,blk_end)
            # set heading for picture to be taken and fetch pic
            if direction == 'n_s':
                instantiated_class.fetch_pictures(even_heading=270,odd_heading=90)
            if direction == 'e_w':
                instantiated_class.fetch_pictures(even_heading=180,odd_heading=0)

''' -----------------  Code for LABELING pics  ----------------- '''

def label_pics(file_path_to_label,file_path_of_labeled):
    from label_pics import Labeler
    instantiated_class = Labeler(file_path_to_label,file_path_of_labeled)
    instantiated_class.label_pics('labeled')

def resize_label_pics(filepath_to_resize,
                    filepath_store_resized,
                    dataframe_name,
                    num_pixels,
                    show_resized_pic=False):
    from label_pics import Resizer

    instantiated_class = Resizer(filepath=filepath_to_resize,
                                    resized_file_path=filepath_store_resized,
                                    dataframe_name=dataframe_name,
                                    num_pixels=num_pixels,
                                    show_resized_pic=show_resized_pic)

    instantiated_class.resize_pics(to_np_array=True)

''' ---------------- Code for TRAINING neural network ---------------'''
def run_NN():
    from cnn import NeuralNetwork
    instantiated_class = NeuralNetwork()
    instantiated_class.import_data(label_file_path='data/labeled.pkl',
                                    array_file_path='data/dataframe.pkl',
                                    merge_on='filename',)

    instantiated_class.train_test_split(X_col_name='np_array',
                                        y_col_name='label',
                                        train_split=0.75)
    instantiated_class.set_parameters(random_seed=1337,
                                        batch_size=200,
                                        classes=10,
                                        epochs=500,
                                        image_dims=(100,50),
                                        num_filters=10,
                                        pool = (3, 3),
                                        kern_size = (3, 3),
                                        colors=3)
    instantiated_class.run_models()



''' ---------------- IF NAME MAIN --------------- '''

if __name__ == '__main__':

    # get_them_pics('brewerytown',19121)
    # label_pics(file_path_to_label='/Users/benjaminreverett/Desktop/Pics/brewerytown/00_random',
            #    file_path_of_labeled='/Users/benjaminreverett/Desktop/Pics/00_labeled')
    # resize_label_pics(filepath_to_resize='/Users/benjaminreverett/Desktop/Pics/00_labeled',
                        # filepath_store_resized='resized',
                        # num_pixels=50,
                        # show_resized_pic=False
                        # )
    run_NN()
