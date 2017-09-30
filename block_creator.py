import numpy as np
import os

class BlockCreator(object):
    def __init__(self,neighborhood):
        self.dict = self._read_dict(neighborhood)
        self.neighborhood = neighborhood

    def _read_dict(self,neighborhood):
        with open('data/{}.txt'.format(neighborhood),'r') as f:
            return eval(f.read())

    def get_blocks(self):
        blk_dict = self._get_dict()
        self.blocks = self._create_blocks(dct = blk_dict)

    def _get_dict(self):
        # get dict of blocks
        blocks = self.dict[self.neighborhood]['n_s']
        blocks.update(self.dict[self.neighborhood]['e_w'])
        return blocks

    def _create_blocks(self,dct):
        blocks = []
        for street in dct.keys():
            current_block = dct[street][0]
            end_block = dct[street][1]
            while current_block <= end_block:
                blocks.append((street,current_block))
                current_block += 100
        return blocks

    def sanity_check(self):
        print(self.dict)
        print('\n\n')
        print(self.blocks)

class Randomizer(BlockCreator):

    def __init__(self,neighborhood,num_pics,random_seed=17):
        super(self.__class__,self).__init__(neighborhood)
        self.num_pics = num_pics
        self.folder_path = 'pics/{}'.format(neighborhood)
        self._initialize(neighborhood)
        self.random_seed=random_seed

    def get_random_pics(self):
        np.random.seed(self.random_seed)
        # shuffle blocks
        np.random.shuffle(self.blocks)
        self.pics_to_label = []
        while len(self.pics_to_label) < self.num_pics:
            block = self.blocks.pop(0)
            for num in self._get_addresses(block).intersection(self.set_of_files):
                self.pics_to_label.append(num)

    def _initialize(self,neighborhood):
        self.get_blocks()
        self.set_of_files = self._get_set_of_pics()

    def _get_set_of_pics(self):
        # get file name, split off '.jpg' and take out zipcode
        # e.g. 1345_n_26th_st_philadelphia_pa_19125.jpg
        # -> 1345 n 26th st philadelphia pa
        return {' '.join(os.fsdecode(file).split('_')[:-1]) for file in os.listdir(self.folder_path)}

    def _get_addresses(self,block):
        return {'{} {} philadelphia pa'.format(num,block[0]) for num in range(block[1],block[1]+100)}


if __name__ == '__main__':
    # fairmount = BlockCreator('fairmount')
    # fairmount.get_blocks()
    fairmount = Randomizer('fairmount',1000)
