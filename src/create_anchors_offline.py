from os.path import join, dirname, realpath
from ssloc_offline import ssloc

def main():

    data_folder = '/home/zaid/datasets/processed'
    
    images_folder = join(data_folder, 'images')

    # # create new anchors    
    n = ssloc(data_folder=images_folder, create_new_anchors=True, detector='loftr',matcher='loftr')
    n.create_offline_anchors()    
    # n.create_offline_anchors(skip=1, num_images=250)    

if __name__ == '__main__':
    main()