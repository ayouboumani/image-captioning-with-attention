from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='foot',
                       karpathy_json_path='../Dataset/dataset_football.json',
                       features_folder='../Dataset/features',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='../Dataset/football_dataset',
                       max_len=50,
                       word_map_file='../data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
