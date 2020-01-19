import os
import numpy as np
import h5py
import json
import torch
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample


def create_input_files(dataset, karpathy_json_path, features_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=100, word_map_file=None, clean=False):
    """
    Creates input files for training, validation, and test data.
    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_features_paths = []
    train_image_captions = []
    train_ind = []
    val_features_paths = []
    val_image_captions = []
    val_ind = []
    test_features_paths = []
    test_image_captions = []
    test_ind = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])

        if len(captions) == 0:
            continue
        feat_name = img['filename'][:-4] + '.npy'
        path = os.path.join(features_folder, img['filepath'], feat_name) if dataset == 'coco' else os.path.join(
            features_folder, feat_name)

        if img['split'] in {'TRAIN','train', 'restval'}:
            train_features_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val','VAL'}:
            val_features_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_features_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_features_paths) == len(train_image_captions)
    assert len(val_features_paths) == len(val_image_captions)
    assert len(test_features_paths) == len(test_image_captions)

    if word_map_file != None:
        # Load existing word map
        with open(word_map_file, 'r') as j:
            word_map = json.load(j)
    else:
        # Create word map
        words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
        word_map = {k: v + 1 for v, k in enumerate(words)}
        word_map['<unk>'] = len(word_map) + 1
        word_map['<start>'] = len(word_map) + 1
        word_map['<end>'] = len(word_map) + 1
        word_map['<pad>'] = 0

    # Create a base/root name for all output files
    base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    if word_map_file == None:
        with open(os.path.join(output_folder, 'WORDMAP_' + base_filename + '.json'), 'w') as j:
            json.dump(word_map, j)

    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    seed(123)

    if clean:
        for featpaths, imcaps, split in [(train_features_paths, train_image_captions, 'TRAIN'),
                                         (val_features_paths, val_image_captions, 'VAL'),
                                         (test_features_paths, test_image_captions, 'TEST')]:
            for i, path in enumerate(tqdm(featpaths)):
                captions = imcaps[i]
                unk = False
                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    if word_map['<unk>'] in enc_c:
                        unk =True
                        break
                        
                if unk==False:
                    if split == 'TRAIN':
                        train_ind.append(i)
                    if split == 'VAL':
                        val_ind.append(i)
                    if split =='TEST':
                        test_ind.append(i)

        print('Clean Train = ', len(train_ind) / len(train_features_paths), ' ', len(train_features_paths))
        print('Clean Val = ', len(val_ind) / len(val_features_paths), ' ', len(val_features_paths))
        print('Clean Test = ', len(test_ind) / len(test_features_paths), ' ', len(test_features_paths))

    for featpaths, imcaps, split, ind in [(train_features_paths, train_image_captions, 'TRAIN', train_ind),
                                          (val_features_paths, val_image_captions, 'VAL', val_ind),
                                          (test_features_paths, test_image_captions, 'TEST', test_ind)]:

        with h5py.File(os.path.join(output_folder, split + '_FEATURES_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store features
            features = h.create_dataset('features', (len(ind), 36, 2048), dtype='float32')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(featpaths)):
                i_index=-1
                if i in ind:
                    i_index+=1

                    # Sample captions
                    if len(imcaps[i]) < captions_per_image:
                        captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                    else:
                        captions = sample(imcaps[i], k=captions_per_image)

                    # Sanity check
                    assert len(captions) == captions_per_image

                    # Read features
                    # feat_dic = np.load(featpaths[i])[()]
                    feat = np.load(featpaths[i])
                    assert feat.shape == (36, 2048)

                    # Save features to HDF5 file
                    features[i_index] = feat

                    for j, c in enumerate(captions):
                        # Encode captions
                        enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                            word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                        # Find caption lengths
                        c_len = len(c) + 2

                        enc_captions.append(enc_c)
                        caplens.append(c_len)

            # Sanity check
            assert features.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.
    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def save_checkpoint(data_name, epoch, epochs_since_improvement, decoder, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param decoder: decoder model
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'decoder': decoder,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + str(epoch) + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

