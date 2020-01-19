import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import torch.nn.functional as F
from tqdm import tqdm
from matplotlib import pyplot as plt
import sys
import cv2
import numpy as np
import matplotlib.cm as cm

# Parameters
images_dir = '../images'
features_dir = '../features'
data_folder = '../data/coco_dataset'  # folder with data files saved by create_input_files.py
data_name = 'coco_5_cap_per_img_5_min_word_freq'  # base name shared by data files
checkpoint_file = 'BEST_34checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint

word_map_file = 'WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Load model
# Load model
torch.nn.Module.dump_patches = True
checkpoint = torch.load(checkpoint_file, map_location=device)
decoder = checkpoint['decoder']
decoder = decoder.to(device)
decoder.eval()
# Load word map (word2ix)
word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)


def demo(beam_size):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
    """

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    k = beam_size
    images = []
    features = []
    names = []
    boxes =[]

    for im in os.listdir(images_dir):
        if im[-4] == '.':
            q = 4
        else:
            q = 5
        im_file = im
        feat_file = im[:-q] + '.npy'

        im = plt.imread(os.path.join(images_dir, im_file))
        feat = np.load(os.path.join(features_dir, feat_file))
        feat = feat[()]
        boxes_ = np.array(feat['boxes'], dtype=int)
        features_=np.array(feat['features'])
        images.append(im)
        features.append(features_)
        boxes.append(boxes_)
        names.append(im_file[:-q])



    alphas_=[]
    seq_=[]
    for image, image_features, name in zip(images, features, names):
        print('Captioning ' + name + ' ...')
        k = beam_size
        image_features = image_features.reshape((-1, 36, 2048))
        # Move to GPU device, if available
        image_features = torch.from_numpy(image_features).float()
        image_features = image_features.to(device)
        image_features_mean = image_features.mean(1)
        image_features_mean = image_features_mean.expand(k, 2048)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        seqs_alpha=torch.ones(k,1,36).to(device)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_alpha=list()
        complete_seqs_scores = list()
        # Start decoding
        step = 1
        h1, c1 = decoder.init_hidden_state(k)  # (batch_size, decoder_dim)
        h2, c2 = decoder.init_hidden_state(k)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            h1, c1 = decoder.top_down_attention(
                torch.cat([h2, image_features_mean, embeddings], dim=1),
                (h1, c1))  # (batch_size_t, decoder_dim)

            attention_weighted_encoding, alpha = decoder.attention(image_features, h1)
            alpha=alpha.view(-1,36)
            h2, c2 = decoder.language_model(
                torch.cat([attention_weighted_encoding, h1], dim=1), (h2, c2))

            scores = decoder.fc(h2)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],dim=1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            seqs_alpha = seqs_alpha[incomplete_inds]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            image_features_mean = image_features_mean[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        seq_.append(seq)
        alphas = complete_seqs_alpha[i]
        alphas_.append(alphas)


    return names,images,seq_, alphas_,boxes

def visualize_att(names,images,seq_,alphas_, boxes_, rev_word_map):
    """
    Visualizes caption with weights at every word.
    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param rev_word_map: reverse word mapping, i.e. ix2word
    """
    for name,image,seq,alphas,boxes in zip(names,images,seq_,alphas_,boxes_):
        words = [rev_word_map[ind] for ind in seq]
        
        for t in range(len(words)):
            if t > 50:
                break
            plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)
            
            plt.text(0,1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
            image_t=np.copy(image)
            current_alpha = np.array(alphas[t]).reshape(36)
            i_max=np.argmax(current_alpha)
            b=boxes[i_max]
            overlay=np.copy(image_t)
            cv2.rectangle(overlay, (b[0], b[1]), (b[2], b[3]), color=(255, 0, 0), thickness=-1)
            image_=cv2.addWeighted(overlay,0.3,image_t,1-0.3,0)
            plt.imshow(image_)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')


        plt.savefig("../captions/" + name + ".png", bbox_inches='tight')
        plt.close()




if __name__ == '__main__':
    beam_size = 5
    names,images,seq_, alphas_,boxes_=demo(beam_size)
    visualize_att(names,images, seq_, alphas_, boxes_, rev_word_map)
