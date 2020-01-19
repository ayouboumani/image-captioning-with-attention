
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import DecoderWithAttention
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.functional import softmax
from nlgeval import compute_metric
from nlgeval.pycocoevalcap.cider.cider import Cider
from nlgeval.pycocoevalcap.rouge.rouge import Rouge
# Data parameters
data_folder = '../Dataset/football_dataset'  # folder with data files saved by create_input_files.py
data_name = 'foot_5_cap_per_img_5_min_word_freq'  # base name shared by data files

# Model parameters

emb_dim = 1024  # dimension of word embeddings
attention_dim = 1024  # dimension of attention linear layers
decoder_dim = 1024  # dimension of decoder RNN
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 100000  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 64
workers = 1  # for data-loading; right now, only 1 works with h5py
best_cidr = 0  # BLEU-4 score right now
print_freq = 10  # print training/validation stats every __ batches
checkpoint = 'BEST_0checkpoint_foot_5_cap_per_img_5_min_word_freq.pth.tar'  # path to checkpoint, None if none
word_map_file = '../data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'


def main():
    """Training and validation."""

    global best_cidr, epochs_since_improvement, checkpoint, start_epoch, data_name, word_map

    # Read word map
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)

        decoder_optimizer = torch.optim.Adamax(params=filter(lambda p: p.requires_grad, decoder.parameters()))

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_cidr = 0
        decoder = checkpoint['decoder']
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=5*1e-5)

    # Move to GPU, if available
    decoder = decoder.to(device)

 
    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'TRAIN'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, data_name, 'VAL'),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 200:
            break
        #if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            #adjust_learning_rate(decoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              decoder=decoder,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              rev_word_map=rev_word_map)
        # One epoch's validation
        recent_cidr = validate(val_loader=val_loader,
                                decoder=decoder,rev_word_map=rev_word_map)

        # Check if there was an improvement
        is_best = recent_cidr > best_cidr
        best_cidr = max(recent_cidr, best_cidr)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, 0, epochs_since_improvement, decoder,decoder_optimizer, recent_cidr, is_best)


def train(train_loader, decoder, decoder_optimizer, epoch, rev_word_map):
    """
    Performs one epoch's training.
    :param train_loader: DataLoader for training data
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, allcaps) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        # Forward prop.
        scores,scores1, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

        # /!\ scores shape: (batch_size, max_captions_real_length,vocab_size)
        # scores[0, t, :]= proba(y[t]|y[1:t-1])

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]  # (batch_size, max_caption_real_length)

        scores_copy = scores.clone()
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate cross entropy
        # crit = criterion_xe(scores, targets)

        # References
        references = list()
        allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
        for j in range(allcaps.shape[0]):
            img_caps = allcaps[j].tolist()
            img_captions = list(map(lambda c: [rev_word_map[w] for w in c if
                                               w not in {word_map['<start>'], word_map['<pad>']}],
                                    img_caps))  # remove <start> and pads
            ref_caps = [' '.join(c) for c in img_captions]
            references.append(ref_caps)
         #print(references[-1])
        # Hypotheses
        hypotheses = list()
        _, preds = torch.max(scores_copy, dim=2)
        preds = preds.tolist()
        temp_preds = list()
        for j, p in enumerate(preds):
            temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
        preds = temp_preds
        # print(preds[0])
        preds_caption = list(map(lambda c: [rev_word_map[w] for w in c if
                                            w not in {word_map['<start>'], word_map['<pad>']}],
                                 preds))
        preds_caption = [' '.join(c) for c in preds_caption]

        hypotheses.extend(preds_caption)

        assert len(references) == len(hypotheses)

        # Sample decoding
        samples = list()
        proba = softmax(scores_copy, dim=2)
        B, T, V = proba.size()
        sampled = np.zeros((B, T), dtype=np.int32)
        sampled_entropy = torch.zeros([B, T]).to(device)
        for b in range(B):
            for t in range(decode_lengths[b]):
                sampled[b][t] = torch.multinomial(proba[b][t].view(-1), 1).item()
                sampled_entropy[b][t] = torch.log(proba[b][t][sampled[b][t]])
        temp_sampled = list()
        for j, p in enumerate(sampled):
            temp_sampled.append(sampled[j][:decode_lengths[j]])  # remove pads

        log_proba = torch.sum(sampled_entropy, dim=1)

        sampled_caption = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                temp_sampled))
        sampled_caption = [' '.join(c) for c in sampled_caption]

        samples.extend(sampled_caption)

        # print(samples)

        # Calculate loss
        cider = Cider()
        cider_ = Cider()

        baseline = torch.Tensor(compute_metric(cider_, references, hypotheses)).to(device)
        reward = torch.Tensor(compute_metric(cider, references, samples)).to(device)

        # print(log_proba.requires_grad)
        # loss = -(compute_metric(cider, references,samples) - compute_metric(cider_,references, hypotheses)) * crit
        loss = -torch.sum((reward-baseline) * log_proba)

        # Back prop.
        decoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients when they are getting too large
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, decoder.parameters()), 0.25)

        # Update weights
        decoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))

            print('Reward : ', torch.mean(reward).item())
            print('Baseline : ', torch.mean(baseline).item())


def validate(val_loader, decoder,rev_word_map):
    """
    Performs one epoch's validation.
    :param val_loader: DataLoader for validation data.
    :param decoder: decoder model
    :param criterion_ce: cross entropy loss layer
    :param criterion_dis : discriminative loss layer
    :return: BLEU-4 score
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references_ = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses_ = list()  # hypotheses (predictions)

    # Batches
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):

            # Move to GPU, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            scores, scores1,caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)

            # /!\ scores shape: (batch_size, max_captions_real_length,vocab_size)
            # scores[0, t, :]= proba(y[t]|y[1:t-1])

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]  # (batch_size, max_caption_real_length)

            scores_copy = scores.clone()
            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate cross entropy
            # crit = criterion_xe(scores, targets)

            # References
            references = list()
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(map(lambda c: [rev_word_map[w] for w in c if
                                                   w not in {word_map['<start>'],
                                                             word_map['<pad>']}],
                                        img_caps))  # remove <start> and pads
                ref_caps = [' '.join(c) for c in img_captions]
                references.append(ref_caps)

            # Hypotheses
            hypotheses = list()
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            # print(preds[0])
            preds_caption = list(map(lambda c: [rev_word_map[w] for w in c if
                                                w not in {word_map['<start>'], word_map['<pad>']}],
                                     preds))
            preds_caption = [' '.join(c) for c in preds_caption]

            hypotheses.extend(preds_caption)

            assert len(references) == len(hypotheses)

            # Sample decoding
            samples = list()
            proba = softmax(scores_copy, dim=2)
            B, T, V = proba.size()
            sampled = np.zeros((B, T), dtype=np.int32)
            sampled_entropy = torch.zeros([B, T]).to(device)
            for b in range(B):
                for t in range(decode_lengths[b]):
                    sampled[b][t] = torch.multinomial(proba[b][t].view(-1), 1).item()
                    sampled_entropy[b][t] = torch.log(proba[b][t][sampled[b][t]])
            temp_sampled = list()
            for j, p in enumerate(sampled):
                temp_sampled.append(sampled[j][:decode_lengths[j]])  # remove pads

            log_proba = torch.sum(sampled_entropy, dim=1)

            sampled_caption = list(
                map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                    temp_sampled))
            sampled_caption = [' '.join(c) for c in sampled_caption]

            samples.extend(sampled_caption)

            # print(samples)

            # Calculate loss
            cider = Cider()
            cider_ = Cider()

            baseline = torch.Tensor(compute_metric(cider_, references, hypotheses)).to(device)
            reward = torch.Tensor(compute_metric(cider, references, samples)).to(device)

            # print(log_proba.requires_grad)
            # loss = -(compute_metric(cider, references,samples) - compute_metric(cider_,references, hypotheses)) * crit
            loss = -torch.sum((reward-baseline) * log_proba)
            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader),
                                                                                batch_time=batch_time,
                                                                                loss=losses, top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
            references_.extend(references)
            hypotheses_.extend(hypotheses)
         
    # Calculate BLEU-4 scores
    bleu4 = corpus_bleu(references_, hypotheses_)
    bleu4 = round(bleu4, 4)

    #calculate CIDEr
    avg_cider=Cider()
    #print(references)
    #print(hypotheses)
    print(len(compute_metric(avg_cider, references_, hypotheses_)))
    avg_reward=np.mean(compute_metric(avg_cider, references_, hypotheses_))
    print('val reward', avg_reward)
    print(
        '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu} , CIDEr - {cidr}\n'.format(
            loss=losses,
            top5=top5accs,
            bleu=bleu4,
            cidr=avg_reward))

    return avg_reward


if __name__ == '__main__':
    main()


