import torch
import torch.nn as nn
import torchaudio

from char_map import TextTransform
from config import get_parameters

config = get_parameters()

train_audio_transformer = nn.Sequential(
    torchaudio.transforms.MFCC(sample_rate=config.sampling_rate, 
                               n_mfcc=config.n_mfcc),
    #torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    #torchaudio.transforms.TimeMasking(time_mask_param=100)
)
valid_audio_transforms = torchaudio.transforms.MFCC(sample_rate=config.sampling_rate, 
                                                    n_mfcc=config.n_mfcc)

text_transform = TextTransform()

def data_preprocessing(data, data_type='train'):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transformer(waveform).squeeze(0).transpose(0,1)
        elif data_type == 'valid':
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths