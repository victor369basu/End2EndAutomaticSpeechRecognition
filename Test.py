import numpy as np
import os

import mlflow

import torch.nn.functional as F
import torchaudio
import torch

from FeatureExtraction import data_preprocessing
from decoder import GreedyDecoder

def TestModel(model_name, Stage, base_dir, file_id):
    model_production_uri = "models:/{model_name}/{Stage}".format(model_name=model_name,Stage=Stage)

    # Loading registered model version from URI
    print("Loading registered model version from URI: '{model_uri}'".format(model_uri=model_production_uri))
    model_production = mlflow.pyfunc.load_model(model_production_uri)
 
    BASE_PATH = base_dir #'/content/data/LibriSpeech/test-clean'
    ext_txt = ".trans.txt"
    ext_audio = ".flac"
    speaker_id, chapter_id, utterance_id = file_id.split("-")

    file_text = speaker_id + "-" + chapter_id + ext_txt
    file_text = os.path.join(BASE_PATH, speaker_id, chapter_id, file_text)

    fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
    file_audio = fileid_audio + ext_audio
    file_audio = os.path.join(BASE_PATH, speaker_id, chapter_id, file_audio)

    # Load audio
    waveform, sample_rate = torchaudio.load(file_audio)

    # Load text
    with open(file_text) as ft:
        for line in ft:
            fileid_text, utterance = line.strip().split(" ", 1)
            if fileid_audio == fileid_text:
                break
        else:
            # Translation not found
            raise FileNotFoundError("Translation not found for " + fileid_audio)
                              

    data = [[waveform, sample_rate, utterance, int(speaker_id), int(chapter_id), int(utterance_id)]]
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    spectrograms, labels, input_lengths, label_lengths = data_preprocessing(data, 'valid')
    spectrograms, labels = spectrograms, labels
    
    output = model_production.predict(np.array(spectrograms))  # (batch, time, n_class)
    output = F.log_softmax(torch.tensor(output), dim=2)
    output = output.transpose(0, 1) # (time, batch, n_class)

    decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0,1), labels, label_lengths)

    print("Predicted Speech : "+decoded_preds[0]+"\nTarget Speech : "+decoded_targets[0])
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test_{file_id}.txt".format(file_id=file_id), "w") as f:
        f.write("Predicted Speech : "+decoded_preds[0]+"\nTarget Speech : "+decoded_targets[0])
        
    mlflow.log_artifacts("outputs")