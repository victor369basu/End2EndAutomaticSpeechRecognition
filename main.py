
import os
import torchaudio
import torch.utils.data as data_utils
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data

import mlflow 
from mlflow.tracking import MlflowClient

from Train import train
from Validate import validate
from Test import TestModel
from model import SpeechRecognitionModel
from FeatureExtraction import data_preprocessing

from config import get_parameters

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def trainer(learning_rate=5e-4,
         batch_size=20, 
         epochs=10,
         train_url="train-clean-100", 
         test_url="test-clean",
         run=None,
         config=None
         ):
    hparams = {
        "n_cnn_layers": config.n_cnn_layers,
        "n_rnn_layers": config.n_rnn_layers,
        "rnn_dim": config.rnn_dim,
        "n_class": config.n_class,
        "n_feats": config.n_feats,
        "stride": config.stride,
        "dropout": config.dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")
    
    if not os.path.isdir("./data"):
        os.makedirs("./data")

    train_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=train_url, download=True)
    test_dataset = torchaudio.datasets.LIBRISPEECH("./data", url=test_url, download=True)

    indices = torch.arange(2000)
    train_dataset = data_utils.Subset(train_dataset, indices)
    test_dataset = data_utils.Subset(test_dataset, indices)

    train_loader = data.DataLoader(
        dataset = train_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=lambda x: data_preprocessing(x, 'train'),
    )
    val_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=hparams['batch_size'],
        shuffle=True,
        collate_fn=lambda x: data_preprocessing(x, 'valid'),
        
    )
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], 
        hparams['n_rnn_layers'], 
        hparams['rnn_dim'],
        hparams['n_class'], 
        hparams['n_feats'], 
        hparams['stride'], 
        hparams['dropout'],
        config.alpha
    ).to(device)

    mlflow.log_params(hparams)
    mlflow.log_param("device", device)
    #print(model)
    #print('Num Model Parameters', sum([param.nelement() for param in model.parameters()]))

    optimizer = optim.AdamW(model.parameters(), 
                            hparams['learning_rate'],
                            betas=(config.beta1, config.beta1),
                            eps=config.epsilon
                            )
    criterion = nn.CTCLoss(blank=28,
        reduction='sum', 
        zero_infinity=True).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=hparams['learning_rate'], 
        steps_per_epoch=int(len(train_loader)),
        epochs=hparams['epochs'],
        anneal_strategy='linear'
    )
    
    iter_meter = IterMeter()
    
    for epoch in range(1, epochs + 1):
        model_trained = train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        validate(model, device, val_loader, criterion, epoch, iter_meter)
        if epoch == config.epoch:
            # convert to scripted model and log the model
            mlflow.pytorch.log_model(model_trained, 
                                     "model",
                                     registered_model_name=config.registered_model
                                     )
            
            scripted_pytorch_model = torch.jit.script(model)
            mlflow.pytorch.log_model(scripted_pytorch_model, 
                                     "scripted_model",
                                     registered_model_name=config.registered_scripted_model
                                     )
    
            print("run_id: {}".format(run.info.run_id))
            for artifact_path in ["model/data", "scripted_model/data"]:
                artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id,
                            artifact_path)]
                print("artifacts: {}".format(artifacts))
            client = MlflowClient()
            client.transition_model_version_stage(
                name=config.registered_model,
                version=config.model_version,
                stage=config.model_stage
            )
            client.transition_model_version_stage(
                name=config.registered_scripted_model,
                version=config.scripted_model_version,
                stage=config.scripted_model_stage
            )

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    learning_rate = config.learning_rate
    batch_size = config.batch
    epochs = config.epoch
    libri_train_set = "train-clean-100"
    libri_test_set = "test-clean"
    
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run(experiment_id='1',run_name=config.experiment_name,nested=True) as run:
        if config.train:
            trainer(learning_rate, batch_size, epochs, libri_train_set, libri_test_set, run, config)
        else:
            TestModel(config.registered_model, config.model_stage, config.dataset, config.file_id)