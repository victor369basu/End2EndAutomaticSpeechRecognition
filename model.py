import torch.nn as nn

class CNNLayerNorm(nn.Module):
    """Batch normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.BatchNorm2d(n_feats)

    def forward(self, x):
        x = x.transpose(2,3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time)

class ResidualCNN(nn.Module):
    """
        Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
    """
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats, alpha):
        super(ResidualCNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        self.dropout1 = nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats//2)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self,x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = self.leaky_relu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x += residual
        return x # (batch, channel, feature, time)

class BidirectionalGRU(nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, alpha, batch_first):
        super(BidirectionalGRU, self).__init__()
        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True
        )
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.leaky_relu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_classes, n_feats, stride, dropout, alpha):
        super(SpeechRecognitionModel, self).__init__()
        n_feats = n_feats//2
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)  # cnn for extracting heirachal features
        
        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(
            *[
              ResidualCNN(32, 32, kernel=3, stride=1, dropout=dropout, n_feats=n_feats, alpha=alpha)
              for _ in range(n_cnn_layers)
            ]
        )
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(
            *[
              BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                               hidden_size=rnn_dim, dropout=dropout, alpha=alpha, batch_first=i==0)
              for i in range(n_rnn_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),
            nn.LeakyReLU(alpha), 
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_classes)
        )
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x