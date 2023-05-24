# ###################################
# @author:Fxj
# @email:fxjswjtu@my.swjtu.edu.cn
# refer to learning opencv
# ###################################

import numpy as np
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

WINDOW_SIZE = 32  # 32 continuous frames
TOT_ACTION_CLASSES1 = 5
TOT_ACTION_CLASSES2 = 4
file=open("E:\GitHub\my_mediapipe\lightning_logs/training.txt","w", encoding="utf-8")
class PoseDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PoseDataModule(pl.LightningDataModule):
    def __init__(self, data_root, batch_size):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.X_train_path = self.data_root + "X_train.txt"
        self.X_test_path = self.data_root + "X_test.txt"
        self.y_train_path = self.data_root + "Y_train.txt"
        self.y_test_path = self.data_root + "Y_test.txt"


    def convert_to_detectron_format(self, row):
        row = row.split(',')
        # filtering out coordinate of neck joint from the training/validation set originally generated using OpenPose.
        #temp = row  ##34
        # change to Detectron2 order of key points
        temp = [row[i] for i in range(len(row))]  # 34
        return temp
    def load_X(self, X_path):
        file = open(X_path, 'r')
        X = np.array(
            [elem for elem in [
                self.convert_to_detectron_format(row) for row in file
            ]],
            dtype=np.float32
        )
        file.close()
        blocks = int(len(X) / WINDOW_SIZE)
        X_ = np.array(np.split(X, blocks))
        return X_

    # Load the networks outputs
    def load_y(self, y_path):
        file = open(y_path, 'r')
        y = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()
        # for 0-based indexing
        return y - 1

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        X_train = self.load_X(self.X_train_path)
        X_test = self.load_X(self.X_test_path)
        y_train = self.load_y(self.y_train_path)
        y_test = self.load_y(self.y_test_path)
        self.train_dataset = PoseDataset(X_train, y_train)
        self.val_dataset = PoseDataset(X_test, y_test)

    def train_dataloader(self):
        # train loader
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        # validation loader
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        return val_loader

##单臂分类模型
#lstm classifier definition
class ActionClassificationLSTM(pl.LightningModule):
    # initialise method
    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
        #hidden_size　LSTM中隐层的维度
        #batch_first 这个要注意，通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),
        #而batch_first默认是False,所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
        self.lstm = nn.LSTM(input_features, hidden_dim,num_layers=8, batch_first=True) #input_features=32 hidden_dim=50
        # The linear layer that maps from hidden state space to classes
        self.linear1=nn.Linear(hidden_dim,int(1/2*hidden_dim))
        self.linear2=nn.Linear(int(1/2*hidden_dim),int(1/4*hidden_dim))

        self.linear3= nn.Linear(int(1/4*hidden_dim), TOT_ACTION_CLASSES1)

    def forward(self, x):
        # invoke lstm layer
        lstm_out, (ht, ct) = self.lstm(x)
        # invoke linear layer              #ht:torch.Size([1,1,50])
        return self.linear3(self.linear2(self.linear1(ht[-1])))

    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_train_loss': loss,
            'batch_train_acc': acc
        }
        # log the metrics for pytorch lightning progress bar or any other operations
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc', acc, prog_bar=True)
        #return loss and dict
        return {'loss': loss, 'result': dic}

    def training_epoch_end(self, training_step_outputs):
        # calculate average training loss end of the epoch
        avg_train_loss = torch.tensor([x['result']['batch_train_loss'] for x in training_step_outputs]).mean()
        # calculate average training accuracy end of the epoch
        avg_train_acc = torch.tensor([x['result']['batch_train_acc'] for x in training_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('train_loss', avg_train_loss, prog_bar=True)
        self.log('train_acc', avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_val_loss': loss,
            'batch_val_acc': acc
        }
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('batch_val_loss', loss, prog_bar=True)
        self.log('batch_val_acc', acc, prog_bar=True)
        #return dict
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        # calculate average validation loss end of the epoch
        avg_val_loss = torch.tensor([x['batch_val_loss']
                                     for x in validation_step_outputs]).mean()
        # calculate average validation accuracy end of the epoch
        avg_val_acc = torch.tensor([x['batch_val_acc']
                                    for x in validation_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_acc', avg_val_acc, prog_bar=True)

    def configure_optimizers(self):
        # adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # learning rate reducer scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-15, verbose=True)
        # scheduler reduces learning rate based on the value of val_loss metric
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}}

             

##双臂分类模型
#lstm classifier definition
class ActionClassificationLSTM2(pl.LightningModule):
    # initialise method
    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super().__init__()
        # save hyperparameters
        self.save_hyperparameters()
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #input_size 输入数据的特征维数，通常就是embedding_dim(词向量的维度)
        #hidden_size　LSTM中隐层的维度
        #batch_first 这个要注意，通常我们输入的数据shape=(batch_size,seq_length,embedding_dim),
        #而batch_first默认是False,所以我们的输入数据最好送进LSTM之前将batch_size与seq_length这两个维度调换
        self.lstm = nn.LSTM(input_features, hidden_dim,num_layers=10, batch_first=True) #input_features=32 hidden_dim=50
        # The linear layer that maps from hidden state space to classes
        self.linear1=nn.Linear(hidden_dim,int(1/2*hidden_dim))
        self.linear2=nn.Linear(int(1/2*hidden_dim),int(1/4*hidden_dim))

        self.linear3= nn.Linear(int(1/4*hidden_dim), TOT_ACTION_CLASSES2)

    def forward(self, x):
        # invoke lstm layer
        lstm_out, (ht, ct) = self.lstm(x)   #x: 2，32，96 ht:10,2,192
        # invoke linear layer              #ht:torch.Size([1,1,50])
        x=self.linear1(ht[-1]) #2 96
        x=self.linear2(x) #2 48
        x=self.linear3(x) #2 4
        return 

    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_train_loss': loss,
            'batch_train_acc': acc
        }
        # log the metrics for pytorch lightning progress bar or any other operations
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc', acc, prog_bar=True)
        #return loss and dict
        return {'loss': loss, 'result': dic}

    def training_epoch_end(self, training_step_outputs):
        # calculate average training loss end of the epoch
        avg_train_loss = torch.tensor([x['result']['batch_train_loss'] for x in training_step_outputs]).mean()
        # calculate average training accuracy end of the epoch
        avg_train_acc = torch.tensor([x['result']['batch_train_acc'] for x in training_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('train_loss', avg_train_loss, prog_bar=True)
        self.log('train_acc', avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_val_loss': loss,
            'batch_val_acc': acc
        }
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('batch_val_loss', loss, prog_bar=True)
        self.log('batch_val_acc', acc, prog_bar=True)
        #return dict
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        # calculate average validation loss end of the epoch
        avg_val_loss = torch.tensor([x['batch_val_loss']
                                     for x in validation_step_outputs]).mean()
        # calculate average validation accuracy end of the epoch
        avg_val_acc = torch.tensor([x['batch_val_acc']
                                    for x in validation_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_acc', avg_val_acc, prog_bar=True)

    def configure_optimizers(self):
        # adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # learning rate reducer scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-15, verbose=True)
        # scheduler reduces learning rate based on the value of val_loss metric
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}}
    

class BiLSTMWithAttention(pl.LightningModule):
    def __init__(self, input_features, hidden_dim, learning_rate=0.001):
        super(BiLSTMWithAttention, self).__init__()
        self.save_hyperparameters()
        self.input_size= input_features
        self.hidden_size = hidden_dim
        self.list=[]
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers=4, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(self.hidden_size, num_heads=1)
        self.fc = nn.Linear(self.hidden_size , TOT_ACTION_CLASSES2)

    def forward(self, x):
        # LSTM
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # initial hidden state 4 16 64
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # initial cell state 4 16 64
        out, (ht,ct) = self.lstm(x)   #out: 16 50 2*64  ht: 4,16,64

        # Attention
        # out = out.permute(0, 2, 1)  # swap dimensions for attention  16 128 50
        out, _ = self.attention(ht, ht, ht) #4 16 64
        # out = out.permute(0, 2, 1)  # swap dimensions back

        # Classification
        out = F.relu(out[-1]) 
        out = self.fc(out)

        return out
    def training_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_train_loss': loss,
            'batch_train_acc': acc
        }
        # log the metrics for pytorch lightning progress bar or any other operations
        self.log('batch_train_loss', loss, prog_bar=True)
        self.log('batch_train_acc', acc, prog_bar=True)
        #return loss and dict
        # self.list.append(loss)
        # self.list.append('\n')
        file.write(str(loss.cpu().detach().numpy())+'\n')
        return {'loss': loss, 'result': dic}

    def training_epoch_end(self, training_step_outputs):
        # calculate average training loss end of the epoch
        avg_train_loss = torch.tensor([x['result']['batch_train_loss'] for x in training_step_outputs]).mean()
        # calculate average training accuracy end of the epoch
        avg_train_acc = torch.tensor([x['result']['batch_train_acc'] for x in training_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('train_loss', avg_train_loss, prog_bar=True)
        self.log('train_acc', avg_train_acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # get data and labels from batch
        x, y = batch
        # reduce dimension
        y = torch.squeeze(y)
        # convert to long
        y = y.long()
        # get prediction
        y_pred = self(x)
        # calculate loss
        loss = F.cross_entropy(y_pred, y)
        # get probability score using softmax
        prob = F.softmax(y_pred, dim=1)
        # get the index of the max probability
        pred = prob.data.max(dim=1)[1]
        # calculate accuracy
        acc = torchmetrics.functional.accuracy(pred, y)
        dic = {
            'batch_val_loss': loss,
            'batch_val_acc': acc
        }
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('batch_val_loss', loss, prog_bar=True)
        self.log('batch_val_acc', acc, prog_bar=True)
        #return dict
        return dic

    def validation_epoch_end(self, validation_step_outputs):
        # calculate average validation loss end of the epoch
        avg_val_loss = torch.tensor([x['batch_val_loss']
                                     for x in validation_step_outputs]).mean()
        # calculate average validation accuracy end of the epoch
        avg_val_acc = torch.tensor([x['batch_val_acc']
                                    for x in validation_step_outputs]).mean()
        # log the metrics for pytorch lightning progress bar and any further processing
        self.log('val_loss', avg_val_loss, prog_bar=True)
        self.log('val_acc', avg_val_acc, prog_bar=True)

    def configure_optimizers(self):
        # adam optimiser
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # learning rate reducer scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-15, verbose=True)
        # scheduler reduces learning rate based on the value of val_loss metric
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1, "monitor": "val_loss"}}