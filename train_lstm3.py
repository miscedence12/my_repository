'''
双臂模型训练
'''

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from lstm_hand.lstm import BiLSTMWithAttention, PoseDataModule

##数据集路径
DATASET_PATH = "双臂视频/"
from argparse import ArgumentParser

def configuration_parser(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--data_root', type=str, default=DATASET_PATH)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_class', type=int, default=5)
    return parser

def do_training_validation():
    pl.seed_everything(21)
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = configuration_parser(parser)
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # 模型初始化
    model = BiLSTMWithAttention(96, 96*2, learning_rate=args.learning_rate)
    data_module = PoseDataModule(data_root=args.data_root,
                                        batch_size=args.batch_size)
    #保存val_loss最小的模型
    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    #trainer
    trainer = pl.Trainer.from_argparse_args(args,
        # fast_dev_run=True,
        max_epochs=args.epochs,
        deterministic=True,
        gpus=1,
        progress_bar_refresh_rate=1,
        callbacks=[EarlyStopping(monitor='train_loss', patience=15), checkpoint_callback, lr_monitor])
    trainer.fit(model, data_module)
    return model

if __name__ =="__main__":
    do_training_validation()