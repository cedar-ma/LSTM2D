import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dataloader_utils import NumpyDataset, get_dataloader
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import wandb
from lightning.pytorch.loggers import WandbLogger
import glob
from pathlib import Path
import json
from LSTM2D import ConvLSTMModel


if __name__ == "__main__":

    torch.set_float32_matmul_precision('medium')
    hparams = {
        'net_name': "lstm_h3",
        'learning_rate': 1e-3,
        'batch_size': 1,
        'epochs': 200,
        'val_interval': 5,
        'T_in': 10,
        'T_out': 30,
        'seed': 189031465,
        'input_channels': 1,
        'hidden_channels': 3,
        'kernel_size': 3,
        'num_layers': 1,
        'patience': 9999
    }

    # Load the data
    seed = 189031465

    n_samples = 101
    split = [0.6, 0.2, 0.2]
    image_ids = np.random.randint(low=0, high=100, size=(n_samples,))
    # train_dataloader, val_dataloader, test_dataloader = load_data()
    train_dataloader, val_dataloader, test_dataloader = get_dataloader(image_ids, 
                                                                       data_path="/scratch/08780/cedar996/lbfoam/fno/lbfoam_rough",
                                                                       t_in=hparams['T_in'],
                                                                       t_out=hparams['T_out'],
                                                                       seed=seed,
                                                                       split=split,
                                                                       num_workers=2,
                                                                       pin_memory=True)

    wandb.init(project="LSTM", name=hparams['net_name'],
               config=hparams, save_code=True, resume=True, id=hparams['net_name'])
    logger = WandbLogger()
    run_id = logger.experiment.id
    # Try loading a model first
    try:
        model_dir = f"lightning_logs/{hparams['net_name']}/checkpoints"
        model_loc = glob.glob(f'{model_dir}/*val*.ckpt')[0]
        print(f'Loading {model_loc}')
        with open(f"lightning_logs/{hparams['net_name']}/hparam_config.json", 'r') as f:
            json_string = f.read()

        hparams = json.loads(json_string)

        hparams['seed'] = 189031465
        np.random.seed(hparams['seed'])
        model = ConvLSTMModel.load_from_checkpoint(model_loc,
                                           input_channels=hparams['input_channels'],
                                           hidden_channels=hparams['hidden_channels'],
                                           kernel_size=hparams['kernel_size'],
                                           lr=hparams['learning_rate'],
                                           num_layers=hparams['num_layers'],
                                           )

    except IndexError:
        # Instantiate the model
        print('Instantiating a new model...')
        model = ConvLSTMModel(#net_name=hparams['net_name'],
                    input_channels=hparams['input_channels'],
                    hidden_channels=hparams['hidden_channels'],
                    kernel_size=hparams['kernel_size'],
                      lr=hparams['learning_rate'],
                      num_layers=hparams['num_layers'],)
        log_path = Path(f"./lightning_logs/{hparams['net_name']}")
        log_path.mkdir(parents=True, exist_ok=True)
        with open(log_path / "hparam_config.json", 'w') as f:
            json.dump(hparams, f)

    # Add some checkpointing callbacks
    cbs = [ModelCheckpoint(monitor="loss", filename="{epoch:02d}-{loss:.2f}",
                           dirpath=logger.log_dir,
                           save_top_k=1,
                           mode="min"),
           ModelCheckpoint(monitor="val_loss", filename="{epoch:02d}-{val_loss:.2f}",
                           dirpath=logger.log_dir,
                           save_top_k=1,
                           mode="min"),
           EarlyStopping(monitor="val_loss", check_finite=False, patience=hparams['patience'])]

    trainer = pl.Trainer(
        #strategy='ddp_find_unused_parameters_true',
        callbacks=cbs,  # Add the checkpoint callback
        max_epochs=hparams['epochs'],
        check_val_every_n_epoch=hparams['val_interval'],
        log_every_n_steps=n_samples * split[0],
        logger=logger
    )

    trainer.fit(model, train_dataloader, val_dataloader)

    wandb.finish()
