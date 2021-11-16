import gc
import logging
import os
import sys
import warnings
from datetime import datetime
from statistics import mean, median, stdev
from typing import List, Tuple

from easydict import EasyDict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchinfo import summary
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, FashionMNIST
from tqdm import tqdm

from datasets import (
    CIFAR10DataModule,
    CIFAR100DataModule,
    EMNISTDataModule,
    FashionMNISTDataModule,
    MNISTDataModule,
)
from model import MLP, ResNet, ResNeXt, SimpleCNN, WideResNet
from utils import format_output

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

dataset_fns = {
    'mnist': MNIST,
    'fashion-mnist': FashionMNIST,
    'emnist': EMNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
}

data_modules = {
    'mnist': MNISTDataModule,
    'fashion-mnist': FashionMNISTDataModule,
    'emnist': EMNISTDataModule,
    'cifar10': CIFAR10DataModule,
    'cifar100': CIFAR100DataModule,
}


def train(args, train_size: int) -> Tuple[List, List, List]:
    adv_config = EasyDict(
        adv_train_mode=args.adv_train_mode,
        adv_test_mode=args.adv_test_mode,
        attack_type=args.attack_type,
        eps=args.eps,
        eps_iter=args.eps_iter,
        nb_iter=args.nb_iter,
    )
    train_acc_log, val_acc_log, test_acc_log = [], [], []
    if 'emnist' in args.dataset:
        datamodule = EMNISTDataModule(
            EMNIST,
            args.dataset.split('-')[1],
            args.batch_size,
            args.num_workers,
        )
    else:
        datamodule = data_modules[args.dataset](
            dataset_fns[args.dataset], args.batch_size, args.num_workers
        )

    datamodule.download_data()
    with tqdm(
        desc=f'train_size = {train_size}',
        total=args.num_trials,
        file=sys.stdout,
    ) as pbar:
        for _ in range(args.num_trials):
            datamodule.sample_data(train_size, args.seed)
            train_dataloader = datamodule.train_dataloader()
            val_dataloader = datamodule.val_dataloader()
            test_dataloader = datamodule.test_dataloader()

            assert args.model_type in [
                'mlp-small',
                'mlp-medium',
                'mlp-large',
                'cnn-small',
                'cnn-medium',
                'cnn-large',
                'resnet-18',
                'resnet-34',
                'resnet-50',
                'resnext-50',
                'wideresnet-50',
            ]
            if 'cnn' in args.model_type:
                model = SimpleCNN(
                    args.in_channels,
                    args.height,
                    args.width,
                    args.output_dim,
                    args.model_type.split('-')[1],
                    adv_config,
                )
            elif 'mlp' in args.model_type:
                model = MLP(
                    args.height,
                    args.width,
                    args.in_channels,
                    args.output_dim,
                    args.model_type.split('-')[1],
                    adv_config,
                )
            elif 'resnet' in args.model_type:
                model = ResNet(
                    args.in_channels,
                    args.output_dim,
                    int(args.model_type.split('-')[1]),
                    adv_config,
                )
            elif 'resnext' in args.model_type:
                model = ResNeXt(args.in_channels, args.output_dim, adv_config)
            elif 'wide-resnet' in args.model_type:
                model = WideResNet(
                    args.in_channels, args.output_dim, adv_config
                )
            else:
                raise ValueError('Invalid model')
            if args.verbose:
                print(
                    summary(
                        model,
                        input_size=(
                            args.batch_size,
                            args.in_channels,
                            args.height,
                            args.width,
                        ),
                    )
                )

            model_checkpoint_callback = ModelCheckpoint(
                dirpath=args.model_path,
                filename='{epoch}-{avg_val_acc}',
                monitor='avg_val_acc',
                save_top_k=10,
                mode='max',
                every_n_epochs=1,
            )
            early_stopping_callback = EarlyStopping(
                monitor='avg_val_acc',
                patience=args.patience,
                verbose=False,
                mode='max',
            )
            trainer = Trainer(
                gpus=-1,
                progress_bar_refresh_rate=0,
                callbacks=[model_checkpoint_callback, early_stopping_callback],
                max_epochs=args.max_epochs,
                weights_summary=None,
                check_val_every_n_epoch=1,
            )
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            train_acc_log.append(
                model.train_hist[
                    int(
                        model_checkpoint_callback.best_model_path.split('/')[
                            -1
                        ]
                        .split('-')[0]
                        .split('=')[1]
                    )
                ]['avg_train_acc']
            )
            val_acc_log.append(
                model_checkpoint_callback.best_model_score.item()
            )
            trainer.test(
                dataloaders=test_dataloader,
                ckpt_path=model_checkpoint_callback.best_model_path,
                verbose=0,
            )
            test_acc_log.append(model.results['avg_test_acc'])

            del model
            gc.collect()
            os.system(f'rm -rf {args.model_path}')
            pbar.update()
            pbar.refresh()
            if not args.notebook:
                print('', flush=True)

    return (train_acc_log, val_acc_log, test_acc_log)


def run(args) -> List:
    output = []
    for train_size in range(
        args.min_train_size, args.max_train_size, args.step_train_size
    ):
        # Train for one training set size
        train_acc_log, val_acc_log, test_acc_log = train(args, train_size)
        # Calculate mean, median, and std of train, val, and test accuracy
        result = {
            train_size: [
                mean(train_acc_log),
                mean(val_acc_log),
                mean(test_acc_log),
                median(train_acc_log),
                median(val_acc_log),
                median(test_acc_log),
                stdev(train_acc_log),
                stdev(val_acc_log),
                stdev(test_acc_log),
            ]
        }
        # Append result to output
        output.append(result)
        # Get output to be printed
        print_out = format_output(output)
        # Print output
        print(print_out)
        # Get current times stamp
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        # Define filename
        filename = f'{args.dataset}_{args.model_type}_{now}.txt'
        # Create output file
        log = open(filename, 'w+')
        log.write(print_out)
        log.close()
        # Check if target dir exists
        if not os.path.isdir(args.target_dir):
            os.makedirs(args.target_dir)
        # Move file to target dir
        os.system(f'cp {filename} {args.target_dir}')
    return output
