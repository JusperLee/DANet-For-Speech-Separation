import sys
sys.path.append('./')
from torch.utils.data import DataLoader as Loader
from data_loader import Dataloader, AudioData
from model import model
from logger import set_logger
import logging
from config import option
import argparse
import torch
from trainer import Trainer



def make_dataloader(opt):
    # make train's dataloader
    train_mix_reader = AudioData(
        opt['datasets']['train']['dataroot_mix'], **opt['datasets']['audio_setting'])
    train_target_readers = [AudioData(opt['datasets']['train']['dataroot_targets'][0], **opt['datasets']['audio_setting']),
                            AudioData(opt['datasets']['train']['dataroot_targets'][1], **opt['datasets']['audio_setting'])]
    train_dataset = Dataloader.dataset(
        train_mix_reader, train_target_readers, opt['datasets']['dataloader_setting']['cmvn_file'])
    train_dataloader = Loader(train_dataset,
                              batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                              num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                              shuffle=opt['datasets']['dataloader_setting']['shuffle'])

    # make validation dataloader
    val_mix_reader = AudioData(
        opt['datasets']['val']['dataroot_mix'], **opt['datasets']['audio_setting'])
    val_target_readers = [AudioData(opt['datasets']['val']['dataroot_targets'][0], **opt['datasets']['audio_setting']),
                          AudioData(opt['datasets']['val']['dataroot_targets'][1], **opt['datasets']['audio_setting'])]
    val_dataset = Dataloader.dataset(
        val_mix_reader, val_target_readers, opt['datasets']['dataloader_setting']['cmvn_file'])
    val_dataloader = Loader(val_dataset,
                            batch_size=opt['datasets']['dataloader_setting']['batch_size'],
                            num_workers=opt['datasets']['dataloader_setting']['num_workers'],
                            shuffle=opt['datasets']['dataloader_setting']['shuffle'])
    return train_dataloader, val_dataloader


def make_optimizer(params, opt):
    optimizer = getattr(torch.optim, opt['optim']['name'])
    if opt['optim']['name'] == 'Adam':
        optimizer = optimizer(
            params, lr=opt['optim']['lr'], weight_decay=opt['optim']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=opt['optim']['lr'], weight_decay=opt['optim']
                              ['weight_decay'], momentum=opt['optim']['momentum'])

    return optimizer


def train():
    parser = argparse.ArgumentParser(
        description='Parameters for training DANet')
    parser.add_argument('--opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    opt = option.parse(args.opt)
    set_logger.setup_logger(opt['logger']['name'], opt['logger']['path'],
                            screen=opt['logger']['screen'], tofile=opt['logger']['tofile'])
    logger = logging.getLogger(opt['logger']['name'])

    logger.info("Building the model of DANet")
    danet = model.DANet(**opt['DANet'])

    logger.info("Building the optimizer of DANet")
    optimizer = make_optimizer(danet.parameters(), opt)

    logger.info('Building the dataloader of DANet')
    train_dataloader, val_dataloader = make_dataloader(opt)

    logger.info('Train Datasets Length: {}, Val Datasets Length: {}'.format(
        len(train_dataloader), len(val_dataloader)))
    logger.info('Building the Trainer of DANet')
    trainer = Trainer(train_dataloader, val_dataloader, danet, optimizer, opt)
    trainer.run()


if __name__ == "__main__":
    train()
