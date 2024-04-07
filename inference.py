import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import argparse
import os
import sys
import subprocess
import setproctitle
import colorama
import numpy as np
from tqdm import tqdm
from glob import glob
from os.path import *

import models
import losses
import datasets
from utils import flow_utils, tools
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int,
                        default=8, help="Batch size")
    # parser.add_argument('--train_n_batches', type=int, default = -1, help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[
                        256, 256], help="Spatial dimension to crop training samples for training")
    # parser.add_argument('--gradient_clip', type=float, default=None)
    # parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
    # parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default=255.)

    parser.add_argument('--number_workers', '-nw',
                        '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int,
                        default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str,
                        help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work',
                        type=str, help='directory for saving')
    parser.add_argument('--render_validation', action='store_true',
                        help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+',
                        default=[-1, -1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true',
                        help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter',
                        type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    tools.add_arguments_for_module(
        parser, models, argument_for_class='model', default='FlowNet2')

    tools.add_arguments_for_module(
        parser, losses, argument_for_class='loss', default='L1Loss')

    tools.add_arguments_for_module(
        parser, torch.optim, argument_for_class='optimizer', default='Adam', skip_params=['params'])

    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel/flow/training',
                                                       'replicates': 1})

    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)
# Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments


@torch.no_grad()
def main():
    with tools.TimerBlock("Parsing Arguments") as block:
        global args, save_path
        args = parser.parse_args()
        if args.number_gpus < 0:
            args.number_gpus = torch.cuda.device_count()

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE',  action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        # for argument, value in sorted(vars(args).items()):
        #    reset = colorama.Style.RESET_ALL
        #    color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
        #    block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[
            args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.inference_dataset_class = tools.module_to_dict(datasets)[
            args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # args.current_hash = subprocess.check_output(
        #    ["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers,
                   'pin_memory': True,
                   'drop_last': True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers
    print(args)
    inference_dataset = args.inference_dataset_class(
        args, False, **tools.kwargs_from_args(args, 'inference_dataset'))
    block.log('Inference Dataset: {}'.format(args.inference_dataset))
    block.log('Inference Input: {}'.format(
        ' '.join([str([d for d in x.size()]) for x in inference_dataset[0][0]])))
    block.log('Inference Targets: {}'.format(
        ' '.join([str([d for d in x.size()]) for x in inference_dataset[0][1]])))
    inference_loader = DataLoader(
        inference_dataset, batch_size=args.effective_inference_batch_size, shuffle=False, **inf_gpuargs)

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments

    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)

            def forward(self, data, target, inference=False):
                output = self.model(data)

                loss_values = self.loss(output, target)

                return loss_values, output

        model_and_loss = ModelAndLoss(args)

    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)

            def forward(self, data, target, inference=True):
                output = self.model(data)

                loss_values = self.loss(output, target)

                if not inference:
                    return loss_values
                else:
                    return loss_values, output

        model_and_loss = ModelAndLoss(args)

    if args.resume and os.path.isfile(args.resume):
        block.log("Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        best_err = checkpoint['best_EPE']
        model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
        block.log("Loaded checkpoint '{}' (at epoch {})".format(
            args.resume, checkpoint['epoch']))

    def inference(args, epoch, data_loader, model, offset=0):

        model.to(device)
        model.eval()
        cudnn.benchmark = True

        if args.save_flow or args.render_validation:
            flow_folder = "{}/inference/{}.epoch-{}-flow-field".format(
                args.save, args.name.replace('/', '.'), epoch)
            print(f"saved to {flow_folder}")
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)
                print(f"saved to {flow_folder}")

        args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches), desc='Inferencing ',
                        leave=True, position=offset)

        statistics = []
        total_loss = 0
        for batch_idx, (data, target) in enumerate(progress):
            if args.cuda:
                data = [d.cuda(non_blocking=True) for d in data]
                target = [t.cuda(non_blocking=True) for t in target]
            data = [d.to(device) for d in data]
            target = [t.to(device) for t in target]
            # when ground-truth flows are not available for inference_dataset,
            # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows,
            # depending on the type of loss norm passed in
            losses, output = model(data[0], target[0], inference=True)

            losses = [torch.mean(loss_value) for loss_value in losses]
            loss_val = losses[0]  # Collect first loss for weight update
            total_loss += loss_val.item()
            loss_values = [v.item() for v in losses]

            # gather loss_labels, direct return leads to recursion limit error as it looks for variables to gather'
            loss_labels = list(model.module.loss.loss_labels)

            statistics.append(loss_values)
            # import IPython; IPython.embed()
            if args.save_flow or args.render_validation:
                for i in range(args.inference_batch_size):
                    _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                    flow_utils.writeFlow(join(flow_folder, '%06d.flo' % (
                        batch_idx * args.inference_batch_size + i)),  _pflow)

            progress.set_description('Inference Averages for Epoch {}: '.format(
                epoch) + tools.format_dictionary_of_losses(loss_labels, np.array(statistics).mean(axis=0)))
            progress.update(1)

            if batch_idx == (args.inference_n_batches - 1):
                break
        print("\n")
        print("done 1\n")

        progress.close()

        return

    best_err = 1e8
    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)),
                    miniters=1, ncols=100, desc='Overall Progress', leave=True, position=0)
    offset = 1
    last_epoch_time = progress._time()
    global_iteration = 0

    for epoch in progress:
        if args.inference or (args.render_validation and ((epoch - 1) % args.validation_frequency) == 0):
            stats = inference(args=args, epoch=epoch - 1,
                              data_loader=inference_loader, model=model_and_loss, offset=offset)
            offset += 1

        last_epoch_time = progress._time()
    print("\n")
    print("done 1\n")


if __name__ == "__main__":
    main()
