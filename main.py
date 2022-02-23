import argparse
import logging
import sys
import os
from configparser import ConfigParser
from timeit import default_timer
from tqdm import trange
from collections import defaultdict
import numpy as np
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import logging
from torch import optim,nn
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from disvae import init_specific_model, Trainer, Evaluator
from disvae.utils.modelIO import save_model, load_model, load_metadata
from disvae.models.losses import LOSSES, RECON_DIST, get_loss_f
from disvae.models.vae import MODELS
from utils.datasets import get_dataloaders, get_img_size, DATASETS, get_train_dataloaders, get_test_dataloaders
from utils.datasets import get_train_datasets, get_test_datasets, get_datasets
from utils.helpers import (create_safe_directory, get_device, set_seed, get_n_param,
                           get_config_section, update_namespace_, FormatterNoDuplicate)
from utils.visualize import GifTraversalsTraining

TRAIN_LOSSES_LOGFILE = "train_losses.log"
CONFIG_FILE = "hyperparam.ini"
RES_DIR = "results"
LOG_LEVELS = list(logging._levelToName.values())
ADDITIONAL_EXP = ['custom', "debug", "best_celeba"]
EXPERIMENTS = ADDITIONAL_EXP + ["{}_{}".format(loss, data)
                                for loss in LOSSES
                                for data in DATASETS]


def parse_arguments(args_to_parse):
    """Parse the command line arguments.

    Parameters
    ----------
    args_to_parse: list of str
        Arguments to parse (splitted on whitespaces).
    """
    default_config = get_config_section([CONFIG_FILE], "Custom")

    description = "PyTorch implementation and evaluation of disentangled Variational AutoEncoders and metrics."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=FormatterNoDuplicate)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument('name', type=str,
                         help="Name of the model for storing and loading purposes.")
    general.add_argument('-L', '--log-level', help="Logging levels.",
                         default=default_config['log_level'], choices=LOG_LEVELS)
    general.add_argument('--no-progress-bar', action='store_true',
                         default=default_config['no_progress_bar'],
                         help='Disables progress bar.')
    general.add_argument('--no-cuda', action='store_true',
                         default=default_config['no_cuda'],
                         help='Disables CUDA training, even when have one.')
    general.add_argument('-s', '--seed', type=int, default=default_config['seed'],
                         help='Random seed. Can be `None` for stochastic behavior.')

    # Learning options
    training = parser.add_argument_group('Training specific options')
    training.add_argument('--checkpoint-every',
                          type=int, default=default_config['checkpoint_every'],
                          help='Save a checkpoint of the trained model every n epoch.')
    training.add_argument('-d', '--dataset', help="Path to training data.",
                          default=default_config['dataset'])
    training.add_argument('-x', '--experiment',
                          default=default_config['experiment'], choices=EXPERIMENTS,
                          help='Predefined experiments to run. If not `custom` this will overwrite some other arguments.')
    training.add_argument('-e', '--epochs', type=int,
                          default=default_config['epochs'],
                          help='Maximum number of epochs to run for.')
    training.add_argument('-b', '--batch-size', type=int,
                          default=default_config['batch_size'],
                          help='Batch size for training.')
    training.add_argument('--lr', type=float, default=default_config['lr'],
                          help='Learning rate.')
    training.add_argument('--image_size', type=int, default=default_config['image_size'],
                          help='the size of the image, it is squared, so it would be 32,64,128.')
    training.add_argument('--start_checkpoint', type=int, default=default_config['s_chk'],
                          help='Which epoch would you like to choose. (-1) - start a new model, 0<= - reload another checkpoint')

    # Model Options
    model = parser.add_argument_group('Model specfic options')
    model.add_argument('-m', '--model-type',
                       default=default_config['model'], choices=MODELS,
                       help='Type of encoder and decoder to use.')
    model.add_argument('-z', '--latent-dim', type=int,
                       default=default_config['latent_dim'],
                       help='Dimension of the latent variable.')
    model.add_argument('-l', '--loss',
                       default=default_config['loss'], choices=LOSSES,
                       help="Type of VAE loss function to use.")
    model.add_argument('-r', '--rec-dist', default=default_config['rec_dist'],
                       choices=RECON_DIST,
                       help="Form of the likelihood ot use for each pixel.")
    model.add_argument('-a', '--reg-anneal', type=float,
                       default=default_config['reg_anneal'],
                       help="Number of annealing steps where gradually adding the regularisation. What is annealed is specific to each loss.")

    # Loss Specific Options
    betaH = parser.add_argument_group('BetaH specific parameters')
    betaH.add_argument('--betaH-B', type=float,
                       default=default_config['betaH_B'],
                       help="Weight of the KL (beta in the paper).")

    betaB = parser.add_argument_group('BetaB specific parameters')
    betaB.add_argument('--betaB-initC', type=float,
                       default=default_config['betaB_initC'],
                       help="Starting annealed capacity.")
    betaB.add_argument('--betaB-finC', type=float,
                       default=default_config['betaB_finC'],
                       help="Final annealed capacity.")
    betaB.add_argument('--betaB-G', type=float,
                       default=default_config['betaB_G'],
                       help="Weight of the KL divergence term (gamma in the paper).")

    factor = parser.add_argument_group('factor VAE specific parameters')
    factor.add_argument('--factor-G', type=float,
                        default=default_config['factor_G'],
                        help="Weight of the TC term (gamma in the paper).")
    factor.add_argument('--lr-disc', type=float,
                        default=default_config['lr_disc'],
                        help='Learning rate of the discriminator.')

    btcvae = parser.add_argument_group('beta-tcvae specific parameters')
    btcvae.add_argument('--btcvae-A', type=float,
                        default=default_config['btcvae_A'],
                        help="Weight of the MI term (alpha in the paper).")
    btcvae.add_argument('--btcvae-G', type=float,
                        default=default_config['btcvae_G'],
                        help="Weight of the dim-wise KL term (gamma in the paper).")
    btcvae.add_argument('--btcvae-B', type=float,
                        default=default_config['btcvae_B'],
                        help="Weight of the TC term (beta in the paper).")

    # Learning options
    evaluation = parser.add_argument_group('Evaluation specific options')
    evaluation.add_argument('--is-eval-only', action='store_true',
                            default=default_config['is_eval_only'],
                            help='Whether to only evaluate using precomputed model `name`.')
    evaluation.add_argument('--is-metrics', action='store_true',
                            default=default_config['is_metrics'],
                            help="Whether to compute the disentangled metrcics. Currently only possible with `dsprites` as it is the only dataset with known true factors of variations.")
    evaluation.add_argument('--no-test', action='store_true',
                            default=default_config['no_test'],
                            help="Whether not to compute the test losses.`")
    evaluation.add_argument('--eval-batchsize', type=int,
                            default=default_config['eval_batchsize'],
                            help='Batch size for evaluation.')

    args = parser.parse_args(args_to_parse)
    if args.experiment != 'custom':
        if args.experiment not in ADDITIONAL_EXP:
            # update all common sections first
            model, dataset = args.experiment.split("_")
            common_data = get_config_section([CONFIG_FILE], "Common_{}".format(dataset))
            update_namespace_(args, common_data)
            common_model = get_config_section([CONFIG_FILE], "Common_{}".format(model))
            update_namespace_(args, common_model)

        try:
            experiments_config = get_config_section([CONFIG_FILE], args.experiment)
            update_namespace_(args, experiments_config)
        except KeyError as e:
            if args.experiment in ADDITIONAL_EXP:
                raise e  # only reraise if didn't use common section

    return args

def train(config, checkpoint_dir=None, args=None):
    # if args.loss == "factor":
    #         logger.info("FactorVae needs 2 batches per iteration. To replicate this behavior while being consistent, we double the batch size and the the number of epochs.")
    #         args.batch_size *= config['batch_size']
    #         args.epochs *= 2
    model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
    device = "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    # PREPARES DATA
    train_set= get_train_datasets(args.dataset)
    test_abs = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(
    train_set, [test_abs, len(train_set) - test_abs])
    train_loader = DataLoader(
        train_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=8)
    val_loader = DataLoader(
        val_subset,
        batch_size=int(config['batch_size']),
        shuffle=True,
        num_workers=8)
    # logger.info("Train {} with {} samples".format(args.dataset, len(train_loader)))

    # PREPARES MODEL 
    model.to(device)     
    if args.start_checkpoint > -1:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    # logger.info('Num parameters in model: {}'.format(get_n_param(model)))

    # TRAINS
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss_f = get_loss_f(args.loss,config=config,
                        n_data=len(train_loader),
                        device=device,
                        **vars(args))
    start = default_timer()
    model.train()
    for epoch in range(args.epochs):
        storer = defaultdict(list)
        epoch_loss = 0.
        kwargs = dict(desc="Epoch {}".format(epoch + 1), leave=False,
                    disable=args.no_progress_bar)
        with trange(len(train_loader), **kwargs) as t:
            # for _, (data, _) in enumerate(data_loader):
            for _, data in enumerate(train_loader):
                data = data.to(device)
                try:
                    recon_batch, latent_dist, latent_sample = model(data)
                    loss = loss_f(data, recon_batch, latent_dist, model.training,
                                    storer, latent_sample=latent_sample)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except ValueError:
                    # for losses that use multiple optimizers (e.g. Factor)
                    loss = loss_f.call_optimize(data, model, optimizer, storer)

                iter_loss = loss.item()
                epoch_loss += iter_loss

                t.set_postfix(loss=iter_loss)
                t.update()

        mean_epoch_loss = epoch_loss / len(train_loader)
        # mean_epoch_loss = self._train_epoch(data_loader, storer, epoch)
        # logger.info('Epoch: {} Average loss per image: {:.2f}'.format(epoch + 1,mean_epoch_loss))
        # losses_logger.log(epoch, storer)

        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                data = data.to(device)
                try:
                    recon_batch, latent_dist, latent_sample = model(data)
                    loss = loss_f(data, recon_batch, latent_dist, model.training,
                                    storer, latent_sample=latent_sample)
                    total += data.size(0)
                    if loss <10:
                        correct 

                    val_loss += loss.cpu().numpy()
                    val_steps += 1
                except ValueError:
                    # for losses that use multiple optimizers (e.g. Factor)
                    loss = loss_f.call_optimize(data, model, optimizer, storer)

        if epoch % args.checkpoint_every == 0:
            # save_model(model, exp_dir,filename="model-{}.pt".format(epoch), epoch=epoch, tune=tune)
            with tune.checkpoint_dir(epoch) as directory:
                path_to_model = os.path.join(directory, "checkpoint")
                torch.save(model.state_dict(), path_to_model)
        tune.report(loss=(val_loss / val_steps), accuracy = val_loss/total)

        model.eval()

        delta_time = (default_timer() - start) / 60
        # logger.info('Finished training after {:.1f} min.'.format(delta_time))

    # SAVE MODEL AND EXPERIMENT INFORMATION
    # save_model(model, exp_dir, metadata=vars(args),tune=tune)
    # with tune.checkpoint_dir(epoch) as directory:
    #     path_to_model = os.path.join(directory, "model.pt")
    #     torch.save(model.state_dict(), path_to_model)

def test(args, model,device,logger=None, config=None):
    print("Evaluation")
    # model = init_specific_model(args.model_type, args.img_size, args.latent_dim).to(device)
    # model.to(device)
    # if args.start_checkpoint > -1:
    #     model_state, optimizer_state = torch.load(
    #         os.path.join(checkpoint_dir, "checkpoint"))
    #     model.load_state_dict(model_state)
    test_set = get_test_datasets(args.dataset)
    test_loader = DataLoader(dataset=test_set, batch_size=args.eval_batchsize,shuffle=False)
    loss_f = get_loss_f(args.loss,config=config,
                        n_data=len(test_loader),
                        device=device,
                        **vars(args))
    evaluator = Evaluator(model, loss_f,
                            device=device,
                            logger=logger,
                            is_progress_bar=not args.no_progress_bar)

    metric,losses = evaluator(test_loader, is_metrics=args.is_metrics, is_losses=not args.no_test)
    return losses

def main(args):
    """Main train and evaluation function.

    Parameters
    ----------
    args: argparse.Namespace
        Arguments
    """
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    set_seed(args.seed)
    # device = get_device(is_gpu=not args.no_cuda)
    exp_dir = os.path.join(RES_DIR, args.name)
    logger.info("Root directory for saving and loading experiments: {}".format(exp_dir))
    device = "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        device = "cuda:0"
        # if torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    config = {
        'batch_size': tune.choice([2, 4, 8, 16,128]),
        'betaB_finC': tune.sample_from(lambda _: np.random.randint(1, 100)),
        'betaB_G': tune.choice([50,100,150,1000]),
        'lr': tune.loguniform(1e-4, 1e-3)
    }
    args.img_size = (3,args.image_size,args.image_size)
    # print(config["batch_size"])
    # if not args.is_eval_only:
    create_safe_directory(exp_dir, logger=logger)
    gpus_per_trial = 2
    num_samples = 10
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss","accuracy", "training_iteration"])
    result = tune.run(
        partial(train,checkpoint_dir=exp_dir,args=args),
        resources_per_trial={"cpu": 8, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        # checkpoint_at_end=True
        )       

    # if args.is_metrics or not args.no_test:
    # test(args, exp_dir, logger)
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = init_specific_model(args.model_type, args.img_size, args.latent_dim)
    # Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        # if gpus_per_trial > 1:
        #     best_trained_model = nn.DataParallel(best_trained_model)
    

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    best_trained_model.to(device)

    test_acc = test(args=args,model=best_trained_model,device=device,logger=logger, config=best_trial.config)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
