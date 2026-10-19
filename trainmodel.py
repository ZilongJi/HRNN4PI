import numpy as np
import torch
import os

from utils import generate_run_ID
from place_cells import PlaceCells
from direction_cells import DirectionCells
from trajectory_generator import TrajectoryGenerator
from models import LMN_PaS_Conj_mEC
from trainer import Trainer

# Training options and hyperparameters
class Options:
    pass


def build_options():
    options = Options()

    options.save_dir = 'models_4layer/'
    options.n_steps = 1000    # number of training steps
    options.epochs = 201      # number of training epochs
    options.batch_size = 500  # number of trajectories per batch
    options.sequence_length = 20  # number of steps in trajectory
    options.learning_rate = 1e-4  # gradient descent learning rate
    # options.lr_milestones = [60, 80]
    # options.lr_gamma = 0.3

    # early stopping
    options.freeze_pas_patience = 100
    options.freeze_pas_min_delta = 1e-8

    options.Nd = 128           # number of direction cells
    options.Np = 512           # number of place cells
    options.Nlmn = 256        # number of LMN cells
    options.Npas = 256        # number of PaS cells
    options.Ncg = 512         # number of conjuntive mEC cells
    options.Ng = 512         # number of mEC cells

    options.place_cell_rf = 0.12  # width of place cell center tuning curve (m)
    options.surround_scale = 2    # if DoG, ratio of sigma2^2 to sigma1^2
    options.direction_cell_rf = 30 * np.pi / 180  # width of direction cell tuning curve (radians)
    options.activation = 'relu'   # recurrent nonlinearity
    options.weight_decay_lmn = 1e-4   # strength of weight decay on recurrent weights in lmn
    options.weight_decay_pas = 1e-4   # strength of weight decay on recurrent weights in pas
    options.weight_decay_conj = 0   # strength of weight decay on recurrent weights in conjunctive mEC RNN
    options.weight_decay_mec = 0   # strength of weight decay on recurrent weights in mEC RNN
    options.use_pas_layernorm = False  # apply LayerNorm to PaS before feeding Conj
    options.DoG = True
    options.periodic = False
    options.box_width = 2.2
    options.box_height = 2.2

    options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return options


def train_single_model(ng, weight_decay_mec):
    options = build_options()
    options.Ng = ng
    options.weight_decay_mec = weight_decay_mec
    options.run_ID = generate_run_ID(options)

    print(f'\n===== Start config: Ng={options.Ng}, weight_decay_mec={options.weight_decay_mec} =====')
    print(f'Run ID: {options.run_ID}')

    place_cells = PlaceCells(options)
    direction_cells = DirectionCells(options)
    model = LMN_PaS_Conj_mEC(options, direction_cells, place_cells).to(options.device)

    trajectory_generator = TrajectoryGenerator(
        options,
        direction_cells=direction_cells,
        place_cells=place_cells,
    )

    ckpt_path = os.path.join(options.save_dir, options.run_ID, 'most_recent_model.pth')

    if os.path.exists(ckpt_path):
        Trainer(options, model, trajectory_generator, restore=True)
        print(f'Checkpoint exists, skip training: {ckpt_path}')
        return

    trainer = Trainer(options, model, trajectory_generator, restore=False)
    trainer.train(n_epochs=options.epochs, n_steps=options.n_steps, save=True)


if __name__ == '__main__':
    ng_list = [512]
    weight_decay_mec_list = [0] #[0, 1e-7, 1e-6, 1e-5] #[1e-5, 1e-6, 1e-7, 0]

    for ng in ng_list:
        for weight_decay_mec in weight_decay_mec_list:
            print(f'\n===== Training model with Ng={ng}, weight_decay_mec={weight_decay_mec} =====') 
            train_single_model(ng, weight_decay_mec)
