import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm

from direction_cells import DirectionCells
from models import LMN_PaS_Conj_mEC
from place_cells import PlaceCells
from scores import GridScorer
from trainer import Trainer
from trajectory_generator import TrajectoryGenerator
from utils import generate_run_ID
from visualize import compute_ratemaps_lmn_pas_conj_mec

# %load_ext autoreload
# %autoreload 2


# ============================================================
# Training options and hyperparameters
# ============================================================
class Options:
    pass


options = Options()

options.save_dir = "models_4layer/"
options.n_steps = 1000         # number of training steps
options.epochs = 100           # number of training epochs
options.batch_size = 500       # number of trajectories per batch
options.sequence_length = 20   # number of steps in trajectory
options.learning_rate = 1e-4   # gradient descent learning rate

# early stopping
options.freeze_pas_patience = 100
options.freeze_pas_min_delta = 1e-9  # 0.0001

options.Nd = 128               # number of direction cells
options.Np = 512               # number of place cells
options.Nlmn = 256             # number of LMN cells
options.Npas = 256             # number of PaS cells
options.Ncg = 512              # number of conjuntive mEC cells
options.Ng = 512               # number of mEC cells

options.place_cell_rf = 0.12   # width of place cell center tuning curve (m)
options.surround_scale = 2     # if DoG, ratio of sigma2^2 to sigma1^2
options.direction_cell_rf = 30 * np.pi / 180  # width of direction cell tuning curve (radians)
options.activation = "relu"    # recurrent nonlinearity
options.weight_decay_lmn = 1e-4
options.weight_decay_pas = 1e-4
options.weight_decay_conj = 0
options.weight_decay_mec = 0
options.DoG = True
options.periodic = False
options.box_width = 2.2
options.box_height = 2.2

options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
options.run_ID = generate_run_ID(options)


# ============================================================
# Model setup
# ============================================================
place_cells = PlaceCells(options)
direction_cells = DirectionCells(options)

model = LMN_PaS_Conj_mEC(options, direction_cells, place_cells).to(options.device)

trajectory_generator = TrajectoryGenerator(
    options,
    direction_cells=direction_cells,
    place_cells=place_cells,
)

ckpt_path = os.path.join(options.save_dir, options.run_ID, "most_recent_model.pth")

if os.path.exists(ckpt_path):
    trainer = Trainer(options, model, trajectory_generator, restore=True)
    print(f"Checkpoint exists, skip training: {ckpt_path}")
else:
    trainer = Trainer(options, model, trajectory_generator, restore=False)
    trainer.train(n_epochs=options.epochs, n_steps=options.n_steps, save=True)


# ============================================================
# Plot for Each Fraction
# PaS -> Conj perturbation study in LMN-PaS-Conj-mEC model:
# knock out different fractions of PaS outputs to Conj
# x = knockout fraction (%)
# y = mean grid score of non-silent Conj and mEC cells
# additionally: save grid tuning maps for each iteration
# ============================================================

# ------------------------------------------------------------
# settings
# ------------------------------------------------------------
options.sequence_length = 100
options.n_steps=10
options.batch_size = 10

res = 60
n_avg = 100
thr = 1e-8

# fractions = np.arange(1, 1.0001, 0.1)   # 0%, 10%, ..., 100%
fractions = np.arange(0, .0001, 0.1)   # 0%, 10%, ..., 100%
n_repeats = 1
knockout_seed = 123

# where to save plots
save_root = "Figures_KnockOut_PaS_to_Conj_5"
os.makedirs(save_root, exist_ok=True)

summary_fig_path = os.path.join(save_root, "summary_grid_score_vs_pas_to_conj_knockout.png")


# ------------------------------------------------------------
# helper: compute grid scores for CURRENT model weights
# and also return Conj / mEC activations and keep masks
# ------------------------------------------------------------
def compute_grid_scores_current_model(
    model,
    trajectory_generator,
    options,
    res=60,
    n_avg=10,
    thr=1e-8,
):
    results = compute_ratemaps_lmn_pas_conj_mec(
        model,
        trajectory_generator,
        options,
        res=res,
        n_avg=n_avg,
        N_lmn=options.Nlmn,
        N_pas=options.Npas,
        N_conj=options.Ncg,
        N_mec=options.Ng,
    )

    (
        activations_lmn_dir,
        rate_map_lmn_dir,
        activations_pas_dir,
        rate_map_pas_dir,
        activations_conj_dir,
        rate_map_conj_dir,
        activations_mec_dir,
        rate_map_mec_dir,
        activations_lmn_spatial,
        rate_map_lmn_spatial,
        activations_pas_spatial,
        rate_map_pas_spatial,
        activations_conj_spatial,
        rate_map_conj_spatial,
        activations_mec_spatial,
        rate_map_mec_spatial,
        lmn,
        pas,
        conj,
        mec,
        ang,
        pos,
    ) = results

    # keep non-silent Conj cells
    keep_spatial_conj = np.max(np.abs(activations_conj_spatial), axis=(1, 2)) > thr
    keep_trace_conj = np.max(np.abs(conj), axis=0) > thr
    keep_conj = keep_spatial_conj & keep_trace_conj
    rm_conj_ns = rate_map_conj_spatial[keep_conj]

    # keep non-silent mEC cells
    keep_spatial_mec = np.max(np.abs(activations_mec_spatial), axis=(1, 2)) > thr
    keep_trace_mec = np.max(np.abs(mec), axis=0) > thr
    keep_mec = keep_spatial_mec & keep_trace_mec
    rm_mec_ns = rate_map_mec_spatial[keep_mec]

    res_pos = activations_mec_spatial.shape[-1]
    starts = [0.2] * 10
    ends = np.linspace(0.4, 1.0, num=10)

    coord_range = (
        (-options.box_width / 2, options.box_width / 2),
        (-options.box_height / 2, options.box_height / 2),
    )

    scorer = GridScorer(res_pos, coord_range, zip(starts, ends.tolist()))

    if rm_conj_ns.shape[0] > 0:
        (
            score_60_conj,
            score_90_conj,
            max_60_mask_conj,
            max_90_mask_conj,
            sac_conj,
            max_60_ind_conj,
        ) = zip(*[scorer.get_scores(rm.reshape(res_pos, res_pos)) for rm in rm_conj_ns])
        grid_score_conj = np.asarray(score_60_conj)
    else:
        grid_score_conj = np.array([])

    if rm_mec_ns.shape[0] > 0:
        (
            score_60_mec,
            score_90_mec,
            max_60_mask_mec,
            max_90_mask_mec,
            sac_mec,
            max_60_ind_mec,
        ) = zip(*[scorer.get_scores(rm.reshape(res_pos, res_pos)) for rm in rm_mec_ns])
        grid_score_mec = np.asarray(score_60_mec)
    else:
        grid_score_mec = np.array([])

    return (
        grid_score_conj,
        grid_score_mec,
        keep_conj,
        keep_mec,
        activations_conj_spatial,
        activations_mec_spatial,
        conj,
        mec,
        ang,
        pos,
    )


# ------------------------------------------------------------
# helper: save only grid tuning maps
# ------------------------------------------------------------
def save_grid_tuning_maps_only(
    activations_spatial,
    keep,
    save_path,
    title=None,
    max_cells=None,
    cells_per_row=8,
    cmap="jet",
    empty_text="No non-silent cells",
):
    keep_ids = np.where(keep)[0]

    if max_cells is not None:
        keep_ids = keep_ids[:max_cells]

    n_cells = len(keep_ids)

    if n_cells == 0:
        fig = plt.figure(figsize=(4, 3), dpi=300)
        plt.text(0.5, 0.5, empty_text, ha="center", va="center", fontsize=12)
        plt.axis("off")
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    n_cols = min(cells_per_row, n_cells)
    n_rows = int(math.ceil(n_cells / n_cols))

    fig_w = 1.8 * n_cols
    fig_h = 1.8 * n_rows + 0.6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=300)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[None, :]
    elif n_cols == 1:
        axes = axes[:, None]

    axes_flat = axes.ravel()

    for ax, cell_id in zip(axes_flat, keep_ids):
        gmap = activations_spatial[cell_id]
        ax.imshow(gmap, cmap=cmap, origin="lower")
        ax.set_title(f"{cell_id}", fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes_flat[n_cells:]:
        ax.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=10, y=0.995)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# save original PaS -> Conj weight matrix
# PasSpeedconj takes [speed, pas_1, pas_2, ..., pas_Npas]
# so column 0 = speed, columns 1: = PaS cells
# ------------------------------------------------------------
orig_w = model.PasSpeedconj.weight.detach().clone()

rng = np.random.default_rng(knockout_seed)

all_mean_scores_conj = np.full((len(fractions), n_repeats), np.nan)
all_median_scores_conj = np.full((len(fractions), n_repeats), np.nan)
all_n_kept_conj = np.zeros((len(fractions), n_repeats), dtype=int)

all_mean_scores_mec = np.full((len(fractions), n_repeats), np.nan)
all_median_scores_mec = np.full((len(fractions), n_repeats), np.nan)
all_n_kept_mec = np.zeros((len(fractions), n_repeats), dtype=int)

try:
    for i, frac in enumerate(tqdm(fractions, desc="Knockout fractions")):
        n_ko = int(round(frac * options.Npas))
        frac_percent = int(round(frac * 100))

        frac_dir = save_root
        os.makedirs(frac_dir, exist_ok=True)

        for r in range(n_repeats):
            # restore original weights first
            with torch.no_grad():
                model.PasSpeedconj.weight.copy_(orig_w)

                # Zero selected PaS -> Conj synapses. When all PaS inputs are
                # knocked out, also remove the speed input (column 0) so the
                # third layer receives no external drive through PasSpeedconj.
                if n_ko > 0:
                    ko_idx = rng.choice(options.Npas, size=n_ko, replace=False)
                    model.PasSpeedconj.weight[:, 1 + ko_idx] = 0.0
                    if n_ko >= options.Npas:
                        model.PasSpeedconj.weight[:, 0] = 0.0
                else:
                    ko_idx = np.array([], dtype=int)

            # evaluate grid scores with this perturbation
            (
                grid_score_conj,
                grid_score_mec,
                keep_conj,
                keep_mec,
                activations_conj,
                activations_mec,
                conj,
                mec,
                ang,
                pos,
            ) = compute_grid_scores_current_model(
                model,
                trajectory_generator,
                options,
                res=res,
                n_avg=n_avg,
                thr=thr,
            )

            all_n_kept_conj[i, r] = int(np.sum(keep_conj))
            all_n_kept_mec[i, r] = int(np.sum(keep_mec))

            if grid_score_conj.size > 0:
                all_mean_scores_conj[i, r] = np.nanmean(grid_score_conj)
                all_median_scores_conj[i, r] = np.nanmedian(grid_score_conj)

            if grid_score_mec.size > 0:
                all_mean_scores_mec[i, r] = np.nanmean(grid_score_mec)
                all_median_scores_mec[i, r] = np.nanmedian(grid_score_mec)

            # ------------------------------------------------

            # save per-iteration data

            # ------------------------------------------------

            iter_data_path = os.path.join(
                frac_dir,
                f"iter_data_frac_{frac_percent:03d}_repeat_{r+1:02d}.npz"
            )

            np.savez(
                iter_data_path,
                frac=frac,
                frac_percent=frac_percent,
                repeat_index=r,
                repeat_number=r + 1,
                n_ko=n_ko,
                ko_idx=ko_idx,
                grid_score_conj=grid_score_conj,
                grid_score_mec=grid_score_mec,
                keep_conj=keep_conj,
                keep_mec=keep_mec,

            )


            #------------------------------------------------
            #save grid tuning map plot for this iteration
            #only grid tuning, no directional polar plots
            #------------------------------------------------
            if r != 0:
                continue

            iter_save_path_conj = os.path.join(
                frac_dir,
                f"conj_grid_tuning_frac_{frac_percent:03d}_repeat_{r+1:02d}.png",
            )
            iter_save_path_mec = os.path.join(
                frac_dir,
                f"mec_grid_tuning_frac_{frac_percent:03d}_repeat_{r+1:02d}.png",
            )

            iter_title_conj = (
                f"PaS→Conj knockout = {frac_percent}% | repeat = {r+1} | "
                f"n_ko = {n_ko} | non-silent Conj = {int(np.sum(keep_conj))}"
            )
            iter_title_mec = (
                f"PaS→Conj knockout = {frac_percent}% | repeat = {r+1} | "
                f"n_ko = {n_ko} | non-silent mEC = {int(np.sum(keep_mec))}"
            )

            save_grid_tuning_maps_only(
                activations_spatial=activations_conj,
                keep=keep_conj,
                save_path=iter_save_path_conj,
                title=iter_title_conj,
                max_cells=None,
                cells_per_row=8,
                cmap="jet",
                empty_text="No non-silent Conj cells",
            )

            save_grid_tuning_maps_only(
                activations_spatial=activations_mec,
                keep=keep_mec,
                save_path=iter_save_path_mec,
                title=iter_title_mec,
                max_cells=None,
                cells_per_row=8,
                cmap="jet",
                empty_text="No non-silent mEC cells",
            )

finally:
    # always restore original weights
    with torch.no_grad():
        model.PasSpeedconj.weight.copy_(orig_w)


# ------------------------------------------------------------
# summarize across repeats
# ------------------------------------------------------------
x_percent = fractions * 100.0

mean_curve_conj = np.nanmean(all_mean_scores_conj, axis=1)
valid_counts_conj = np.sum(~np.isnan(all_mean_scores_conj), axis=1)

sem_curve_conj = np.full_like(mean_curve_conj, np.nan, dtype=float)
ok_conj = valid_counts_conj > 1
sem_curve_conj[ok_conj] = (
    np.nanstd(all_mean_scores_conj[ok_conj], axis=1, ddof=1)
    / np.sqrt(valid_counts_conj[ok_conj])
)

median_curve_conj = np.nanmean(all_median_scores_conj, axis=1)
mean_n_kept_conj = np.mean(all_n_kept_conj, axis=1)

mean_curve_mec = np.nanmean(all_mean_scores_mec, axis=1)
valid_counts_mec = np.sum(~np.isnan(all_mean_scores_mec), axis=1)

sem_curve_mec = np.full_like(mean_curve_mec, np.nan, dtype=float)
ok_mec = valid_counts_mec > 1
sem_curve_mec[ok_mec] = (
    np.nanstd(all_mean_scores_mec[ok_mec], axis=1, ddof=1)
    / np.sqrt(valid_counts_mec[ok_mec])
)

median_curve_mec = np.nanmean(all_median_scores_mec, axis=1)
mean_n_kept_mec = np.mean(all_n_kept_mec, axis=1)


# ------------------------------------------------------------
# plot summary curve
# ------------------------------------------------------------
plt.figure(figsize=(7, 5), dpi=300)
plt.plot(x_percent, mean_curve_conj, marker="o", linewidth=2, label="Conj mean grid score")
plt.fill_between(
    x_percent,
    mean_curve_conj - sem_curve_conj,
    mean_curve_conj + sem_curve_conj,
    alpha=0.20,
)

plt.plot(x_percent, mean_curve_mec, marker="o", linewidth=2, label="mEC mean grid score")
plt.fill_between(
    x_percent,
    mean_curve_mec - sem_curve_mec,
    mean_curve_mec + sem_curve_mec,
    alpha=0.20,
)

plt.xlabel("PaS knockout fraction (%)")
plt.ylabel("Grid score")
plt.title("Effect of PaS→Conj knockout on Conj and mEC grid score")
plt.xticks(np.arange(0, 101, 10))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
#plt.savefig(summary_fig_path, dpi=300, bbox_inches="tight")
plt.show()


# ------------------------------------------------------------
# print numeric results
# ------------------------------------------------------------
print("Conj: Knockout % | Mean grid score | SEM | Mean non-silent Conj cells")
for xp, m, s, nk in zip(x_percent, mean_curve_conj, sem_curve_conj, mean_n_kept_conj):
    print(f"{xp:9.0f} | {m:15.6f} | {s:7.6f} | {nk:8.1f}")

print("\nmEC: Knockout % | Mean grid score | SEM | Mean non-silent mEC cells")
for xp, m, s, nk in zip(x_percent, mean_curve_mec, sem_curve_mec, mean_n_kept_mec):
    print(f"{xp:9.0f} | {m:15.6f} | {s:7.6f} | {nk:8.1f}")

print(f"\nSaved per-iteration grid tuning plots to: {save_root}")
print(f"Saved summary figure to: {summary_fig_path}")
