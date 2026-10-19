# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt

import scipy
import scipy.stats
import seaborn as sns
from imageio import imsave
import cv2

from place_cells import PlaceCells
from direction_cells import DirectionCells


def concat_images(images, image_width, spacer_size):
    """ Concat image horizontally with spacer """
    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret


def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """ Concat images in rows """
    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret


def convert_to_colormap(im, cmap):
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):
    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')  # ignore divide by zero err
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    if smooth:
        im = cv2.GaussianBlur(im, (3,3), sigmaX=1, sigmaY=0)
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):
    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


def plot_ratemaps_1d(activations, n_plots, width=8, polar=False):
    n_plots = min(n_plots, activations.shape[0])
    n_cols = min(width, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))
    if polar:
        fig = plt.figure(figsize=(2.2 * n_cols, 2.2 * n_rows))
        angles = np.linspace(0, 2 * np.pi, activations.shape[-1], endpoint=False)
        angles_closed = np.concatenate([angles, angles[:1] + 2 * np.pi])
        for i in range(n_rows * n_cols):
            r, c = divmod(i, n_cols)
            idx = r * n_cols + c
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="polar")
            if idx < n_plots:
                vals = activations[idx]
                vals = np.maximum(vals, 0)
                vals = vals / (np.max(vals) + 1e-8)
                vals_closed = np.concatenate([vals, vals[:1]])
                ax.fill_between(angles_closed, 0, vals_closed, color="#009FB9", alpha=0.35)
                ax.plot(angles_closed, vals_closed, linewidth=1)
                ax.set_ylim(0, 1.0)
            else:
                ax.set_axis_off()
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.6 * n_rows), squeeze=False)
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        if i < n_plots:
            x = np.arange(activations.shape[-1])
            ax.fill_between(x, 0, activations[i], color="#F18D00", alpha=0.35)
            ax.plot(activations[i], linewidth=1)
            ax.set_xlim(0, activations.shape[-1] - 1)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


def plot_ratemaps_1d_cw_ccw(
    activations_cw,
    activations_ccw,
    n_plots=None,
    width=8,
    polar=False,
    order=None,
    cw_title="Clockwise",
    ccw_title="Counter-Clockwise",
):
    '''Plot clockwise and counter-clockwise 1d ratemaps overlaid per cell.'''

    def _circular_center_of_mass(vals):
        vals = np.maximum(vals, 0)
        total = np.sum(vals)
        if total <= 1e-8:
            return None
        angles = np.linspace(0, 2 * np.pi, vals.shape[-1], endpoint=False)
        x = np.sum(vals * np.cos(angles)) / total
        y = np.sum(vals * np.sin(angles)) / total
        return np.arctan2(y, x) % (2 * np.pi)

    if activations_cw.shape[0] != activations_ccw.shape[0]:
        raise ValueError("activations_cw and activations_ccw must have the same number of units.")

    if n_plots is None:
        n_plots = min(activations_cw.shape[0], activations_ccw.shape[0])
    n_plots = min(n_plots, activations_cw.shape[0], activations_ccw.shape[0])

    if order is None:
        combined = np.maximum(activations_cw[:n_plots], 0) + np.maximum(activations_ccw[:n_plots], 0)
        pref_idx = np.argmax(combined, axis=1)
        order = np.argsort(pref_idx)
    else:
        order = np.asarray(order)[:n_plots]

    activations_cw = activations_cw[:n_plots][order]
    activations_ccw = activations_ccw[:n_plots][order]

    n_cols = min(width, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    if polar:
        fig = plt.figure(figsize=(2.2 * n_cols, 2.2 * n_rows))
        angles = np.linspace(0, 2 * np.pi, activations_cw.shape[-1], endpoint=False)
        angles_closed = np.concatenate([angles, angles[:1] + 2 * np.pi])

        for i in range(n_rows * n_cols):
            r, c = divmod(i, n_cols)
            idx = r * n_cols + c
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="polar")

            if i < n_plots:
                vals_cw = np.maximum(activations_cw[i], 0)
                vals_ccw = np.maximum(activations_ccw[i], 0)
                scale = max(np.max(vals_cw), np.max(vals_ccw)) + 1e-8
                com_cw = _circular_center_of_mass(vals_cw)
                com_ccw = _circular_center_of_mass(vals_ccw)
                vals_cw = vals_cw / scale
                vals_ccw = vals_ccw / scale

                vals_cw_closed = np.concatenate([vals_cw, vals_cw[:1]])
                vals_ccw_closed = np.concatenate([vals_ccw, vals_ccw[:1]])
                ax.fill_between(angles_closed, 0, vals_cw_closed, color="#E36414", alpha=0.18)
                ax.plot(angles_closed, vals_cw_closed, linewidth=1.8, color="#9A3412")
                ax.fill_between(angles_closed, 0, vals_ccw_closed, color="#0F8B8D", alpha=0.18)
                ax.plot(angles_closed, vals_ccw_closed, linewidth=1.8, color="#155E75")
                if com_cw is not None:
                    ax.plot([com_cw, com_cw], [0, 1.0], linestyle="--", linewidth=1.2, color="#9A3412", alpha=0.9)
                if com_ccw is not None:
                    ax.plot([com_ccw, com_ccw], [0, 1.0], linestyle="--", linewidth=1.2, color="#155E75", alpha=0.9)
                ax.set_ylim(0, 1.0)
            else:
                ax.set_axis_off()

            ax.set_xticks([])
            ax.set_yticks([])

            if r == 0 and c == 0:
                ax.set_title(f"{cw_title} / {ccw_title}", fontsize=10, pad=12)

        legend_handles = [
            plt.Line2D([0], [0], color="#9A3412", linewidth=2.0, label=cw_title),
            plt.Line2D([0], [0], color="#155E75", linewidth=2.0, label=ccw_title),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
            fontsize=20,
            handlelength=2.8,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        return fig

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.8 * n_rows), squeeze=False)
    for i in range(n_rows * n_cols):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]

        if i < n_plots:
            vals_cw = np.maximum(activations_cw[i], 0)
            vals_ccw = np.maximum(activations_ccw[i], 0)
            scale = max(np.max(vals_cw), np.max(vals_ccw)) + 1e-8
            com_cw = _circular_center_of_mass(vals_cw)
            com_ccw = _circular_center_of_mass(vals_ccw)
            vals_cw = vals_cw / scale
            vals_ccw = vals_ccw / scale
            x = np.arange(activations_cw.shape[-1])
            ax.fill_between(x, 0, vals_cw, color="#E36414", alpha=0.18)
            ax.plot(vals_cw, linewidth=1.8, color="#9A3412")

            ax.fill_between(x, 0, vals_ccw, color="#0F8B8D", alpha=0.18)
            ax.plot(vals_ccw, linewidth=1.8, color="#155E75")
            if com_cw is not None:
                ax.axvline((com_cw / (2 * np.pi)) * (len(x) - 1), linestyle="--", linewidth=1.2, color="#9A3412", alpha=0.9)
            if com_ccw is not None:
                ax.axvline((com_ccw / (2 * np.pi)) * (len(x) - 1), linestyle="--", linewidth=1.2, color="#155E75", alpha=0.9)
            ax.set_xlim(0, activations_ccw.shape[-1] - 1)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

        if r == 0 and c == 0:
            ax.set_title(f"{cw_title} / {ccw_title}")

    legend_handles = [
        plt.Line2D([0], [0], color="#9A3412", linewidth=2.0, label=cw_title),
        plt.Line2D([0], [0], color="#155E75", linewidth=2.0, label=ccw_title),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=20,
        handlelength=2.8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def compute_ratemaps_gc(model, trajectory_generator, options, res=20, n_avg=None, Ng=512, idxs=None):
    '''Compute spatial firing fields'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not np.any(idxs):
        idxs = np.arange(Ng)
    idxs = idxs[:Ng]

    g = np.zeros([n_avg, options.batch_size * options.sequence_length, Ng])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ng, res, res]) 
    counts  = np.zeros([res, res])

    for index in range(n_avg):
        inputs, _, targets = trajectory_generator.get_test_batch()
        if isinstance(targets, tuple):
            pos_batch = targets[0]
        else:
            pos_batch = targets

        if isinstance(inputs, tuple) and len(inputs) == 4 and hasattr(model, "PaS") and hasattr(model, "mEC"):
            s, init_actv, ang_v, init_dir_actv = inputs
            if hasattr(model, "Lmn"):
                lmn_batch = model.Lmn((ang_v, init_dir_actv))
                pas_out = model.PaS(lmn_batch, init_dir_actv)
            else:
                pas_out = model.PaS((ang_v, init_dir_actv))
            pas_batch = pas_out[0] if isinstance(pas_out, tuple) else pas_out
            if hasattr(model, "Conj"):
                conj_batch = model.Conj((s, init_actv), pas_batch)
                g_batch = model.mEC((s, init_actv), conj_batch).detach().cpu().numpy()
            else:
                g_batch = model.mEC((s, init_actv), pas_batch).detach().cpu().numpy()
        else:
            g_batch = model.g(inputs).detach().cpu().numpy()
        
        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        g_batch = g_batch[:,:,idxs].reshape(-1, Ng)
        
        g[index] = g_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + options.box_width/2) / (options.box_width) * res
        y_batch = (pos_batch[:,1] + options.box_height/2) / (options.box_height) * res

        for i in range(options.batch_size*options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += g_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    g = g.reshape([-1, Ng])
    pos = pos.reshape([-1, 2])

    # # scipy binned_statistic_2d is slightly slower
    # activations = scipy.stats.binned_statistic_2d(pos[:,0], pos[:,1], g.T, bins=res)[0]
    rate_map = activations.reshape(Ng, -1)

    return activations, rate_map, g, pos


def compute_ratemaps_conj(model, trajectory_generator, options, res=20, n_avg=None, Ncg=512, idxs=None):
    '''Compute spatial firing fields for the conjunctive layer.'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not hasattr(model, "Conj"):
        raise ValueError("Model must implement Conj() for compute_ratemaps_conj.")

    if hasattr(model, "Ncg"):
        Ncg = min(Ncg, getattr(model, "Ncg"))

    if idxs is None or not np.any(idxs):
        idxs = np.arange(Ncg)
    idxs = np.asarray(idxs)[:Ncg]

    conj = np.zeros([n_avg, options.batch_size * options.sequence_length, Ncg])
    pos = np.zeros([n_avg, options.batch_size * options.sequence_length, 2])

    activations = np.zeros([Ncg, res, res])
    counts = np.zeros([res, res])

    for index in range(n_avg):
        inputs, _, targets = trajectory_generator.get_test_batch()
        if isinstance(targets, tuple):
            pos_batch = targets[0]
        else:
            pos_batch = targets

        if not (isinstance(inputs, tuple) and len(inputs) == 4 and hasattr(model, "PaS")):
            raise ValueError(
                "Expected hierarchical inputs=(speed, init_place_actv, ang_vel, init_dir_actv) "
                "and model with PaS() for compute_ratemaps_conj."
            )

        s, init_actv, ang_v, init_dir_actv = inputs
        if hasattr(model, "Lmn"):
            lmn_batch = model.Lmn((ang_v, init_dir_actv))
            pas_out = model.PaS(lmn_batch, init_dir_actv)
        else:
            pas_out = model.PaS((ang_v, init_dir_actv))
        pas_batch = pas_out[0] if isinstance(pas_out, tuple) else pas_out
        conj_batch = model.Conj((s, init_actv), pas_batch).detach().cpu().numpy()

        pos_batch = np.reshape(pos_batch.cpu(), [-1, 2])
        conj_batch = conj_batch[:, :, idxs].reshape(-1, Ncg)

        conj[index] = conj_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:, 0] + options.box_width / 2) / options.box_width * res
        y_batch = (pos_batch[:, 1] + options.box_height / 2) / options.box_height * res

        for i in range(options.batch_size * options.sequence_length):
            x = x_batch[i]
            y = y_batch[i]
            if x >= 0 and x < res and y >= 0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += conj_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    conj = conj.reshape([-1, Ncg])
    pos = pos.reshape([-1, 2])
    rate_map = activations.reshape(Ncg, -1)

    return activations, rate_map, conj, pos


def compute_ratemaps_dc_split_by_rotation(
    model,
    trajectory_generator,
    options,
    res=60,
    n_avg=None,
    N=512,
    idxs=None,
    vel_threshold=0.0,
    layer="lmn",
):
    '''Compute 1d angular ratemaps split by clockwise and counter-clockwise motion.

    Positive angular velocity is treated as counter-clockwise and negative
    angular velocity as clockwise. Samples with |ang_v| <= vel_threshold are
    ignored for the split maps. Direction-layer activity at time t is aligned
    to the target angle from the previous time step within each sequence.
    '''
    layer = layer.lower()
    if layer not in {"lmn", "pas"}:
        raise ValueError(f"Unsupported layer '{layer}'. Expected 'lmn' or 'pas'.")

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if layer == "lmn" and hasattr(model, "Nlmn"):
        N = min(N, getattr(model, "Nlmn"))
    elif layer == "pas" and hasattr(model, "Npas"):
        N = min(N, getattr(model, "Npas"))
    elif hasattr(model, "Npas"):
        N = min(N, getattr(model, "Npas"))
    else:
        raise ValueError(f"Model does not define the requested '{layer}' layer.")

    if idxs is None:
        idxs = np.arange(N)
    else:
        idxs = np.asarray(idxs)
    idxs = idxs[:N]

    angle_min = getattr(options, "angle_min", 0.0)
    angle_max = getattr(options, "angle_max", 2 * np.pi)
    angle_range = angle_max - angle_min

    activations_cw = np.zeros([N, res])
    activations_ccw = np.zeros([N, res])
    counts_cw = np.zeros([res])
    counts_ccw = np.zeros([res])

    dc_state_cw = []
    dc_state_ccw = []
    ang_cw = []
    ang_ccw = []

    for _ in range(n_avg):
        inputs, _, targets = trajectory_generator.get_test_batch()
        if isinstance(inputs, tuple) and len(inputs) == 4:
            _, _, ang_v, init_dir_actv = inputs
            actual_inputs = (ang_v, init_dir_actv)
        elif isinstance(inputs, tuple) and len(inputs) == 2:
            actual_inputs = inputs
            ang_v = inputs[0]
        else:
            raise ValueError("Expected direction inputs containing angular velocity.")

        if layer == "lmn":
            if not hasattr(model, "Lmn"):
                raise ValueError("Model does not define Lmn(), cannot compute LMN ratemaps.")
            dc_batch = model.Lmn(actual_inputs)
        else:
            if not hasattr(model, "PaS"):
                raise ValueError("Model does not define PaS(), cannot compute PaS ratemaps.")
            if hasattr(model, "Lmn"):
                lmn = model.Lmn(actual_inputs)
                dc_batch = model.PaS(lmn, init_dir_actv)
            else:
                dc_batch = model.PaS(actual_inputs)
        dc_batch = (dc_batch[0] if isinstance(dc_batch, tuple) else dc_batch).detach().cpu().numpy()

        if isinstance(targets, tuple):
            ang_batch = None
            for t in targets:
                if not hasattr(t, "dim"):
                    continue
                if t.dim() == 2:
                    ang_batch = t
                    break
                if t.dim() == 3 and t.shape[-1] == 1:
                    ang_batch = t.squeeze(-1)
                    break
            if ang_batch is None:
                raise ValueError("No angular targets found for direction ratemap.")
        else:
            ang_batch = targets

        ang_batch = ang_batch.detach().cpu().numpy()
        ang_v_batch = ang_v.detach().cpu().numpy()
        dc_batch = dc_batch[:, :, idxs]

        if ang_batch.ndim == 3 and ang_batch.shape[-1] == 1:
            ang_batch = ang_batch.squeeze(-1)
        if ang_v_batch.ndim == 3 and ang_v_batch.shape[-1] == 1:
            ang_v_batch = ang_v_batch.squeeze(-1)

        if ang_batch.ndim != 2 or ang_v_batch.ndim != 2 or dc_batch.ndim != 3:
            raise ValueError(
                f"Unexpected shapes for previous-step alignment: "
                f"ang_batch={ang_batch.shape}, ang_v_batch={ang_v_batch.shape}, dc_batch={dc_batch.shape}"
            )

        T = min(dc_batch.shape[0], ang_batch.shape[0], ang_v_batch.shape[0])
        B = min(dc_batch.shape[1], ang_batch.shape[1], ang_v_batch.shape[1])
        if T < 2:
            raise ValueError("Need at least two time steps for previous-step target alignment.")

        dc_batch = dc_batch[:T, :B, :]
        ang_batch = ang_batch[:T, :B]
        ang_v_batch = ang_v_batch[:T, :B]

        # The model state produced at step t predicts the next head direction.
        # For visualization we align that state with the previous-step target,
        # i.e. pair dc_batch[t] with ang_batch[t-1] within each sequence.
        dc_batch = dc_batch[1:, :, :].transpose(1, 0, 2).reshape(-1, N)
        ang_batch = ang_batch[:-1, :].transpose(1, 0).reshape(-1)
        ang_v_batch = ang_v_batch[1:, :].transpose(1, 0).reshape(-1)

        # dc_batch = dc_batch[2:, :, :].transpose(1, 0, 2).reshape(-1, N)
        # ang_batch = ang_batch[:-2, :].transpose(1, 0).reshape(-1)
        # ang_v_batch = ang_v_batch[2:, :].transpose(1, 0).reshape(-1)

        ang_norm = np.mod(ang_batch - angle_min, angle_range)
        bins = (ang_norm / angle_range) * res

        cw_mask = ang_v_batch < -float(vel_threshold)
        ccw_mask = ang_v_batch > float(vel_threshold)

        if np.any(cw_mask):
            dc_state_cw.append(dc_batch[cw_mask])
            ang_cw.append(ang_batch[cw_mask])
        if np.any(ccw_mask):
            dc_state_ccw.append(dc_batch[ccw_mask])
            ang_ccw.append(ang_batch[ccw_mask])

        for i in np.flatnonzero(cw_mask):
            b = int(bins[i])
            if b >= 0 and b < res:
                counts_cw[b] += 1
                activations_cw[:, b] += dc_batch[i, :]

        for i in np.flatnonzero(ccw_mask):
            b = int(bins[i])
            if b >= 0 and b < res:
                counts_ccw[b] += 1
                activations_ccw[:, b] += dc_batch[i, :]

    for b in range(res):
        if counts_cw[b] > 0:
            activations_cw[:, b] /= counts_cw[b]
        if counts_ccw[b] > 0:
            activations_ccw[:, b] /= counts_ccw[b]

    dc_state_cw = np.concatenate(dc_state_cw, axis=0) if dc_state_cw else np.zeros((0, N))
    dc_state_ccw = np.concatenate(dc_state_ccw, axis=0) if dc_state_ccw else np.zeros((0, N))
    ang_cw = np.concatenate(ang_cw, axis=0) if ang_cw else np.zeros((0,))
    ang_ccw = np.concatenate(ang_ccw, axis=0) if ang_ccw else np.zeros((0,))
    rate_map_cw = activations_cw.reshape(N, -1)
    rate_map_ccw = activations_ccw.reshape(N, -1)

    return (
        activations_cw,
        rate_map_cw,
        dc_state_cw,
        ang_cw,
        activations_ccw,
        rate_map_ccw,
        dc_state_ccw,
        ang_ccw,
    )


def compute_angular_velocity_tuning(
    model,
    trajectory_generator,
    options,
    res=21,
    n_avg=None,
    N=512,
    idxs=None,
    vel_min=None,
    vel_max=None,
    use_abs=False,
    layer="lmn",
):
    """Compute tuning curves as a function of angular velocity.

    Args:
        model: Direction model exposing Lmn() and/or PaS().
        trajectory_generator: Batch generator.
        options: Experiment options.
        res: Number of angular-velocity bins.
        n_avg: Number of batches to average over.
        N: Number of units to keep.
        idxs: Optional unit indices.
        vel_min: Lower bin edge. Defaults to min observed velocity.
        vel_max: Upper bin edge. Defaults to max observed velocity.
        use_abs: If True, tune to |ang_v| instead of signed ang_v.
        layer: Which state to analyze. One of {"lmn", "pas"}.

    Returns:
        activations_vel: [N, res] mean activity per velocity bin.
        rate_map_vel: alias of activations_vel reshaped to [N, res].
        dc_state: [num_samples, N] unit activity samples.
        ang_v_vals: [num_samples] angular velocity samples used for binning.
        bin_centers: [res] centers of angular-velocity bins.
        counts: [res] sample counts per bin.
    """
    layer = layer.lower()
    if layer not in {"lmn", "pas"}:
        raise ValueError(f"Unsupported layer '{layer}'. Expected 'lmn' or 'pas'.")

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if layer == "lmn":
        if not hasattr(model, "Lmn"):
            raise ValueError("Model does not define Lmn(), cannot compute LMN angular velocity tuning.")
        N = min(N, getattr(model, "Nlmn"))
    else:
        if not hasattr(model, "PaS"):
            raise ValueError("Model does not define PaS(), cannot compute PaS angular velocity tuning.")
        N = min(N, getattr(model, "Npas"))

    if idxs is None:
        idxs = np.arange(N)
    else:
        idxs = np.asarray(idxs)
    idxs = idxs[idxs < N][:N]

    num_samples = options.batch_size * options.sequence_length
    dc_state = np.zeros((n_avg, num_samples, len(idxs)))
    ang_v_all = np.zeros((n_avg, num_samples))

    for index in range(n_avg):
        inputs, _, _ = trajectory_generator.get_test_batch()
        if not (isinstance(inputs, tuple) and len(inputs) == 4):
            raise ValueError("Expected inputs=(speed, init_place_actv, ang_v, init_dir_actv).")
        _, _, ang_v, init_dir_actv = inputs
        actual_inputs = (ang_v, init_dir_actv)

        if layer == "lmn":
            dc_batch = model.Lmn(actual_inputs)
        else:
            if hasattr(model, "Lmn"):
                lmn = model.Lmn(actual_inputs)
                dc_batch = model.PaS(lmn, init_dir_actv)
            else:
                dc_batch = model.PaS(actual_inputs)

        dc_batch = (dc_batch[0] if isinstance(dc_batch, tuple) else dc_batch).detach().cpu().numpy()
        dc_state[index] = dc_batch[:, :, idxs].reshape(-1, len(idxs))
        ang_v_all[index] = ang_v.detach().cpu().numpy().reshape(-1)

    dc_state = dc_state.reshape(-1, len(idxs))
    ang_v_all = ang_v_all.reshape(-1)

    ang_v_vals = np.abs(ang_v_all) if use_abs else ang_v_all
    if vel_min is None:
        vel_min = float(np.min(ang_v_vals))
    if vel_max is None:
        vel_max = float(np.max(ang_v_vals))
    if vel_max <= vel_min:
        vel_max = vel_min + 1e-6

    bin_edges = np.linspace(vel_min, vel_max, res + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    activations_vel = np.zeros((len(idxs), res))
    counts = np.zeros(res)

    bin_ids = np.digitize(ang_v_vals, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, res - 1)

    for b in range(res):
        mask = bin_ids == b
        if np.any(mask):
            activations_vel[:, b] = np.mean(dc_state[mask], axis=0)
            counts[b] = np.sum(mask)

    rate_map_vel = activations_vel.reshape(dc_state.shape[1], -1)
    return activations_vel, rate_map_vel, dc_state, ang_v_vals, bin_centers, counts


def score_angular_velocity_tuning(activations_vel, vel_bin_centers=None, eps=1e-8):
    """Return simple modulation metrics for angular-velocity tuning curves.

    Args:
        activations_vel: [N, res] velocity tuning curves.
        vel_bin_centers: Optional [res] velocity-bin centers used to compute a
            signed linear slope. If None, evenly spaced bins on [-1, 1] are used.
        eps: Numerical stabilizer.

    Returns:
        modulation_index: [N] signed slope of tuning curve versus velocity.
            More negative values indicate preference for negative angular
            velocities; more positive values indicate preference for positive
            angular velocities.
        tuning_index: [N] signed tuning strength, computed as
            sign(modulation_index) * (max - min) / (max + min + eps).
        peak_bin: [N] preferred velocity-bin index.
        peak_value: [N] peak response.
    """
    vals = np.maximum(activations_vel, 0)
    if vel_bin_centers is None:
        x = np.linspace(-1.0, 1.0, vals.shape[1], dtype=np.float64)
    else:
        x = np.asarray(vel_bin_centers, dtype=np.float64)
        if x.ndim != 1 or x.shape[0] != vals.shape[1]:
            raise ValueError("vel_bin_centers must have shape [res] matching activations_vel.shape[1].")

    x_centered = x - np.mean(x)
    x_var = np.sum(x_centered ** 2) + eps
    y_centered = vals - np.mean(vals, axis=1, keepdims=True)
    modulation_index = (y_centered @ x_centered) / x_var

    peak_value = np.max(vals, axis=1)
    min_value = np.min(vals, axis=1)
    tuning_strength = (peak_value - min_value) / (peak_value + min_value + eps)
    tuning_index = np.sign(modulation_index) * tuning_strength
    peak_bin = np.argmax(vals, axis=1)
    return modulation_index, tuning_index, peak_bin, peak_value


def compute_ratemaps_dc_lmn_pas(
    model,
    trajectory_generator,
    options,
    res=60,
    n_avg=None,
    N_lmn=512,
    N_pas=512,
    idxs_lmn=None,
    idxs_pas=None,
):
    '''Compute 1d angular firing fields for LMN and PaS (two-layer model).'''

    if not n_avg:
        n_avg = 1000 // options.sequence_length
    # Clamp N to model sizes if available
    if hasattr(model, "Nlmn"):
        N_lmn = min(N_lmn, getattr(model, "Nlmn"))
    if hasattr(model, "Npas"):
        N_pas = min(N_pas, getattr(model, "Npas"))

    if not np.any(idxs_lmn):
        idxs_lmn = np.arange(N_lmn)
    if not np.any(idxs_pas):
        idxs_pas = np.arange(N_pas)
    # Ensure idxs are in bounds
    idxs_lmn = np.asarray(idxs_lmn)
    idxs_pas = np.asarray(idxs_pas)
    idxs_lmn = idxs_lmn[idxs_lmn < N_lmn]
    idxs_pas = idxs_pas[idxs_pas < N_pas]
    idxs_lmn = idxs_lmn[:N_lmn]
    idxs_pas = idxs_pas[:N_pas]

    angle_min = getattr(options, "angle_min", 0.0)
    angle_max = getattr(options, "angle_max", 2 * np.pi)
    angle_range = angle_max - angle_min

    lmn = np.zeros([n_avg, options.batch_size * options.sequence_length, N_lmn])
    pas = np.zeros([n_avg, options.batch_size * options.sequence_length, N_pas])
    ang = np.zeros([n_avg, options.batch_size * options.sequence_length])

    activations_lmn = np.zeros([N_lmn, res])
    activations_pas = np.zeros([N_pas, res])
    counts = np.zeros([res])

    for index in range(n_avg):
        inputs, _, targets = trajectory_generator.get_test_batch()
        if isinstance(inputs, tuple) and len(inputs) == 4:
            _, _, ang_v, init_dir_actv = inputs
            actual_inputs = (ang_v, init_dir_actv)
        else:
            actual_inputs = inputs

        if hasattr(model, "Lmn") and hasattr(model, "PaS"):
            lmn_batch = model.Lmn(actual_inputs)
            pas_out = model.PaS(lmn_batch, actual_inputs[1])
            pas_batch = pas_out[0] if isinstance(pas_out, tuple) else pas_out
        else:
            raise ValueError("Model does not support LMN->PaS hierarchy for compute_ratemaps_dc_lmn_pas.")

        if isinstance(targets, tuple):
            ang_batch = None
            for t in targets:
                if not hasattr(t, "dim"):
                    continue
                if t.dim() == 2:
                    ang_batch = t
                    break
                if t.dim() == 3 and t.shape[-1] == 1:
                    ang_batch = t.squeeze(-1)
                    break
            if ang_batch is None:
                raise ValueError("No angular targets found for direction ratemap.")
        else:
            ang_batch = targets

        ang_batch = ang_batch.detach().cpu().numpy().reshape(-1)
        lmn_batch = lmn_batch.detach().cpu().numpy()[:, :, idxs_lmn].reshape(-1, N_lmn)
        pas_batch = pas_batch.detach().cpu().numpy()[:, :, idxs_pas].reshape(-1, N_pas)

        lmn[index] = lmn_batch
        pas[index] = pas_batch
        ang[index] = ang_batch

        ang_norm = np.mod(ang_batch - angle_min, angle_range)
        bins = (ang_norm / angle_range) * res

        for i in range(options.batch_size * options.sequence_length):
            b = int(bins[i])
            if b >= 0 and b < res:
                counts[b] += 1
                activations_lmn[:, b] += lmn_batch[i, :]
                activations_pas[:, b] += pas_batch[i, :]

    for b in range(res):
        if counts[b] > 0:
            activations_lmn[:, b] /= counts[b]
            activations_pas[:, b] /= counts[b]

    lmn = lmn.reshape([-1, N_lmn])
    pas = pas.reshape([-1, N_pas])
    ang = ang.reshape([-1])
    rate_map_lmn = activations_lmn.reshape(N_lmn, -1)
    rate_map_pas = activations_pas.reshape(N_pas, -1)

    return activations_lmn, rate_map_lmn, activations_pas, rate_map_pas, lmn, pas, ang


def compute_ratemaps_lmn_pas_conj_mec(
    model,
    trajectory_generator,
    options,
    res=60,
    res_pos=20,
    n_avg=None,
    N_lmn=512,
    N_pas=512,
    N_conj=512,
    N_mec=512,
    idxs_lmn=None,
    idxs_pas=None,
    idxs_conj=None,
    idxs_mec=None,
):
    """Compute directional and spatial ratemaps for LMN/PaS/Conj/mEC in one pass."""

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    if not (hasattr(model, "Lmn") and hasattr(model, "PaS") and hasattr(model, "Conj") and hasattr(model, "mEC")):
        raise ValueError(
            "Model must implement Lmn, PaS, Conj, and mEC for compute_ratemaps_dc_lmn_pas_mec."
        )

    # Clamp N to model sizes if available
    if hasattr(model, "Nlmn"):
        N_lmn = min(N_lmn, getattr(model, "Nlmn"))
    if hasattr(model, "Npas"):
        N_pas = min(N_pas, getattr(model, "Npas"))
    if hasattr(model, "Ncg"):
        N_conj = min(N_conj, getattr(model, "Ncg"))
    if hasattr(model, "Ng"):
        N_mec = min(N_mec, getattr(model, "Ng"))

    if not np.any(idxs_lmn):
        idxs_lmn = np.arange(N_lmn)
    if not np.any(idxs_pas):
        idxs_pas = np.arange(N_pas)
    if idxs_conj is None or not np.any(idxs_conj):
        idxs_conj = np.arange(N_conj)
    if not np.any(idxs_mec):
        idxs_mec = np.arange(N_mec)

    # Ensure idxs are in bounds
    idxs_lmn = np.asarray(idxs_lmn)
    idxs_pas = np.asarray(idxs_pas)
    idxs_conj = np.asarray(idxs_conj)
    idxs_mec = np.asarray(idxs_mec)
    idxs_lmn = idxs_lmn[idxs_lmn < N_lmn][:N_lmn]
    idxs_pas = idxs_pas[idxs_pas < N_pas][:N_pas]
    idxs_conj = idxs_conj[idxs_conj < N_conj][:N_conj]
    idxs_mec = idxs_mec[idxs_mec < N_mec][:N_mec]

    angle_min = getattr(options, "angle_min", 0.0)
    angle_max = getattr(options, "angle_max", 2 * np.pi)
    angle_range = angle_max - angle_min

    samples_per_batch = options.batch_size * options.sequence_length
    lmn = np.zeros([n_avg, samples_per_batch, N_lmn])
    pas = np.zeros([n_avg, samples_per_batch, N_pas])
    conj = np.zeros([n_avg, samples_per_batch, N_conj])
    mec = np.zeros([n_avg, samples_per_batch, N_mec])
    ang = np.zeros([n_avg, samples_per_batch])
    pos = np.zeros([n_avg, samples_per_batch, 2])

    activations_lmn_dir = np.zeros([N_lmn, res])
    activations_pas_dir = np.zeros([N_pas, res])
    activations_conj_dir = np.zeros([N_conj, res])
    activations_mec_dir = np.zeros([N_mec, res])
    activations_lmn_spatial = np.zeros([N_lmn, res_pos, res_pos])
    activations_pas_spatial = np.zeros([N_pas, res_pos, res_pos])
    activations_conj_spatial = np.zeros([N_conj, res_pos, res_pos])
    activations_mec_spatial = np.zeros([N_mec, res_pos, res_pos])
    counts_ang = np.zeros([res])
    counts_pos = np.zeros([res_pos, res_pos])

    for index in range(n_avg):
        inputs, _, targets = trajectory_generator.get_test_batch()
        if not (isinstance(inputs, tuple) and len(inputs) == 4):
            raise ValueError("Expected inputs=(speed, init_place_actv, ang_vel, init_dir_actv).")
        s, init_actp, ang_v, init_dir_actv = inputs

        # Forward four-layer hierarchy: LMN -> PaS -> Conj -> mEC
        lmn_batch = model.Lmn((ang_v, init_dir_actv))
        pas_out = model.PaS(lmn_batch, init_dir_actv)
        pas_batch = pas_out[0] if isinstance(pas_out, tuple) else pas_out
        conj_batch = model.Conj((s, init_actp), pas_batch)
        mec_batch = model.mEC((s, init_actp), conj_batch)

        # Parse targets
        if isinstance(targets, tuple):
            pos_batch = targets[0]
            ang_batch = None
            for t in targets:
                if not hasattr(t, "dim"):
                    continue
                if t.dim() == 2:
                    ang_batch = t
                    break
                if t.dim() == 3 and t.shape[-1] == 1:
                    ang_batch = t.squeeze(-1)
                    break
            if ang_batch is None:
                raise ValueError("No angular targets found for direction ratemap.")
        else:
            raise ValueError("Expected targets tuple with position and angle.")

        ang_batch = ang_batch.detach().cpu().numpy().reshape(-1)
        pos_batch = pos_batch.detach().cpu().numpy().reshape(-1, 2)
        lmn_batch = lmn_batch.detach().cpu().numpy()[:, :, idxs_lmn].reshape(-1, N_lmn)
        pas_batch = pas_batch.detach().cpu().numpy()[:, :, idxs_pas].reshape(-1, N_pas)
        conj_batch = conj_batch.detach().cpu().numpy()[:, :, idxs_conj].reshape(-1, N_conj)
        mec_batch = mec_batch.detach().cpu().numpy()[:, :, idxs_mec].reshape(-1, N_mec)

        lmn[index] = lmn_batch
        pas[index] = pas_batch
        conj[index] = conj_batch
        mec[index] = mec_batch
        ang[index] = ang_batch
        pos[index] = pos_batch

        # Angular bins for LMN/PaS
        ang_norm = np.mod(ang_batch - angle_min, angle_range)
        bins = (ang_norm / angle_range) * res

        # Position bins for mEC
        x_batch = (pos_batch[:, 0] + options.box_width / 2) / options.box_width * res_pos
        y_batch = (pos_batch[:, 1] + options.box_height / 2) / options.box_height * res_pos

        for i in range(samples_per_batch):
            b = int(bins[i])
            if 0 <= b < res:
                counts_ang[b] += 1
                activations_lmn_dir[:, b] += lmn_batch[i, :]
                activations_pas_dir[:, b] += pas_batch[i, :]
                activations_conj_dir[:, b] += conj_batch[i, :]
                activations_mec_dir[:, b] += mec_batch[i, :]

            x = int(x_batch[i])
            y = int(y_batch[i])
            if 0 <= x < res_pos and 0 <= y < res_pos:
                counts_pos[x, y] += 1
                activations_lmn_spatial[:, x, y] += lmn_batch[i, :]
                activations_pas_spatial[:, x, y] += pas_batch[i, :]
                activations_conj_spatial[:, x, y] += conj_batch[i, :]
                activations_mec_spatial[:, x, y] += mec_batch[i, :]

    for b in range(res):
        if counts_ang[b] > 0:
            activations_lmn_dir[:, b] /= counts_ang[b]
            activations_pas_dir[:, b] /= counts_ang[b]
            activations_conj_dir[:, b] /= counts_ang[b]
            activations_mec_dir[:, b] /= counts_ang[b]

    for x in range(res_pos):
        for y in range(res_pos):
            if counts_pos[x, y] > 0:
                activations_lmn_spatial[:, x, y] /= counts_pos[x, y]
                activations_pas_spatial[:, x, y] /= counts_pos[x, y]
                activations_conj_spatial[:, x, y] /= counts_pos[x, y]
                activations_mec_spatial[:, x, y] /= counts_pos[x, y]

    lmn = lmn.reshape([-1, N_lmn])
    pas = pas.reshape([-1, N_pas])
    conj = conj.reshape([-1, N_conj])
    mec = mec.reshape([-1, N_mec])
    ang = ang.reshape([-1])
    pos = pos.reshape([-1, 2])

    rate_map_lmn_dir = activations_lmn_dir.reshape(N_lmn, -1)
    rate_map_pas_dir = activations_pas_dir.reshape(N_pas, -1)
    rate_map_conj_dir = activations_conj_dir.reshape(N_conj, -1)
    rate_map_mec_dir = activations_mec_dir.reshape(N_mec, -1)
    rate_map_lmn_spatial = activations_lmn_spatial.reshape(N_lmn, -1)
    rate_map_pas_spatial = activations_pas_spatial.reshape(N_pas, -1)
    rate_map_conj_spatial = activations_conj_spatial.reshape(N_conj, -1)
    rate_map_mec_spatial = activations_mec_spatial.reshape(N_mec, -1)

    return (
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
    )


def save_ratemaps(model, trajectory_generator, options, step, res=20, n_avg=None):
    if not n_avg:
        n_avg = 300 // options.sequence_length
    imdir = options.save_dir + "/" + options.run_ID

    (
        lmn_activations_dir,
        _,
        pas_activations_dir,
        _,
        conj_activations_dir,
        _,
        mec_activations_dir,
        _,
        lmn_activations_spatial,
        _,
        pas_activations_spatial,
        _,
        conj_activations_spatial,
        _,
        mec_activations_spatial,
        _,
        lmn,
        pas,
        conj,
        mec,
        ang,
        pos,
    ) = compute_ratemaps_lmn_pas_conj_mec(
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

    conj_fig = plot_ratemaps(conj_activations_spatial, n_plots=len(conj_activations_spatial))
    imsave(imdir + "/" + str(step) + "_conj.png", conj_fig)

    mec_fig = plot_ratemaps(mec_activations_spatial, n_plots=len(mec_activations_spatial))
    imsave(imdir + "/" + str(step) + "_gc.png", mec_fig)

    lmn_spatial_fig = plot_ratemaps(lmn_activations_spatial, n_plots=len(lmn_activations_spatial))
    imsave(imdir + "/" + str(step) + "_lmn_spatial.png", lmn_spatial_fig)

    pas_spatial_fig = plot_ratemaps(pas_activations_spatial, n_plots=len(pas_activations_spatial))
    imsave(imdir + "/" + str(step) + "_pas_spatial.png", pas_spatial_fig)

    # LMN plot
    pref_idx_lmn = np.argmax(lmn_activations_dir, axis=1)
    order_lmn = np.argsort(pref_idx_lmn)
    lmn_sorted = lmn_activations_dir[order_lmn]
    n_plots_lmn = min(512, len(lmn_sorted))
    lmn_fig = plot_ratemaps_1d(lmn_sorted, n_plots=n_plots_lmn, width=16, polar=True)
    lmn_fig.savefig(imdir + "/" + str(step) + "_dc_lmn.png", dpi=150)
    plt.close(lmn_fig)

    # PaS plot
    pref_idx_pas = np.argmax(pas_activations_dir, axis=1)
    order_pas = np.argsort(pref_idx_pas)
    pas_sorted = pas_activations_dir[order_pas]
    n_plots_pas = min(512, len(pas_sorted))
    pas_fig = plot_ratemaps_1d(pas_sorted, n_plots=n_plots_pas, width=16, polar=True)
    pas_fig.savefig(imdir + "/" + str(step) + "_dc_pas.png", dpi=150)
    plt.close(pas_fig)

    pref_idx_conj = np.argmax(conj_activations_dir, axis=1)
    order_conj = np.argsort(pref_idx_conj)
    conj_sorted = conj_activations_dir[order_conj]
    n_plots_conj = min(512, len(conj_sorted))
    conj_dir_fig = plot_ratemaps_1d(conj_sorted, n_plots=n_plots_conj, width=16, polar=True)
    conj_dir_fig.savefig(imdir + "/" + str(step) + "_dc_conj.png", dpi=150)
    plt.close(conj_dir_fig)

    pref_idx_mec_dir = np.argmax(mec_activations_dir, axis=1)
    order_mec_dir = np.argsort(pref_idx_mec_dir)
    mec_sorted = mec_activations_dir[order_mec_dir]
    n_plots_mec_dir = min(512, len(mec_sorted))
    mec_dir_fig = plot_ratemaps_1d(mec_sorted, n_plots=n_plots_mec_dir, width=16, polar=True)
    mec_dir_fig.savefig(imdir + "/" + str(step) + "_dc_mec.png", dpi=150)
    plt.close(mec_dir_fig)

def break_at_wrap(angle_series, wrap=np.pi):
    a = np.asarray(angle_series).squeeze()
    out = a.copy()
    jumps = np.abs(np.diff(out))
    out[1:][jumps > wrap] = np.nan
    return out

def eval_model(model, config, trajectory_generator,place_cells,direction_cells):
    inputs, outputs, targets = trajectory_generator.get_test_batch()
    s, init_actp, ang_v, init_dir_actv = inputs
    lmn_inputs = (ang_v, init_dir_actv)
    mec_inputs = (s, init_actp)

    place_outputs, direction_outputs = outputs
    pos, move_dir = targets

    move_dir = move_dir.detach().cpu().numpy()
    if move_dir.ndim == 3 and move_dir.shape[-1] == 1:
        move_dir = move_dir[..., 0]
    pos = pos.detach().cpu().numpy()

    model.eval()
    preds = model.predict(lmn_inputs, mec_inputs)
    if len(preds) == 3:
        lmn_preds, pas_preds, pos_preds = preds
    elif len(preds) == 2:
        pas_preds, pos_preds = preds
        lmn_preds = None
    else:
        raise ValueError(f"Expected model.predict(...) to return 2 or 3 tensors, got {len(preds)}.")

    pred_dir_lmn = None
    if lmn_preds is not None:
        pred_dir_lmn = direction_cells.get_nearest_cell_dir(lmn_preds).detach().cpu().numpy()
    pred_dir_pas = direction_cells.get_nearest_cell_dir(pas_preds, k=5).detach().cpu().numpy()
    pred_pos = place_cells.get_nearest_cell_pos(pos_preds, k=5).detach().cpu().numpy()

    t = np.arange(move_dir.shape[0])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4), dpi=100)

    # left: direction
    ax0 = axs[0]
    for i in range(1):
        true_line = break_at_wrap(move_dir[:, i])
        pas_line = break_at_wrap(pred_dir_pas[:, i])
        if pred_dir_lmn is not None:
            lmn_line = break_at_wrap(pred_dir_lmn[:, i])

        ax0.plot(t, true_line, c='black', linewidth=1, label='True direction' if i == 0 else None)
        if pred_dir_lmn is not None:
            ax0.plot(t, lmn_line, '.-', c='C0', linewidth=1, label='LMN decoded' if i == 0 else None)
        ax0.plot(t, pas_line, '.-', c='C1', linewidth=1, label='PaS decoded' if i == 0 else None)

    ax0.legend()
    sns.despine(ax=ax0)
    ax0.set_yticks([-np.pi, 0, np.pi])
    ax0.set_yticklabels(['-π', '0', 'π'])
    ax0.set_xticks([0, len(t)//2, len(t)-1])
    ax0.set_xlabel('Time step')
    ax0.set_title('Angular integration')

    # right: position
    ax1 = axs[1]
    for i in range(1):
        ax1.plot(pos[:, i, 0], pos[:, i, 1], c='black', linewidth=2, label='True position' if i == 0 else None)
        ax1.plot(pred_pos[:, i, 0], pred_pos[:, i, 1], '.-', c='C1', label='Decoded position' if i == 0 else None)

    us = place_cells.us.detach().cpu().numpy()
    ax1.scatter(us[:, 0], us[:, 1], s=20, alpha=0.5, c='lightgrey')
    ax1.legend()
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    ax1.set_xlim([-config.box_width/2, config.box_width/2])
    ax1.set_ylim([-config.box_height/2, config.box_height/2])
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_title('Positional integration')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def compute_lmn_ratemaps_by_rotation_trueangle(
    model,
    trajectory_generator,
    options,
    res=60,
    n_avg=100,
    N_lmn=None,
    idxs_lmn=None,
    zero_tol=1e-8,
    target_alignment="post_update",   # "post_update" or "pre_update"
):
    """
    Compute LMN CW/CCW tuning curves using TRUE trajectory angle for binning.
    Split CCW/CW using sign of angular velocity.
    """
    if N_lmn is None:
        N_lmn = getattr(model, "Nlmn")

    if idxs_lmn is None:
        idxs_lmn = np.arange(N_lmn)

    idxs_lmn = np.asarray(idxs_lmn)
    idxs_lmn = idxs_lmn[idxs_lmn < N_lmn]
    N_lmn = len(idxs_lmn)

    angle_min = getattr(options, "angle_min", 0.0)
    angle_max = getattr(options, "angle_max", 2 * np.pi)
    angle_range = angle_max - angle_min

    n_total = n_avg * options.batch_size * options.sequence_length

    lmn_all = np.zeros((n_total, N_lmn), dtype=np.float32)
    true_ang_all = np.zeros((n_total,), dtype=np.float32)
    ang_v_all = np.zeros((n_total,), dtype=np.float32)

    activations_lmn_ccw = np.zeros((N_lmn, res), dtype=np.float32)
    activations_lmn_cw  = np.zeros((N_lmn, res), dtype=np.float32)
    counts_ccw = np.zeros((res,), dtype=np.float32)
    counts_cw  = np.zeros((res,), dtype=np.float32)

    write_start = 0

    for _ in range(n_avg):
        inputs, _, targets = trajectory_generator.get_test_batch_circle()

        # inputs
        if isinstance(inputs, tuple) and len(inputs) == 2:
            ang_v_batch_torch, init_dir_actv = inputs
        elif isinstance(inputs, tuple) and len(inputs) == 4:
            _, _, ang_v_batch_torch, init_dir_actv = inputs
        else:
            raise ValueError(
                f"Unsupported input format. Got type={type(inputs)}, "
                f"len={len(inputs) if isinstance(inputs, tuple) else 'NA'}"
            )

        # targets: robustly extract true angle
        if isinstance(targets, tuple):
            ang_batch_torch = None
            for t in targets:
                if not hasattr(t, "dim"):
                    continue
                if t.dim() == 2:
                    ang_batch_torch = t
                    break
                if t.dim() == 3 and t.shape[-1] == 1:
                    ang_batch_torch = t.squeeze(-1)
                    break
            if ang_batch_torch is None:
                raise ValueError("No angular targets found for true-angle ratemap.")
        else:
            ang_batch_torch = targets
            if hasattr(ang_batch_torch, "dim") and ang_batch_torch.dim() == 3 and ang_batch_torch.shape[-1] == 1:
                ang_batch_torch = ang_batch_torch.squeeze(-1)

        # LMN hidden activity
        lmn_batch_torch = model.Lmn((ang_v_batch_torch, init_dir_actv))   # [T, B, Nlmn]
        lmn_batch = lmn_batch_torch.detach().cpu().numpy()[:, :, idxs_lmn]
        ang_v_batch = ang_v_batch_torch.detach().cpu().numpy().squeeze(-1)
        ang_batch = ang_batch_torch.detach().cpu().numpy()

        if lmn_batch.shape[:2] != ang_v_batch.shape[:2]:
            raise ValueError(f"Shape mismatch: lmn={lmn_batch.shape}, ang_v={ang_v_batch.shape}")
        if lmn_batch.shape[:2] != ang_batch.shape[:2]:
            raise ValueError(f"Shape mismatch: lmn={lmn_batch.shape}, ang={ang_batch.shape}")

        # alignment
        if target_alignment == "post_update":
            ang_for_binning = ang_batch
        elif target_alignment == "pre_update":
            ang_for_binning = np.empty_like(ang_batch)
            ang_for_binning[1:] = ang_batch[:-1]
            ang_for_binning[0] = np.nan
        else:
            raise ValueError("target_alignment must be 'post_update' or 'pre_update'")

        # flatten
        lmn_flat = lmn_batch.reshape(-1, N_lmn)
        ang_v_flat = ang_v_batch.reshape(-1)
        ang_flat = ang_for_binning.reshape(-1)

        valid_mask = np.isfinite(ang_flat)
        lmn_flat = lmn_flat[valid_mask]
        ang_v_flat = ang_v_flat[valid_mask]
        ang_flat = ang_flat[valid_mask]

        n_this = ang_flat.shape[0]
        write_end = write_start + n_this

        lmn_all[write_start:write_end] = lmn_flat
        true_ang_all[write_start:write_end] = ang_flat
        ang_v_all[write_start:write_end] = ang_v_flat
        write_start = write_end

        # bin by TRUE angle
        ang_norm = np.mod(ang_flat - angle_min, angle_range)
        bins = (ang_norm / angle_range) * res

        for i in range(n_this):
            b = int(bins[i])
            if b == res:
                b = res - 1
            if 0 <= b < res:
                if ang_v_flat[i] > zero_tol:
                    counts_ccw[b] += 1
                    activations_lmn_ccw[:, b] += lmn_flat[i, :]
                elif ang_v_flat[i] < -zero_tol:
                    counts_cw[b] += 1
                    activations_lmn_cw[:, b] += lmn_flat[i, :]

    # trim
    lmn_all = lmn_all[:write_start]
    true_ang_all = true_ang_all[:write_start]
    ang_v_all = ang_v_all[:write_start]

    # normalize
    for b in range(res):
        if counts_ccw[b] > 0:
            activations_lmn_ccw[:, b] /= counts_ccw[b]
        if counts_cw[b] > 0:
            activations_lmn_cw[:, b] /= counts_cw[b]

    valid_ccw = ang_v_all > zero_tol
    valid_cw  = ang_v_all < -zero_tol

    return {
        "ccw": {
            "activations_lmn": activations_lmn_ccw,
            "lmn": lmn_all[valid_ccw],
            "true_ang": true_ang_all[valid_ccw],
            "ang_v": ang_v_all[valid_ccw],
            "counts": counts_ccw,
        },
        "cw": {
            "activations_lmn": activations_lmn_cw,
            "lmn": lmn_all[valid_cw],
            "true_ang": true_ang_all[valid_cw],
            "ang_v": ang_v_all[valid_cw],
            "counts": counts_cw,
        },
        "all": {
            "lmn": lmn_all,
            "true_ang": true_ang_all,
            "ang_v": ang_v_all,
        }
    }
    
def compute_ratemaps_dc_lmn_pas_by_rotation(
    model,
    trajectory_generator,
    options,
    res=60,
    n_avg=None,
    N_lmn=512,
    N_pas=512,
    idxs_lmn=None,
    idxs_pas=None,
    zero_tol=1e-8,
):
    '''Compute 1d angular firing fields for LMN and PaS separately for
    CCW-rotating and CW-rotating trajectory segments.

    Convention:
        ang_v > 0  -> CCW
        ang_v < 0  -> CW
        |ang_v| <= zero_tol -> ignored
    '''

    if not n_avg:
        n_avg = 1000 // options.sequence_length

    # Clamp N to model sizes if available
    if hasattr(model, "Nlmn"):
        N_lmn = min(N_lmn, getattr(model, "Nlmn"))
    if hasattr(model, "Npas"):
        N_pas = min(N_pas, getattr(model, "Npas"))

    if idxs_lmn is None:
        idxs_lmn = np.arange(N_lmn)
    if idxs_pas is None:
        idxs_pas = np.arange(N_pas)

    idxs_lmn = np.asarray(idxs_lmn)
    idxs_pas = np.asarray(idxs_pas)
    idxs_lmn = idxs_lmn[idxs_lmn < N_lmn]
    idxs_pas = idxs_pas[idxs_pas < N_pas]
    idxs_lmn = idxs_lmn[:N_lmn]
    idxs_pas = idxs_pas[:N_pas]

    angle_min = getattr(options, "angle_min", 0.0)
    angle_max = getattr(options, "angle_max", 2 * np.pi)
    angle_range = angle_max - angle_min

    n_total = n_avg * options.batch_size * options.sequence_length

    # Store all samples
    lmn_all = np.zeros([n_total, N_lmn], dtype=np.float32)
    pas_all = np.zeros([n_total, N_pas], dtype=np.float32)
    ang_all = np.zeros([n_total], dtype=np.float32)
    ang_v_all = np.zeros([n_total], dtype=np.float32)

    # Separate accumulators
    activations_lmn_ccw = np.zeros([N_lmn, res], dtype=np.float32)
    activations_pas_ccw = np.zeros([N_pas, res], dtype=np.float32)
    counts_ccw = np.zeros([res], dtype=np.float32)

    activations_lmn_cw = np.zeros([N_lmn, res], dtype=np.float32)
    activations_pas_cw = np.zeros([N_pas, res], dtype=np.float32)
    counts_cw = np.zeros([res], dtype=np.float32)

    write_start = 0

    for index in range(n_avg):
        inputs, _, targets = trajectory_generator.get_test_batch()

        # Extract actual model inputs
        if isinstance(inputs, tuple) and len(inputs) == 4:
            _, _, ang_v, init_dir_actv = inputs
            actual_inputs = (ang_v, init_dir_actv)
        else:
            actual_inputs = inputs

        # Forward pass through LMN -> PaS hierarchy
        if hasattr(model, "Lmn") and hasattr(model, "PaS"):
            lmn_batch = model.Lmn(actual_inputs)
            pas_out = model.PaS(lmn_batch, actual_inputs[1])
            pas_batch = pas_out[0] if isinstance(pas_out, tuple) else pas_out
        else:
            raise ValueError("Model does not support LMN->PaS hierarchy for compute_ratemaps_dc_lmn_pas_by_rotation.")

        # Extract angular targets
        if isinstance(targets, tuple):
            ang_batch = None
            for t in targets:
                if not hasattr(t, "dim"):
                    continue
                if t.dim() == 2:
                    ang_batch = t
                    break
                if t.dim() == 3 and t.shape[-1] == 1:
                    ang_batch = t.squeeze(-1)
                    break
            if ang_batch is None:
                raise ValueError("No angular targets found for direction ratemap.")
        else:
            ang_batch = targets

        # Extract angular velocity from actual_inputs
        # actual_inputs should be (ang_v, init_dir_actv)
        if not (isinstance(actual_inputs, tuple) and len(actual_inputs) >= 1):
            raise ValueError("Could not extract ang_v from inputs.")
        ang_v_batch = actual_inputs[0]

        # Convert to numpy
        ang_batch = ang_batch.detach().cpu().numpy().reshape(-1)
        ang_v_batch = ang_v_batch.detach().cpu().numpy().reshape(-1)

        lmn_batch = lmn_batch.detach().cpu().numpy()[:, :, idxs_lmn].reshape(-1, N_lmn)
        pas_batch = pas_batch.detach().cpu().numpy()[:, :, idxs_pas].reshape(-1, N_pas)

        n_this = ang_batch.shape[0]
        write_end = write_start + n_this

        lmn_all[write_start:write_end] = lmn_batch
        pas_all[write_start:write_end] = pas_batch
        ang_all[write_start:write_end] = ang_batch
        ang_v_all[write_start:write_end] = ang_v_batch

        write_start = write_end

        # Angular binning
        ang_norm = np.mod(ang_batch - angle_min, angle_range)
        bins = (ang_norm / angle_range) * res

        for i in range(n_this):
            b = int(bins[i])
            if b < 0 or b >= res:
                continue

            if ang_v_batch[i] > zero_tol:
                # CCW
                counts_ccw[b] += 1
                activations_lmn_ccw[:, b] += lmn_batch[i, :]
                activations_pas_ccw[:, b] += pas_batch[i, :]
            elif ang_v_batch[i] < -zero_tol:
                # CW
                counts_cw[b] += 1
                activations_lmn_cw[:, b] += lmn_batch[i, :]
                activations_pas_cw[:, b] += pas_batch[i, :]
            # else: near-zero ang_v, ignore

    # Normalize
    for b in range(res):
        if counts_ccw[b] > 0:
            activations_lmn_ccw[:, b] /= counts_ccw[b]
            activations_pas_ccw[:, b] /= counts_ccw[b]

        if counts_cw[b] > 0:
            activations_lmn_cw[:, b] /= counts_cw[b]
            activations_pas_cw[:, b] /= counts_cw[b]

    rate_map_lmn_ccw = activations_lmn_ccw.reshape(N_lmn, -1)
    rate_map_pas_ccw = activations_pas_ccw.reshape(N_pas, -1)
    rate_map_lmn_cw = activations_lmn_cw.reshape(N_lmn, -1)
    rate_map_pas_cw = activations_pas_cw.reshape(N_pas, -1)

    valid_ccw = ang_v_all > zero_tol
    valid_cw = ang_v_all < -zero_tol

    return {
        "ccw": {
            "activations_lmn": activations_lmn_ccw,
            "rate_map_lmn": rate_map_lmn_ccw,
            "activations_pas": activations_pas_ccw,
            "rate_map_pas": rate_map_pas_ccw,
            "lmn": lmn_all[valid_ccw],
            "pas": pas_all[valid_ccw],
            "ang": ang_all[valid_ccw],
            "ang_v": ang_v_all[valid_ccw],
            "counts": counts_ccw,
        },
        "cw": {
            "activations_lmn": activations_lmn_cw,
            "rate_map_lmn": rate_map_lmn_cw,
            "activations_pas": activations_pas_cw,
            "rate_map_pas": rate_map_pas_cw,
            "lmn": lmn_all[valid_cw],
            "pas": pas_all[valid_cw],
            "ang": ang_all[valid_cw],
            "ang_v": ang_v_all[valid_cw],
            "counts": counts_cw,
        },
        "all": {
            "lmn": lmn_all,
            "pas": pas_all,
            "ang": ang_all,
            "ang_v": ang_v_all,
        }
    }
