# -*- coding: utf-8 -*-
import torch
import numpy as np

from visualize import save_ratemaps
import os


class Trainer(object):
    def __init__(self, options, model, trajectory_generator, restore=True):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        self._build_optimizer()

        self.loss = []
        self.loss_pas = []
        self.loss_mec = []
        self.err = []
        self.err_pas = []
        self.err_mec = []
        self.lmn_pas_frozen = False
        self.best_loss_pas = np.inf
        self.pas_bad_epochs = 0

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, 'most_recent_model.pth')
        self.history_path = os.path.join(self.ckpt_dir, 'most_recent_history.npz')
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(
                torch.load(
                    ckpt_path,
                    map_location=options.device,
                    weights_only=True,
                )
            )
            self._load_history()
            print("Restored trained model from {}".format(ckpt_path))
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print("Saving to: {}".format(self.ckpt_dir))

    def _build_optimizer(self):
        lr = self.options.learning_rate
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(trainable_params, lr=lr)

    def freeze_lmn_pas(self):
        modules = [
            self.model.LmnEncoder,
            self.model.LmnRNN,
            self.model.PaSEncoder,
            self.model.PaSRNN,
            self.model.PaSDecoder,
        ]

        for module in modules:
            for p in module.parameters():
                p.requires_grad = False

        self.lmn_pas_frozen = True
        self._build_optimizer()
        print("LMN and PaS are now frozen. Continuing to train Conj and mEC.")

    def train_step(self, inputs, outputs, targets):
        ''' 
        Train on one batch of trajectories.

        Args:
            inputs: Model inputs (format depends on task).
            outputs: Ground truth outputs (place or direction cell activations).
            targets: Ground truth targets (position or head direction).

        Returns:
            loss: Avg. loss for this training batch.
            err_pas: Avg. decoded angular error in radians.
            err_mec: Avg. decoded position error in meters.
        '''
        self.model.zero_grad()

        loss, loss_pas, loss_mec, err_pas, err_mec = self.model.compute_loss(inputs, outputs, targets)

        loss.backward()
        if hasattr(self.model, "conjRNN") and hasattr(self.model, "mECRNN"):
            pas_grad = None
            if hasattr(self.model, "PaSRNN"):
                if hasattr(self.model.PaSRNN, "weight_hh_l0") and self.model.PaSRNN.weight_hh_l0.grad is not None:
                    pas_grad = self.model.PaSRNN.weight_hh_l0.grad.abs().mean().item()
                elif hasattr(self.model.PaSRNN, "J") and self.model.PaSRNN.J.weight.grad is not None:
                    pas_grad = self.model.PaSRNN.J.weight.grad.abs().mean().item()
            conj_grad = None
            if hasattr(self.model.conjRNN, "weight_hh_l0") and self.model.conjRNN.weight_hh_l0.grad is not None:
                conj_grad = self.model.conjRNN.weight_hh_l0.grad.abs().mean().item()
            mec_grad = None
            if hasattr(self.model.mECRNN, "weight_hh_l0") and self.model.mECRNN.weight_hh_l0.grad is not None:
                mec_grad = self.model.mECRNN.weight_hh_l0.grad.abs().mean().item()
            self._last_mec_grads = (
                conj_grad,
                None,
                mec_grad,
                pas_grad,
            )
        self.optimizer.step()

        return (
            loss.item(),
            loss_pas.item(),
            loss_mec.item(),
            err_pas.item(),
            err_mec.item(),
        )

    def _save_history(self, epoch_idx=None):
        history = {
            "loss": np.asarray(self.loss, dtype=np.float32),
            "loss_pas": np.asarray(self.loss_pas, dtype=np.float32),
            "loss_mec": np.asarray(self.loss_mec, dtype=np.float32),
            "err": np.asarray(self.err, dtype=np.float32),
            "err_pas": np.asarray(self.err_pas, dtype=np.float32),
            "err_mec": np.asarray(self.err_mec, dtype=np.float32),
            "lmn_pas_frozen": np.asarray([int(self.lmn_pas_frozen)], dtype=np.int32),
            "best_loss_pas": np.asarray([self.best_loss_pas], dtype=np.float32),
            "pas_bad_epochs": np.asarray([self.pas_bad_epochs], dtype=np.int32),
        }
        if epoch_idx is not None:
            history["epoch"] = np.asarray([epoch_idx], dtype=np.int32)
        np.savez(self.history_path, **history)

    def _load_history(self):
        if not os.path.isfile(self.history_path):
            return
        history = np.load(self.history_path, allow_pickle=False)
        self.loss = history["loss"].tolist() if "loss" in history else []
        self.loss_pas = history["loss_pas"].tolist() if "loss_pas" in history else []
        self.loss_mec = history["loss_mec"].tolist() if "loss_mec" in history else []
        self.err = [tuple(x) for x in history["err"].tolist()] if "err" in history else []
        self.err_pas = history["err_pas"].tolist() if "err_pas" in history else []
        self.err_mec = history["err_mec"].tolist() if "err_mec" in history else []
        if "lmn_pas_frozen" in history:
            self.lmn_pas_frozen = bool(history["lmn_pas_frozen"][0])
        if "best_loss_pas" in history:
            self.best_loss_pas = float(history["best_loss_pas"][0])
        if "pas_bad_epochs" in history:
            self.pas_bad_epochs = int(history["pas_bad_epochs"][0])
        if self.lmn_pas_frozen:
            self.freeze_lmn_pas()

    def train(self, n_epochs: int = 100, n_steps=1000, save=True):
        ''' 
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        '''

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        if save:
            # Save a baseline ratemap before training
            save_ratemaps(self.model, self.trajectory_generator,
                          self.options, step="init")

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(n_epochs):
            epoch_loss = []
            epoch_loss_pas = []
            epoch_loss_mec = []
            epoch_err_pas = []
            epoch_err_mec = []

            for step_idx in range(n_steps):
                inputs, outputs, targets = next(gen)
                loss, loss_pas, loss_mec, err_pas, err_mec = self.train_step(inputs, outputs, targets)
                self.loss.append(loss)
                self.loss_pas.append(loss_pas)
                self.loss_mec.append(loss_mec)

                err_pas_deg = err_pas * 180 / np.pi
                err_mec_cm = 100 * err_mec
                self.err.append((err_pas_deg, err_mec_cm))
                self.err_pas.append(err_pas_deg)
                self.err_mec.append(err_mec_cm)

                epoch_loss.append(loss)
                epoch_loss_pas.append(loss_pas)
                epoch_loss_mec.append(loss_mec)
                epoch_err_pas.append(err_pas_deg)
                epoch_err_mec.append(err_mec_cm)

            current_lr = self.optimizer.param_groups[0]["lr"]
            print('Epoch: {}/{}. Loss: {} (PaS {}, mEC {}). Angular PI Err: {}° / Positional PI Err: {}cm. LR: {:.2e}'.format(
                epoch_idx, n_epochs,
                np.round(np.mean(epoch_loss), 4),
                np.round(np.mean(epoch_loss_pas), 4),
                np.round(np.mean(epoch_loss_mec), 4),
                np.round(np.mean(epoch_err_pas), 4),
                np.round(np.mean(epoch_err_mec), 4),
                current_lr))

            if not self.lmn_pas_frozen:
                mean_loss_pas = np.mean(epoch_loss_pas)
                min_delta = getattr(self.options, "freeze_pas_min_delta", 0.005)
                patience = getattr(self.options, "freeze_pas_patience", 8)

                if mean_loss_pas < self.best_loss_pas - min_delta:
                    self.best_loss_pas = mean_loss_pas
                    self.pas_bad_epochs = 0
                else:
                    self.pas_bad_epochs += 1
                    print(f"PaS loss plateau count: {self.pas_bad_epochs}/{patience}")

                if self.pas_bad_epochs >= patience:
                    self.freeze_lmn_pas()

            if hasattr(self, "_last_mec_grads"):
                g_conj, _, g_mec, g_pas = self._last_mec_grads
                conj_str = f"{g_conj:.3e}" if g_conj is not None else "n/a"
                mec_str = f"{g_mec:.3e}" if g_mec is not None else "n/a"
                pas_str = f"{g_pas:.3e}" if g_pas is not None else "n/a"
                print(
                    f"Gradients (epoch {epoch_idx}): "
                    f"conjRNN {conj_str} | mECRNN {mec_str} | PaSRNN {pas_str}"
                )

            if save:
                self._save_history(epoch_idx=epoch_idx)
                # Save a picture of rate maps every 10 epochs
                if epoch_idx % 20 == 0:
                    # Save checkpoint
                    ckpt_path = os.path.join(self.ckpt_dir, 'epoch_{}.pth'.format(epoch_idx))
                    torch.save(self.model.state_dict(), ckpt_path)
                    torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir,
                                                                    'most_recent_model.pth'))
                    save_ratemaps(self.model, self.trajectory_generator,
                                self.options, step=epoch_idx, res=60)
