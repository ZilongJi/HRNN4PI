# -*- coding: utf-8 -*-
import torch

class LMN_PaS_Conj_mEC(torch.nn.Module):
    """
    Four layer hierarchical Recurrent Neural Network model of
    Lateral mammillary nucleus (LMN), Parasubiculum (PaS), conjunctive layer (Conj)
    and medial entorhinal cortex (mEC)
    performing angular and positional path integration.

    1, LMN does angular path integration from angular velocity input and output the current angle
    2, PaS receives LMN activity and perform angular PI with speed modulation, output the current angle
    3, Conj receives PaS activity together with speed-related drive
    4, mEC receives Conj activity and outputs place code
    """
    def __init__(self, options, direction_cells, place_cells):
        super(LMN_PaS_Conj_mEC, self).__init__()

        # Dimensions
        self.Nd = options.Nd
        self.Nlmn = options.Nlmn
        self.Npas = options.Npas
        self.Ncg = options.Ncg
        self.Ng = options.Ng
        self.Np = options.Np

        self.sequence_length = options.sequence_length

        # Regularization
        self.weight_decay_lmn = options.weight_decay_lmn
        self.weight_decay_pas = options.weight_decay_pas
        self.weight_decay_conj = options.weight_decay_conj
        self.weight_decay_mec = options.weight_decay_mec
        self.use_pas_layernorm = getattr(options, "use_pas_layernorm", False)

        self.direction_cells = direction_cells
        self.place_cells = place_cells

        # LMN: angular velocity -> LMN state -> direction readout
        self.LmnEncoder = torch.nn.Linear(self.Nd, self.Nlmn, bias=False)
        self.LmnRNN = torch.nn.RNN(
            input_size=1,
            hidden_size=self.Nlmn,
            nonlinearity=options.activation,
            bias=False,
        )

        # PaS: LMN state -> PaS state -> direction readout
        self.PaSEncoder = torch.nn.Linear(self.Nd, self.Npas, bias=False)
        self.PaSRNN = torch.nn.RNN(
            input_size=self.Nlmn,
            hidden_size=self.Npas,
            nonlinearity=options.activation,
            bias=False,
        )
        self.PaSDecoder = torch.nn.Linear(self.Npas, self.Nd, bias=False)

        # Conj: [speed, PaS] -> Conj state
        self.PaSNorm = torch.nn.LayerNorm(self.Npas) if self.use_pas_layernorm else None
        # self.PasSpeedconj = torch.nn.Linear(self.Npas + 1, self.Ncg, bias=False)

        self.conjEncoder = torch.nn.Linear(self.Np, self.Ncg, bias=False)
        self.conjRNN = torch.nn.RNN(
            input_size=self.Npas + 1,
            hidden_size=self.Ncg,
            nonlinearity=options.activation,
            bias=False,
        )

        # mEC: Conj -> mEC state -> place readout
        self.mECEncoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.mECRNN = torch.nn.RNN(
            input_size=self.Ncg,
            hidden_size=self.Ng,
            nonlinearity=options.activation,
            bias=False,
        )
        self.mECDecoder = torch.nn.Linear(self.Ng, self.Np, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

    def Lmn(self, inputs):
        '''
        Compute LMN activations.
        Args:
            inputs: tuple (ang_vel, init_dir_actv)
                ang_vel: [sequence_length, batch_size, 1]
                init_dir_actv: [batch_size, Nd]
        Returns:
            lmn: [sequence_length, batch_size, Nlmn]
        '''
        ang_v, init_dir_actv = inputs
        if init_dir_actv.dim() == 1:
            init_dir_actv = init_dir_actv.unsqueeze(0)
        init_state = self.LmnEncoder(init_dir_actv)[None]
        lmn, _ = self.LmnRNN(ang_v, init_state)
        return lmn

    def PaS(self, lmn, init_dir_actv):
        '''
        Compute Parasubicular activations from LMN activity.
        Args:
            lmn: [sequence_length, batch_size, Nlmn]
            init_dir_actv: [batch_size, Nd]
        Returns:
            pas: [sequence_length, batch_size, Npas]
        '''
        if init_dir_actv.dim() == 1:
            init_dir_actv = init_dir_actv.unsqueeze(0)
        init_state = self.PaSEncoder(init_dir_actv)[None]
        pas, _ = self.PaSRNN(lmn, init_state)
        return pas
    
    def Conj(self, inputs, pas):
        '''
        Compute conjunctive-layer activations.
        Args:
            inputs: 1d speed inputs with shape [sequence_length, batch_size, 1].
            pas: Parasubicular activations with shape [sequence_length, batch_size, Npas].

        Returns:
            conj: Conj activations with shape [sequence_length, batch_size, Ncg].
        '''
        s, p0 = inputs
        if p0.dim() == 1:
            p0 = p0.unsqueeze(0)
        init_state = self.conjEncoder(p0)[None]
        # pas = pas.detach() # Don't backprop through PaS when training mEC, to avoid collapse of PaS representations.
        if self.PaSNorm is not None:
            pas = self.PaSNorm(pas)
        drive_in = torch.cat([s, pas], dim=-1)
        # drive = self.PasSpeedconj(drive_in)
        conj,_ = self.conjRNN(drive_in, init_state)
        return conj

    def mEC(self, inputs, conj):
        '''
        Compute mEC activations from conjunctive activity.
        Args:
            inputs: 1d speed inputs with shape [sequence_length, batch_size, 1].
            conj: Conj activations with shape [sequence_length, batch_size, Ncg].

        Returns:
            mEC: mEC activations with shape [sequence_length, batch_size, Ng].
        '''
        s, p0 = inputs
        if p0.dim() == 1:
            p0 = p0.unsqueeze(0)
        init_state = self.mECEncoder(p0)[None]
        mEC,_ = self.mECRNN(conj, init_state)
        return mEC

    def predict(self, lmn_inputs, mEC_inputs):
        '''
        Predict direction/place cell codes for LMN, PaS and mEC.
        Args:
            lmn_inputs: tuple (ang_vel, init_dir_actv)
            mEC_inputs: tuple (speed, init_place_actv)

        Returns: 
            pas_preds: [sequence_length, batch_size, Nd]
            mec_preds: [sequence_length, batch_size, Np]
        '''
        ang_v, init_dir_actv = lmn_inputs
        lmn = self.Lmn((ang_v, init_dir_actv))
        pas = self.PaS(lmn, init_dir_actv)
        conj = self.Conj(mEC_inputs, pas)
        pas_preds = self.PaSDecoder(pas)
        mec_preds = self.mECDecoder(self.mEC(mEC_inputs, conj))

        outputs = [pas_preds, mec_preds]
        return tuple(outputs)


    def compute_loss(self, inputs, outputs, targets):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: 1d angular velocity inputs with shape [sequence_length, batch_size, 1].
            dc_outputs: Ground truth direction cell activations with shape 
                [sequence_length, batch_size, Nd].
            mdir: Ground truth 1d direction with shape [sequence_length, batch_size, 1].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded angular error in radians.
        '''
        
        s, init_actp, ang_v, init_dir_actv = inputs
        lmn_inputs = (ang_v, init_dir_actv)
        mEC_inputs = (s, init_actp)
        
        place_outputs, dc_outputs = outputs #ground truth place and direction cell activations from single values
        pos, mdir = targets
        
        if mdir.dim() == 2:
            mdir = mdir[:, :, None]

        # Forward once so we can regularize mEC hidden activity directly.
        lmn = self.Lmn(lmn_inputs)
        pas = self.PaS(lmn, init_dir_actv)
        conj = self.Conj(mEC_inputs, pas)
        pas_preds = self.PaSDecoder(pas)
        mec_h = self.mEC(mEC_inputs, conj)
        pos_preds = self.mECDecoder(mec_h)

        # 2) PaS direction PI loss
        y_dir = dc_outputs
        
        # Supervise every time step, matching the original HRNN training protocol.
        supervised_idx = slice(None)
        
        yhat_pas = self.softmax(pas_preds)
        loss_pas = -(y_dir[supervised_idx] * torch.log(yhat_pas[supervised_idx] + 1e-8)).sum(-1).mean()

        # 3) mEC position PI loss

        y_pos = place_outputs
        yhat_mec = self.softmax(pos_preds)
        loss_mec = -(y_pos[supervised_idx] * torch.log(yhat_mec[supervised_idx] + 1e-8)).sum(-1).mean()

        # Weight regularization
        loss_lmn_wd = self.weight_decay_lmn * (self.LmnRNN.weight_hh_l0**2).sum()
        loss_pas += self.weight_decay_pas * (self.PaSRNN.weight_hh_l0**2).sum()
        loss_mec += self.weight_decay_conj * (self.conjRNN.weight_hh_l0**2).sum()
        loss_mec += self.weight_decay_mec * (self.mECRNN.weight_hh_l0**2).sum()
        loss = loss_lmn_wd + loss_pas + loss_mec

        # Compute decoding error
        angle_range = getattr(self.direction_cells, "angle_range", 2 * torch.pi)

        pred_dir_pas = self.direction_cells.get_nearest_cell_dir(pas_preds)
        diff_pas = torch.abs(mdir[supervised_idx] - pred_dir_pas[supervised_idx])
        diff_pas = torch.minimum(diff_pas, angle_range - diff_pas)
        err_pas = torch.sqrt((diff_pas**2).mean())

        pred_pos = self.place_cells.get_nearest_cell_pos(pos_preds)
        err_mec = torch.sqrt(((pos[supervised_idx] - pred_pos[supervised_idx])**2).sum(-1)).mean()

        return loss, loss_pas, loss_mec, err_pas, err_mec
