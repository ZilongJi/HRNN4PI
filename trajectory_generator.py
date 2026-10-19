# -*- coding: utf-8 -*-
import torch
import os
import numpy as np


class TrajectoryGenerator(object):

    def __init__(self, options, place_cells=None, direction_cells=None):
        self.options = options
        self.place_cells = place_cells
        self.direction_cells = direction_cells

    def avoid_wall(self, position, md, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:, 0]
        y = position[:, 1]
        dists = [box_width / 2 - x, box_height / 2 - y, box_width / 2 + x, box_height / 2 + y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4) * np.pi / 2
        theta = angles[np.argmin(dists, axis=0)]
        md = np.mod(md, 2 * np.pi)
        a_wall = md - theta
        a_wall = np.mod(a_wall + np.pi, 2 * np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region) * (np.abs(a_wall) < np.pi / 2)
        turn_angle = np.zeros_like(md)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall]) * (np.pi / 2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(
        self,
        box_width,
        box_height,
        batch_size,
        *,
        samples=None,
        dt=0.02,
        sigma=5.76 * 2,
        b=0.13 * 2 * np.pi,
        mu=0,
        border_region=0.03,
        avoid_walls=True,
        random_turn=None,
        random_vel=None,
        ang_vel_noise_std=0.0,
    ):
        '''Generate a random walk in a rectangular box'''
        if samples is None:
            samples = self.options.sequence_length
        self.border_region = border_region  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples + 2, 2])
        move_dir = np.zeros([batch_size, samples + 2])
        position[:, 0, 0] = np.random.uniform(-box_width / 2, box_width / 2, batch_size)
        position[:, 0, 1] = np.random.uniform(-box_height / 2, box_height / 2, batch_size)
        move_dir[:, 0] = np.random.uniform(0, 2 * np.pi, batch_size)
        velocity = np.zeros([batch_size, samples + 2])

        # Generate sequence of random boosts and turns
        if random_turn is None:
            random_turn = np.random.normal(mu, sigma, [batch_size, samples + 1])
        if ang_vel_noise_std and ang_vel_noise_std > 0:
            random_turn = random_turn + np.random.normal(0.0, ang_vel_noise_std, [batch_size, samples + 1])
        if random_vel is None:
            random_vel = np.random.rayleigh(b, [batch_size, samples + 1])
        v = np.abs(np.random.normal(0, b * np.pi / 2, batch_size))

        for t in range(samples + 1):
            # Update velocity
            v = random_vel[:, t]
            turn_angle = np.zeros(batch_size)

            if avoid_walls and not self.options.periodic:
                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(position[:, t], move_dir[:, t], box_width, box_height)
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt * random_turn[:, t]

            # Take a step
            velocity[:, t] = v * dt
            update = velocity[:, t, None] * np.stack([np.cos(move_dir[:, t]), np.sin(move_dir[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update

            # Rotate movement direction
            move_dir[:, t + 1] = move_dir[:, t] + turn_angle

        # Periodic boundaries
        if self.options.periodic: #Can try Mattias' data here!!!
            position[:, :, 0] = np.mod(position[:, :, 0] + box_width / 2, box_width) - box_width / 2
            position[:, :, 1] = np.mod(position[:, :, 1] + box_height / 2, box_height) - box_height / 2

        move_dir = np.mod(move_dir + np.pi, 2 * np.pi) - np.pi  # Periodic variable

        traj = {}
        # Input variables
        traj['init_md'] = move_dir[:, 0, None]
        traj['init_x'] = position[:, 1, 0, None]
        traj['init_y'] = position[:, 1, 1, None]

        traj['ego_v'] = velocity[:, 1:-1]
        ang_v = np.diff(move_dir, axis=-1)
        # Circular difference to avoid 2pi jumps at the boundary
        ang_v = np.mod(ang_v + np.pi, 2 * np.pi) - np.pi
        traj['ang_v'] = ang_v[:, :-1]
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

        # Target variables
        traj['target_md'] = move_dir[:, 1:-1]
        traj['target_x'] = position[:, 2:, 0]
        traj['target_y'] = position[:, 2:, 1]

        return traj

    def generate_straight_trajectory(
        self,
        box_width,
        box_height,
        batch_size,
        *,
        samples=None,
        dt=0.02,
        b=0.13 * 2 * np.pi,
        border_region=0.1,
        avoid_walls=False,
        ang_vel_noise_std=0.0,
    ):
        '''Generate straight-line trajectories (no turning) in a rectangular box'''
        if samples is None:
            samples = self.options.sequence_length
        random_turn = np.zeros([batch_size, samples + 1])
        return self.generate_trajectory(
            box_width,
            box_height,
            batch_size,
            samples=samples,
            dt=dt,
            sigma=0.0,
            b=b,
            mu=0.0,
            border_region=border_region,
            avoid_walls=avoid_walls,
            random_turn=random_turn,
            ang_vel_noise_std=ang_vel_noise_std,
        )

    def generate_circular_trajectory(
        self,
        box_width,
        box_height,
        batch_size,
        *,
        samples=None,
        dt=0.02,
        b=0.13 * 2 * np.pi,
        omega=1.0,
        speed=None,
        border_region=0.03,
        avoid_walls=False,
        ang_vel_noise_std=0.0,
    ):
        '''Generate circular trajectories (constant turn rate) in a rectangular box'''
        if samples is None:
            samples = self.options.sequence_length
        random_turn = np.full([batch_size, samples + 1], omega)
        random_vel = None
        if speed is not None:
            random_vel = np.full([batch_size, samples + 1], speed)
        return self.generate_trajectory(
            box_width,
            box_height,
            batch_size,
            samples=samples,
            dt=dt,
            sigma=0.0,
            b=b,
            mu=0.0,
            border_region=border_region,
            avoid_walls=avoid_walls,
            random_turn=random_turn,
            random_vel=random_vel,
            ang_vel_noise_std=ang_vel_noise_std,
        )

    def generate_shoot_trajectory(
        self,
        box_width,
        box_height,
        batch_size,
        *,
        start_pos=(0.0, 0.0),
        shoot_dir=0.0,
        samples=None,
        dt=0.02,
        speed=None,
        ego_v=None,
        ang_vel_noise_std=0.0,
    ):
        '''Generate trajectories that start from a specified position and shoot along a specified direction.'''
        if samples is None:
            samples = self.options.sequence_length

        # Parse start position: (2,) or (batch_size, 2)
        start_pos = np.asarray(start_pos, dtype=np.float32)
        if start_pos.ndim == 1:
            if start_pos.shape[0] != 2:
                raise ValueError("start_pos must be shape (2,) or (batch_size, 2).")
            start_pos = np.tile(start_pos[None, :], (batch_size, 1))
        elif start_pos.ndim == 2:
            if start_pos.shape != (batch_size, 2):
                raise ValueError("start_pos with ndim=2 must have shape (batch_size, 2).")
        else:
            raise ValueError("start_pos must be shape (2,) or (batch_size, 2).")

        # Parse shoot direction: scalar or (batch_size,)
        shoot_dir = np.asarray(shoot_dir, dtype=np.float32)
        if shoot_dir.ndim == 0:
            shoot_dir = np.full((batch_size,), float(shoot_dir), dtype=np.float32)
        elif shoot_dir.ndim == 1:
            if shoot_dir.shape[0] != batch_size:
                raise ValueError("shoot_dir with ndim=1 must have shape (batch_size,).")
        else:
            raise ValueError("shoot_dir must be scalar or shape (batch_size,).")

        if ego_v is not None:
            step_speed = np.full((batch_size,), float(ego_v), dtype=np.float32)
        else:
            # Keep compatibility with existing APIs where speed is pre-dt velocity.
            if speed is None:
                speed = 0.13 * 2 * np.pi
            step_speed = np.full((batch_size,), float(speed) * dt, dtype=np.float32)

        position = np.zeros([batch_size, samples + 2, 2], dtype=np.float32)
        move_dir = np.zeros([batch_size, samples + 2], dtype=np.float32)
        velocity = np.zeros([batch_size, samples + 2], dtype=np.float32)

        # Keep compatibility with existing trajectory indexing:
        # init_pos is taken from position[:, 1], and target starts at position[:, 2].
        position[:, 0, :] = start_pos
        position[:, 1, :] = start_pos
        move_dir[:, 0] = shoot_dir
        move_dir[:, 1] = shoot_dir

        if ang_vel_noise_std and ang_vel_noise_std > 0:
            random_turn = np.random.normal(0.0, ang_vel_noise_std, [batch_size, samples + 1]).astype(np.float32)
        else:
            random_turn = np.zeros([batch_size, samples + 1], dtype=np.float32)

        for t in range(1, samples + 1):
            velocity[:, t] = step_speed
            update = velocity[:, t, None] * np.stack([np.cos(move_dir[:, t]), np.sin(move_dir[:, t])], axis=-1)
            position[:, t + 1] = position[:, t] + update
            move_dir[:, t + 1] = move_dir[:, t] + dt * random_turn[:, t]

        # Periodic boundaries
        if self.options.periodic:
            position[:, :, 0] = np.mod(position[:, :, 0] + box_width / 2, box_width) - box_width / 2
            position[:, :, 1] = np.mod(position[:, :, 1] + box_height / 2, box_height) - box_height / 2

        move_dir = np.mod(move_dir + np.pi, 2 * np.pi) - np.pi

        traj = {}
        traj['init_md'] = move_dir[:, 0, None]
        traj['init_x'] = position[:, 1, 0, None]
        traj['init_y'] = position[:, 1, 1, None]

        traj['ego_v'] = velocity[:, 1:-1]
        ang_v = np.diff(move_dir, axis=-1)
        ang_v = np.mod(ang_v + np.pi, 2 * np.pi) - np.pi
        traj['ang_v'] = ang_v[:, :-1]
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:, :-1], np.sin(ang_v)[:, :-1]

        traj['target_md'] = move_dir[:, 1:-1]
        traj['target_x'] = position[:, 2:, 0]
        traj['target_y'] = position[:, 2:, 1]

        return traj

    def get_generator(self, batch_size=None, box_width=None, box_height=None):
        '''
        Returns a generator that yields batches of trajectories
        '''
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        use_place = getattr(self.options, "use_place_cells", self.place_cells is not None)
        use_direction = getattr(self.options, "use_direction_cells", self.direction_cells is not None)
        if not use_place and not use_direction:
            raise ValueError("At least one of place_cells or direction_cells must be provided.")

        while True:
            traj = self.generate_trajectory(box_width, box_height, batch_size)

            v = np.stack([traj['ego_v'] * np.cos(traj['target_md']),
                          traj['ego_v'] * np.sin(traj['target_md'])], axis=-1)
            v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)
            s = torch.tensor(traj['ego_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)

            pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
            pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
            # Put on GPU if GPU is available
            pos = pos.to(self.options.device)
            place_outputs = None
            if use_place:
                place_outputs = self.place_cells.get_activation(pos)

            init_pos = np.stack([traj['init_x'], traj['init_y']], axis=-1)
            init_pos = torch.tensor(init_pos, dtype=torch.float32)
            init_pos = init_pos.to(self.options.device)
            init_actp = None
            if use_place:
                init_actp = self.place_cells.get_activation(init_pos).squeeze(1)

            init_md = torch.tensor(traj['init_md'], dtype=torch.float32)
            init_md = init_md.to(self.options.device)
            init_dir_actv = None
            if use_direction:
                init_dir_actv = self.direction_cells.get_activation(init_md).squeeze(1)

            ang_v = torch.tensor(traj['ang_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
            ang_v = ang_v.to(self.options.device)

            move_dir = torch.tensor(traj['target_md'], dtype=torch.float32).transpose(0, 1)
            move_dir = move_dir.to(self.options.device)
            
            direction_outputs = None
            if use_direction:
                direction_outputs = self.direction_cells.get_activation(move_dir)

            v = v.to(self.options.device)
            s = s.to(self.options.device)
            if use_place and not use_direction:
                inputs = (v, init_actp)
                yield (inputs, place_outputs, pos)
            elif use_direction and not use_place:
                inputs = (ang_v, init_dir_actv)
                yield (inputs, direction_outputs, move_dir)
            else:
                inputs = (s, init_actp, ang_v, init_dir_actv)
                outputs = (place_outputs, direction_outputs)
                targets = (pos, move_dir)
                yield (inputs, outputs, targets)

    def get_test_batch(self, batch_size=None, box_width=None, box_height=None):
        ''' For testing performance, returns a batch of smample trajectories'''
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        use_place = getattr(self.options, "use_place_cells", self.place_cells is not None)
        use_direction = getattr(self.options, "use_direction_cells", self.direction_cells is not None)
        if not use_place and not use_direction:
            raise ValueError("At least one of place_cells or direction_cells must be provided.")

        traj = self.generate_trajectory(box_width, box_height, batch_size)

        v = np.stack([traj['ego_v'] * np.cos(traj['target_md']),
                      traj['ego_v'] * np.sin(traj['target_md'])], axis=-1)
        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)
        s = torch.tensor(traj['ego_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)

        pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
        pos = pos.to(self.options.device)
        place_outputs = None
        if use_place:
            place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj['init_x'], traj['init_y']], axis=-1)
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to(self.options.device)
        init_actp = None
        if use_place:
            init_actp = self.place_cells.get_activation(init_pos).squeeze(1)

        move_dir = torch.tensor(traj['target_md'], dtype=torch.float32).transpose(0, 1)
        move_dir = move_dir.to(self.options.device)
        direction_outputs = None
        if use_direction:
            direction_outputs = self.direction_cells.get_activation(move_dir)

        init_md = torch.tensor(traj['init_md'], dtype=torch.float32)
        init_md = init_md.to(self.options.device)
        init_dir_actv = None
        if use_direction:
            init_dir_actv = self.direction_cells.get_activation(init_md).squeeze(1)

        ang_v = torch.tensor(traj['ang_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
        ang_v = ang_v.to(self.options.device)

        v = v.to(self.options.device)
        s = s.to(self.options.device)
        if use_place and not use_direction:
            inputs = (v, init_actp)
            return (inputs, pos, place_outputs)
        if use_direction and not use_place:
            inputs = (ang_v, init_dir_actv)
            return (inputs, direction_outputs, move_dir)

        inputs = (s, init_actp, ang_v, init_dir_actv)
        outputs = (place_outputs, direction_outputs)
        targets = (pos, move_dir)
        return (inputs, outputs, targets)

    def get_test_batch_straight(
        self,
        batch_size=None,
        box_width=None,
        box_height=None,
        *,
        samples=None,
        dt=0.02,
        b=0.13 * 2 * np.pi,
        border_region=0.03,
        avoid_walls=False,
        ang_vel_noise_std=0.0,
    ):
        '''For testing performance, returns a batch of straight trajectories'''
        return self.get_test_batch_custom(
            batch_size=batch_size,
            box_width=box_width,
            box_height=box_height,
            mode="straight",
            samples=samples,
            dt=dt,
            b=b,
            border_region=border_region,
            avoid_walls=avoid_walls,
            ang_vel_noise_std=ang_vel_noise_std,
        )

    def get_test_batch_circle(
        self,
        batch_size=None,
        box_width=None,
        box_height=None,
        *,
        samples=None,
        dt=0.02,
        b=0.13 * 2 * np.pi,
        omega=1.0,
        speed=None,
        border_region=0.03,
        avoid_walls=False,
        ang_vel_noise_std=0.0,
    ):
        '''For testing performance, returns a batch of circular trajectories'''
        return self.get_test_batch_custom(
            batch_size=batch_size,
            box_width=box_width,
            box_height=box_height,
            mode="circle",
            samples=samples,
            dt=dt,
            b=b,
            omega=omega,
            speed=speed,
            border_region=border_region,
            avoid_walls=avoid_walls,
            ang_vel_noise_std=ang_vel_noise_std,
        )

    def get_test_batch_shoot(
        self,
        batch_size=None,
        box_width=None,
        box_height=None,
        *,
        start_pos=(0.0, 0.0),
        shoot_dir=0.0,
        samples=None,
        dt=0.02,
        speed=None,
        ego_v=None,
        ang_vel_noise_std=0.0,
    ):
        '''For testing performance, returns trajectories from specified start point and shooting direction.'''
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        use_place = getattr(self.options, "use_place_cells", self.place_cells is not None)
        use_direction = getattr(self.options, "use_direction_cells", self.direction_cells is not None)
        if not use_place and not use_direction:
            raise ValueError("At least one of place_cells or direction_cells must be provided.")

        traj = self.generate_shoot_trajectory(
            box_width,
            box_height,
            batch_size,
            start_pos=start_pos,
            shoot_dir=shoot_dir,
            samples=samples,
            dt=dt,
            speed=speed,
            ego_v=ego_v,
            ang_vel_noise_std=ang_vel_noise_std,
        )

        v = np.stack([traj['ego_v'] * np.cos(traj['target_md']),
                      traj['ego_v'] * np.sin(traj['target_md'])], axis=-1)
        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)
        s = torch.tensor(traj['ego_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)

        pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
        pos = pos.to(self.options.device)
        place_outputs = None
        if use_place:
            place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj['init_x'], traj['init_y']], axis=-1)
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to(self.options.device)
        init_actp = None
        if use_place:
            init_actp = self.place_cells.get_activation(init_pos).squeeze(1)

        move_dir = torch.tensor(traj['target_md'], dtype=torch.float32).transpose(0, 1)
        move_dir = move_dir.to(self.options.device)
        direction_outputs = None
        if use_direction:
            direction_outputs = self.direction_cells.get_activation(move_dir)

        init_md = torch.tensor(traj['init_md'], dtype=torch.float32)
        init_md = init_md.to(self.options.device)
        init_dir_actv = None
        if use_direction:
            init_dir_actv = self.direction_cells.get_activation(init_md).squeeze(1)

        ang_v = torch.tensor(traj['ang_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
        ang_v = ang_v.to(self.options.device)

        v = v.to(self.options.device)
        s = s.to(self.options.device)
        if use_place and not use_direction:
            inputs = (v, init_actp)
            return (inputs, pos, place_outputs)
        if use_direction and not use_place:
            inputs = (ang_v, init_dir_actv)
            return (inputs, direction_outputs, move_dir)

        inputs = (s, init_actp, ang_v, init_dir_actv)
        outputs = (place_outputs, direction_outputs)
        targets = (pos, move_dir)
        return (inputs, outputs, targets)

    def get_test_batch_custom(
        self,
        batch_size=None,
        box_width=None,
        box_height=None,
        *,
        mode="straight",
        samples=None,
        dt=0.02,
        b=0.13 * 2 * np.pi,
        omega=1.0,
        speed=None,
        border_region=0.03,
        avoid_walls=False,
        ang_vel_noise_std=0.0,
    ):
        '''For testing performance, returns a batch of straight or circular trajectories'''
        if not batch_size:
            batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height

        use_place = getattr(self.options, "use_place_cells", self.place_cells is not None)
        use_direction = getattr(self.options, "use_direction_cells", self.direction_cells is not None)
        if not use_place and not use_direction:
            raise ValueError("At least one of place_cells or direction_cells must be provided.")

        if mode == "straight":
            traj = self.generate_straight_trajectory(
                box_width,
                box_height,
                batch_size,
                samples=samples,
                dt=dt,
                b=b,
                border_region=border_region,
                avoid_walls=avoid_walls,
                ang_vel_noise_std=ang_vel_noise_std,
            )
        elif mode == "circle":
            traj = self.generate_circular_trajectory(
                box_width,
                box_height,
                batch_size,
                samples=samples,
                dt=dt,
                b=b,
                omega=omega,
                speed=speed,
                border_region=border_region,
                avoid_walls=avoid_walls,
                ang_vel_noise_std=ang_vel_noise_std,
            )
        else:
            raise ValueError("mode must be 'straight' or 'circle'")

        v = np.stack([traj['ego_v'] * np.cos(traj['target_md']),
                      traj['ego_v'] * np.sin(traj['target_md'])], axis=-1)
        v = torch.tensor(v, dtype=torch.float32).transpose(0, 1)
        s = torch.tensor(traj['ego_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)

        pos = np.stack([traj['target_x'], traj['target_y']], axis=-1)
        pos = torch.tensor(pos, dtype=torch.float32).transpose(0, 1)
        pos = pos.to(self.options.device)
        place_outputs = None
        if use_place:
            place_outputs = self.place_cells.get_activation(pos)

        init_pos = np.stack([traj['init_x'], traj['init_y']], axis=-1)
        init_pos = torch.tensor(init_pos, dtype=torch.float32)
        init_pos = init_pos.to(self.options.device)
        init_actp = None
        if use_place:
            init_actp = self.place_cells.get_activation(init_pos).squeeze(1)

        move_dir = torch.tensor(traj['target_md'], dtype=torch.float32).transpose(0, 1)
        move_dir = move_dir.to(self.options.device)
        direction_outputs = None
        if use_direction:
            direction_outputs = self.direction_cells.get_activation(move_dir)

        init_md = torch.tensor(traj['init_md'], dtype=torch.float32)
        init_md = init_md.to(self.options.device)
        init_dir_actv = None
        if use_direction:
            init_dir_actv = self.direction_cells.get_activation(init_md).squeeze(1)

        ang_v = torch.tensor(traj['ang_v'], dtype=torch.float32).transpose(0, 1).unsqueeze(-1)
        ang_v = ang_v.to(self.options.device)

        v = v.to(self.options.device)
        s = s.to(self.options.device)
        if use_place and not use_direction:
            inputs = (v, init_actp)
            return (inputs, pos, place_outputs)
        if use_direction and not use_place:
            inputs = (ang_v, init_dir_actv)
            return (inputs, direction_outputs, move_dir)

        inputs = (s, init_actp, ang_v, init_dir_actv)
        outputs = (place_outputs, direction_outputs)
        targets = (pos, move_dir)
        return (inputs, outputs, targets)
