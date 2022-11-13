import numpy as np


class PSOSolver:

    def __init__(self, n_particles=100, w=0.95, c1=0.25, c2=0.25, v=0.5, max_iters=1000, tol=1e-3, n_tol=0.7):
        self.n_particles = n_particles
        self.w = np.broadcast_to(w, (n_particles,))
        self.v = np.broadcast_to(v, (n_particles,))
        self.c1 = np.broadcast_to(c1, (n_particles,))
        self.c2 = np.broadcast_to(c2, (n_particles,))
        self.max_iters = int(max_iters)
        self.tol = tol
        self.n_tol = n_tol if isinstance(n_tol, int) else n_tol * n_particles

    def solve(self, fun, domain, *, callback=None):
        domain = np.array(domain)
        n_dim = domain.shape[0]

        # state variables
        pos = self._random_positions(domain)                    # particle positions
        vel = self._random_velocities(n_dim)                    # particle velocities
        inv = np.zeros((self.n_particles,), dtype=np.bool_)     # invalid particles (outside domain)
        lv = np.repeat(np.inf, self.n_particles)                # local minimum values
        lp = np.zeros((self.n_particles, n_dim))                # local minimum positions
        gv = np.inf                                             # global minimum value
        gp = np.zeros((n_dim,))                                 # global minimum position

        for i in range(self.max_iters):

            # updating state
            lv, lp, gv, gp = self._get_minimums(lv, lp, gv, gp, fun, pos, inv)
            vel = self._get_new_velocities(pos, vel, lp, gp, inv)
            pos = self._get_new_positions(pos, vel)
            vel, inv = self._bounce_particles_outside_domain(pos, vel, domain)

            # early stop
            if self._early_stop(pos, gp):
                break

            # optional callback
            if callback:
                callback(gp, gv, pos)

        return gp, gv

    def _random_positions(self, domain):
        dims_pos = []
        for constraint in domain:
            start, end = constraint
            dim_pos = np.random.rand(self.n_particles) * (end - start) + start
            dims_pos.append(dim_pos)
        pos = np.column_stack(dims_pos)
        return pos

    def _random_velocities(self, n_dims):
        directions = np.random.rand(self.n_particles, n_dims)
        lengths = np.sqrt(directions[:, 0]**2 + directions[:, 1]**2)
        scale_factors = self.v / lengths
        scale_factors_transformed = np.repeat(np.expand_dims(scale_factors, axis=1), 2, axis=1)
        scaled_directions = directions * scale_factors_transformed
        return scaled_directions

    def _get_minimums(self, lv, lp, gv, gp, fun, pos, inv):
        # calling function
        values = self._get_function_values(fun, pos, inv)

        # local minimums
        where_better = values < lv
        lv[where_better] = values[where_better]
        lp[where_better] = pos[where_better]

        # global minimum
        min_index = np.argmin(values)
        min_value = values[min_index]
        if min_value < gv:
            gv = min_value
            gp = pos[min_index]

        return lv, lp, gv, gp

    @staticmethod
    def _get_function_values(fun, pos, invalid_pos):
        return np.array([fun(args) if not invalid else np.inf for args, invalid in zip(pos, invalid_pos)])

    def _get_new_velocities(self, pos, vel, lp, gp, inv):
        r1, r2 = np.random.rand(2)
        g_diff = gp - pos
        l_diff = lp - pos
        main_term = vel * self.w[..., None]
        local_term = l_diff * r1 * self.c1[..., None]
        global_term = g_diff * r2 * self.c2[..., None]
        new_vel = main_term + local_term + global_term
        mixed_vel = np.where(np.repeat(inv[..., None], vel.shape[1], axis=1), vel, new_vel)
        return mixed_vel

    @staticmethod
    def _get_new_positions(pos, vel):
        return pos + vel

    def _bounce_particles_outside_domain(self, pos, vel, domain):
        invalid = np.zeros((self.n_particles,), dtype=np.bool_)
        new_vel = vel.copy()

        for dim, bounds in enumerate(domain):
            lower_bound, upper_bound = bounds

            # mark particles as invalid
            dim_invalid = (pos[:, dim] < lower_bound) | (pos[:, dim] > upper_bound)
            invalid |= dim_invalid

            # flip particles velocities
            new_vel[dim_invalid, dim] *= -1

        return new_vel, invalid

    def _early_stop(self, pos, gp):
        if self.tol is None:
            return
        diffs = pos - gp
        distances = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
        acceptable = distances < self.tol
        acceptable_count = np.sum(acceptable)
        return acceptable_count >= self.n_tol
