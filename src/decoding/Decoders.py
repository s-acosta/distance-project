import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV


class MLE:

    def __init__(self, device, cells, angle_range=None, monocular=None, distance=None):
        """
        Parameters
        ----------
        device : RadialDevice
            Device object with angle_range, distance_sequence, get_repeat_trace, get_condition_trials.
        cells : array-like
            Cell IDs to use for decoding.
        """
        self.device = device
        self.cells = np.asarray(cells)
        self.n_neurons = len(self.cells)
        self.angle_range = np.ravel(angle_range if angle_range is not None else device.angle_range)
        self.n_angles = len(self.angle_range)

        self.monocular = monocular
        self.distance = distance
        
        self.neuron_means = None
        self.neuron_variances = None

    def fit(self, train_trials):
        """Compute angle tuning curves (mean and variance) from training trials.

        Parameters
        ----------
        train_trials : array-like
            Trial indices to use for fitting.
        """
        self.neuron_means = np.zeros((self.n_neurons, self.n_angles))
        self.neuron_variances = np.zeros((self.n_neurons, self.n_angles))

        for i, cell in enumerate(self.cells):
            trial_responses = np.array([
                self.device.get_repeat_trace(cell, t) for t in train_trials
            ])
            # trial_responses shape: (n_trials, n_angles)
            self.neuron_means[i] = np.mean(trial_responses, axis=0)
            self.neuron_variances[i] = np.var(trial_responses, axis=0)

    def _log_likelihood(self, response_vector, angle_idx):
        """Gaussian log-likelihood for a given angle."""
        expected = self.neuron_means[:, angle_idx]
        variances = np.maximum(self.neuron_variances[:, angle_idx], 1e-6)
        return -0.5 * np.sum(np.log(variances) + (response_vector - expected) ** 2 / variances)

    def decode(self, response_vector):
        """Decode a single timepoint response vector to an angle.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
            Response of each neuron at one timepoint.

        Returns
        -------
        float
            Decoded angle.
        """
        ll = np.array([self._log_likelihood(response_vector, a) for a in range(self.n_angles)])
        return self.angle_range[np.argmax(ll)]

    def get_trial_responses(self, trial_idx):
        """Get response matrix for a single trial.

        Returns
        -------
        array, shape (n_neurons, n_angles)
        """
        return np.array([self.device.get_repeat_trace(cell, trial_idx) for cell in self.cells])

    def decode_trial(self, trial_idx):
        """Decode angle at each timepoint for a single trial.

        Returns
        -------
        array of decoded angles, shape (n_angles,)
        """
        responses = self.get_trial_responses(trial_idx)
        return np.array([self.decode(responses[:, i]) for i in range(responses.shape[1])])

    def decode_trials(self, trial_idxs):
        """Decode angle trajectories for multiple trials.

        Returns
        -------
        array, shape (n_trials, n_angles)
        """
        return np.array([self.decode_trial(t) for t in trial_idxs])

    def evaluate(self, actual_angles, decoded_angles):
        """Compute decoding metrics.

        Parameters
        ----------
        actual_angles : array
            True angle values.
        decoded_angles : array
            Decoded angle values (same shape as actual_angles).

        Returns
        -------
        dict with keys: mae, rmse, bias, correlation, std
        """
        errors = decoded_angles - actual_angles
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual_angles - np.mean(actual_angles)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'bias': np.mean(errors),
            'correlation': np.corrcoef(actual_angles.flatten(), decoded_angles.flatten())[0, 1],
            'r2': r2,
            'std': np.std(errors),
        }

    def cross_validate(self, trial_idxs=None):
        """Run leave-one-out cross-validation for angle decoding.

        Parameters
        ----------
        trial_idxs : array-like, optional
            Trial indices to use. If None, uses all non-control trials.

        Returns
        -------
        dict with keys:
            decoded_angles : array, shape (n_trials, n_angles)
            actual_angles  : array, shape (n_trials, n_angles)
            metrics        : dict with evaluation metrics
        """
        if trial_idxs is None:
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=self.distance)

        trial_idxs = np.asarray(trial_idxs)
        n_trials = len(trial_idxs)
        all_decoded = []

        for i in range(n_trials):
            print(f'Trial {i + 1}/{n_trials}', end='\r')

            test = trial_idxs[i]
            train = np.delete(trial_idxs, i)

            self.fit(train)
            decoded = self.decode_trial(test)
            all_decoded.append(decoded)

        decoded_angles = np.array(all_decoded)
        actual_angles = np.tile(self.angle_range, (len(decoded_angles), 1))
        metrics = self.evaluate(actual_angles, decoded_angles)

        print(f'\nTotal test trials: {len(decoded_angles)}')
        self.print_metrics(metrics)

        return {
            'decoded_angles': decoded_angles,
            'actual_angles': actual_angles,
            'metrics': metrics,
        }

    def plot_results(self, results):
        """Plot decoding results from cross_validate.

        Parameters
        ----------
        results : dict
            Output from cross_validate().
        """
        decoded = results['decoded_angles']
        actual = results['actual_angles']
        metrics = results['metrics']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Decoded vs actual (scatter density)
        ax = axes[0]
        ax.scatter(actual.flatten(), decoded.flatten(), alpha=0.1, s=1, color='k')
        ax.plot([self.angle_range[0], self.angle_range[-1]],
                [self.angle_range[0], self.angle_range[-1]], 'r--', linewidth=1)
        ax.set_xlabel('Actual angle (°)')
        ax.set_ylabel('Decoded angle (°)')
        ax.set_title(f'Decoded vs Actual (R²={metrics["r2"]:.3f})')
        ax.set_aspect('equal')

        # 2. Mean decoded trajectory
        ax = axes[1]
        mean_decoded = np.mean(decoded, axis=0)
        std_decoded = np.std(decoded, axis=0)
        ax.plot(self.angle_range, self.angle_range, 'r--', linewidth=1, label='Perfect')
        ax.plot(self.angle_range, mean_decoded, 'b-', linewidth=2, label='Mean decoded')
        ax.fill_between(self.angle_range, mean_decoded - std_decoded, mean_decoded + std_decoded,
                        alpha=0.2, color='b')
        ax.set_xlabel('Actual angle (°)')
        ax.set_ylabel('Decoded angle (°)')
        ax.set_title('Mean decoded trajectory')
        ax.legend()

        # 3. MAE per angle
        ax = axes[2]
        abs_errors = np.abs(decoded - actual)
        mae_per_angle = np.mean(abs_errors, axis=0)
        sem_per_angle = np.std(abs_errors, axis=0) / np.sqrt(abs_errors.shape[0])
        ax.plot(self.angle_range, mae_per_angle, 'b-', linewidth=2)
        ax.fill_between(self.angle_range, mae_per_angle - sem_per_angle, mae_per_angle + sem_per_angle,
                        alpha=0.2, color='b')
        ax.set_xlabel('Actual angle (°)')
        ax.set_ylabel('MAE (°)')
        ax.set_title(f'MAE per angle (overall={metrics["mae"]:.2f}°)')

        plt.tight_layout()
        plt.show()

    def plot_trial(self, trial_idx):
        """Plot decoded vs actual angle for a single trial.

        Parameters
        ----------
        trial_idx : int
            Trial index to decode and plot.
        """
        decoded = self.decode_trial(trial_idx)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.angle_range, self.angle_range, 'r--', linewidth=1, label='Actual')
        ax.plot(self.angle_range, decoded, 'b-', linewidth=2, label='Decoded')
        ax.set_xlabel('Actual angle (°)')
        ax.set_ylabel('Decoded angle (°)')
        ax.set_title(f'Trial {trial_idx}')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_likelihood_matrix(self, trial_idx, trial_idxs=None):
        """Plot log-likelihood matrix for a single trial (leave-one-out).

        Refits the model on all trials except the test trial, then computes
        likelihoods. Rows are actual angles (timepoints), columns are candidate angles.
        Diagonal should show the highest values if the decoder works well.

        Parameters
        ----------
        trial_idx : int
            Trial index to evaluate.
        trial_idxs : array-like, optional
            Full set of trial indices. If None, uses all non-control trials.
        """
        if trial_idxs is None:
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=self.distance)
        trial_idxs = np.asarray(trial_idxs)
        train = trial_idxs[trial_idxs != trial_idx]

        old_means, old_variances = self.neuron_means, self.neuron_variances
        self.fit(train)

        responses = self.get_trial_responses(trial_idx)
        ll_matrix = np.zeros((self.n_angles, self.n_angles))

        for t in range(self.n_angles):
            for a in range(self.n_angles):
                ll_matrix[t, a] = self._log_likelihood(responses[:, t], a)

        self.neuron_means, self.neuron_variances = old_means, old_variances

        # Normalize each row so the max is 0, making differences visible
        ll_matrix = ll_matrix - np.max(ll_matrix, axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(ll_matrix, aspect='auto', origin='lower',
                       extent=[self.angle_range[0], self.angle_range[-1],
                               self.angle_range[0], self.angle_range[-1]], cmap='viridis', vmin=-1000, vmax=0)
        ax.plot([self.angle_range[0], self.angle_range[-1]],
                [self.angle_range[0], self.angle_range[-1]], 'r--', linewidth=1)
        ax.set_xlabel('Candidate angle (°)')
        ax.set_ylabel('Actual angle (°)')
        condition = 'Monocular' if self.device.monocular_sequence[trial_idx] == 1 else 'Binocular'
        distance = self.device.distance_sequence[trial_idx]
        ax.set_title(f'Trial {trial_idx}, {condition}, Distance: {distance}')
        fig.colorbar(im, ax=ax, label='Log-likelihood')
        plt.tight_layout()
        plt.show()

    def plot_likelihood_matrix_sum(self, trial_idxs=None):
        """Plot summed log-likelihood matrix across all trials (leave-one-out).

        For each trial, refits the model excluding that trial, computes
        likelihoods, and accumulates them.

        Parameters
        ----------
        trial_idxs : array-like, optional
            Trial indices to sum over. If None, uses all non-control trials.
        """
        if trial_idxs is None:
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=self.distance)
        trial_idxs = np.asarray(trial_idxs)

        old_means, old_variances = self.neuron_means, self.neuron_variances
        ll_sum = np.zeros((self.n_angles, self.n_angles))

        for i, trial in enumerate(trial_idxs):
            print(f'Trial {i + 1}/{len(trial_idxs)}', end='\r')
            train = np.delete(trial_idxs, i)
            self.fit(train)
            responses = self.get_trial_responses(trial)
            for t in range(self.n_angles):
                for a in range(self.n_angles):
                    ll_sum[t, a] += self._log_likelihood(responses[:, t], a)

        self.neuron_means, self.neuron_variances = old_means, old_variances

        # Normalize each row so the max is 0
        ll_sum = ll_sum - np.max(ll_sum, axis=1, keepdims=True)

        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(ll_sum, aspect='auto', origin='lower',
                       extent=[self.angle_range[0], self.angle_range[-1],
                               self.angle_range[0], self.angle_range[-1]], cmap='viridis', vmin=-1e5, vmax=0)
        ax.plot([self.angle_range[0], self.angle_range[-1]],
                [self.angle_range[0], self.angle_range[-1]], 'r--', linewidth=1)
        ax.set_xlabel('Candidate angle (°)')
        ax.set_ylabel('Actual angle (°)')
        ax.set_title(f'Summed log-likelihood ({len(trial_idxs)} trials)')
        fig.colorbar(im, ax=ax, label='Log-likelihood')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_metrics(metrics):
        print("=" * 40)
        print(f"  MAE:          {metrics['mae']:.2f}")
        print(f"  RMSE:         {metrics['rmse']:.2f}")
        print(f"  Correlation:  {metrics['correlation']:.3f}")
        print(f"  R²:           {metrics['r2']:.3f}")
        print("=" * 40)


class MLEDistance:
    """Maximum likelihood decoder for distance (continuous output via posterior mean)."""

    def __init__(self, device, cells, monocular=None, chunk_size=1,
                 var_floor=1e-2, var_shrinkage=0.0):
        """
        Parameters
        ----------
        device : RadialDevice
        cells : array-like
            Cell IDs to use for decoding.
        monocular : int or None
            Filter trials by monocular condition.
        chunk_size : int
            Number of angular positions to pool per decoded distance.
            Default 1 decodes at each angle independently.
        var_floor : float
            Minimum variance to prevent overconfident likelihoods.
        var_shrinkage : float, 0 to 1
            Shrinkage toward pooled (distance-averaged) variance.
            0 = use per-distance variance, 1 = fully pooled across distances.
        """
        self.device = device
        self.cells = np.asarray(cells)
        self.n_neurons = len(self.cells)
        self.angle_range = np.ravel(device.angle_range)
        self.n_angles = len(self.angle_range)
        self.distance_values = np.arange(4)
        self.n_distances = 4
        self.monocular = monocular
        self.chunk_size = chunk_size
        self.n_chunks = int(np.ceil(self.n_angles / chunk_size))
        self.var_floor = var_floor
        self.var_shrinkage = var_shrinkage

        # neuron_means/variances shape: (n_neurons, n_distances, n_angles)
        self.neuron_means = None
        self.neuron_variances = None

    def fit(self, train_trials_by_distance):
        """Compute tuning curves conditioned on distance.

        Parameters
        ----------
        train_trials_by_distance : dict
            Maps distance (0-3) to array of trial indices for training.
        """
        self.neuron_means = np.zeros((self.n_neurons, self.n_distances, self.n_angles))
        self.neuron_variances = np.zeros((self.n_neurons, self.n_distances, self.n_angles))

        for dist in range(self.n_distances):
            for i, cell in enumerate(self.cells):
                trial_responses = np.array([
                    self.device.get_repeat_trace(cell, t)
                    for t in train_trials_by_distance[dist]
                ])
                self.neuron_means[i, dist] = np.mean(trial_responses, axis=0)
                self.neuron_variances[i, dist] = np.var(trial_responses, axis=0)

        if self.var_shrinkage > 0:
            # Shrink per-distance variance toward pooled (mean across distances)
            pooled = np.mean(self.neuron_variances, axis=1, keepdims=True)
            self.neuron_variances = ((1 - self.var_shrinkage) * self.neuron_variances
                                     + self.var_shrinkage * pooled)

    def _log_likelihood(self, response_vector, angle_idx, dist_idx):
        """Gaussian log-likelihood for a given (angle, distance)."""
        expected = self.neuron_means[:, dist_idx, angle_idx]
        variances = np.maximum(self.neuron_variances[:, dist_idx, angle_idx], self.var_floor)
        return -0.5 * np.sum(np.log(variances) + (response_vector - expected) ** 2 / variances)

    def get_trial_responses(self, trial_idx):
        """Get response matrix for a single trial.

        Returns
        -------
        array, shape (n_neurons, n_angles)
        """
        return np.array([self.device.get_repeat_trace(cell, trial_idx) for cell in self.cells])

    def decode(self, response_vector, angle_idx):
        """Decode distance at a single angle position using posterior mean.

        Computes log-likelihoods for each distance class, converts to
        probabilities via softmax, and returns the expected value.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
        angle_idx : int

        Returns
        -------
        float : predicted distance (continuous, between 0 and 3)
        """
        log_likelihoods = np.array([
            self._log_likelihood(response_vector, angle_idx, d)
            for d in range(self.n_distances)
        ])
        log_likelihoods = log_likelihoods / self.n_neurons
        log_likelihoods = log_likelihoods - np.max(log_likelihoods)
        probabilities = np.exp(log_likelihoods)
        probabilities = probabilities / np.sum(probabilities)
        return np.sum(self.distance_values * probabilities)

    def _get_chunk_indices(self):
        """Return list of arrays, each containing angle indices for one chunk."""
        return [np.arange(i * self.chunk_size,
                          min((i + 1) * self.chunk_size, self.n_angles))
                for i in range(self.n_chunks)]

    def decode_response_matrix(self, responses):
        """Decode distance at each angle position (or chunk) from a response matrix.

        Parameters
        ----------
        responses : array, shape (n_neurons, n_angles)
            Neural response matrix (single trial or averaged across trials).

        Returns
        -------
        list of float : predicted distance per angle (chunk_size=1) or per chunk
        """
        if self.chunk_size == 1:
            return [
                self.decode(responses[:, angle_idx], angle_idx)
                for angle_idx in range(responses.shape[1])
            ]

        chunks = self._get_chunk_indices()
        decoded = []
        for angle_idxs in chunks:
            ll = np.zeros(self.n_distances)
            for d in range(self.n_distances):
                for a in angle_idxs:
                    ll[d] += self._log_likelihood(responses[:, a], a, d)
            ll = ll / (self.n_neurons * len(angle_idxs))
            ll = ll - np.max(ll)
            probabilities = np.exp(ll)
            probabilities = probabilities / np.sum(probabilities)
            decoded.append(np.sum(self.distance_values * probabilities))
        return decoded

    def decode_trial(self, trial_idx):
        """Decode distance at each angle position (or chunk) for a single trial.

        When chunk_size=1, decodes independently at each angle.
        When chunk_size>1, pools log-likelihoods across angles in each chunk.

        Returns
        -------
        list of float : predicted distance per angle (chunk_size=1) or per chunk
        """
        responses = self.get_trial_responses(trial_idx)
        return self.decode_response_matrix(responses)

    def _build_kfold_splits(self, k_folds):
        """Build stratified k-fold splits (non-overlapping test sets).

        Shuffles trials within each distance, splits into k folds,
        so every trial appears in exactly one test fold.

        Returns
        -------
        list of (train_trials_by_distance, test_trials) for each fold
            train_trials_by_distance : dict mapping distance (0-3) to train indices
            test_trials : list of (trial_idx, distance) tuples
        """
        # Shuffle and split each distance into k folds
        folds_by_dist = {}
        for dist in range(self.n_distances):
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=dist)
            trial_idxs = np.random.permutation(trial_idxs)
            folds_by_dist[dist] = np.array_split(trial_idxs, k_folds)

        splits = []
        for fold in range(k_folds):
            train_by_dist = {}
            test_trials = []
            for dist in range(self.n_distances):
                test_idxs = folds_by_dist[dist][fold]
                train_idxs = np.concatenate([
                    folds_by_dist[dist][f] for f in range(k_folds) if f != fold
                ])
                train_by_dist[dist] = train_idxs
                test_trials.extend([(t, dist) for t in test_idxs])
            splits.append((train_by_dist, test_trials))

        return splits

    def cross_validate(self, k_folds=10):
        """Run stratified k-fold cross-validation for distance decoding.

        Each trial appears in exactly one test fold.

        Parameters
        ----------
        k_folds : int
            Number of cross-validation folds.

        Returns
        -------
        dict with keys:
            all_decoded       : array of decoded distances (flat)
            all_actual        : array of actual distances (flat)
            all_angle_indices : array of angle indices (flat)
            metrics           : dict with evaluation metrics
        """
        splits = self._build_kfold_splits(k_folds)
        all_decoded = []
        all_actual = []
        all_angle_indices = []
        n_output = self.n_angles if self.chunk_size == 1 else self.n_chunks

        for fold, (train_by_dist, test_trials) in enumerate(splits):
            print(f'=== Fold {fold + 1}/{k_folds} ===', end='\r', flush=True)

            self.fit(train_by_dist)

            for test_idx, actual_dist in test_trials:
                decoded_trajectory = self.decode_trial(test_idx)
                all_decoded.extend(decoded_trajectory)
                all_actual.extend([actual_dist] * len(decoded_trajectory))
                all_angle_indices.extend(range(n_output))

        all_decoded = np.array(all_decoded)
        all_actual = np.array(all_actual)
        all_angle_indices = np.array(all_angle_indices)

        metrics = self.evaluate(all_actual, all_decoded)

        print(f'\nTotal predictions: {len(all_decoded)}')
        print(f'Predictions per angle: {len(all_decoded) // n_output}')
        self.print_metrics(metrics)

        return {
            'all_decoded': all_decoded,
            'all_actual': all_actual,
            'all_angle_indices': all_angle_indices,
            'metrics': metrics,
        }

    def cross_validate_mean(self, n_repeats=100):
        """Run repeated 50/50 split cross-validation, decoding from mean test responses.

        For each repeat, stratified 50/50 train/test split per distance.
        Test trial responses are averaged per distance class before decoding.

        Parameters
        ----------
        n_repeats : int
            Number of random train/test splits. Default 100.

        Returns
        -------
        dict with keys:
            all_decoded       : array of decoded distances (flat)
            all_actual        : array of actual distances (flat)
            all_angle_indices : array of angle indices (flat)
            metrics           : dict with evaluation metrics
        """
        all_decoded = []
        all_actual = []
        all_angle_indices = []
        n_output = self.n_angles if self.chunk_size == 1 else self.n_chunks

        # Get all trial indices per distance once
        trials_by_dist = {}
        for dist in range(self.n_distances):
            trials_by_dist[dist] = np.array(
                self.device.get_condition_trials(monocular=self.monocular, distance=dist)
            )

        for rep in range(n_repeats):
            print(f'Repeat {rep + 1}/{n_repeats}', end='\r', flush=True)

            train_by_dist = {}
            test_by_dist = {}

            for dist in range(self.n_distances):
                idxs = np.random.permutation(trials_by_dist[dist])
                half = len(idxs) // 2
                train_by_dist[dist] = idxs[:half]
                test_by_dist[dist] = idxs[half:]

            self.fit(train_by_dist)

            for dist in range(self.n_distances):
                # Average responses across test trials for this distance
                test_responses = np.array([
                    self.get_trial_responses(t) for t in test_by_dist[dist]
                ])  # shape: (n_test_trials, n_neurons, n_angles)
                mean_responses = np.mean(test_responses, axis=0)  # (n_neurons, n_angles)

                decoded_trajectory = self.decode_response_matrix(mean_responses)
                all_decoded.extend(decoded_trajectory)
                all_actual.extend([dist] * len(decoded_trajectory))
                all_angle_indices.extend(range(n_output))

        all_decoded = np.array(all_decoded)
        all_actual = np.array(all_actual)
        all_angle_indices = np.array(all_angle_indices)

        metrics = self.evaluate(all_actual, all_decoded)

        print(f'\nTotal predictions: {len(all_decoded)}')
        print(f'Repeats: {n_repeats}, Distances: {self.n_distances}')
        self.print_metrics(metrics)

        return {
            'all_decoded': all_decoded,
            'all_actual': all_actual,
            'all_angle_indices': all_angle_indices,
            'metrics': metrics,
        }

    def evaluate(self, actual, decoded):
        """Compute distance decoding metrics.

        Parameters
        ----------
        actual : array
            True distance values.
        decoded : array
            Decoded distance values (same shape as actual).

        Returns
        -------
        dict with keys: mae, rmse, bias, std, correlation
        """
        actual = np.array(actual)
        decoded = np.array(decoded)
        errors = decoded - actual
        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'bias': np.mean(errors),
            'std': np.std(errors),
            'correlation': np.corrcoef(actual.flatten(), decoded.flatten())[0, 1],
        }

    def _get_chunk_centers(self):
        """Return the mean angle for each chunk, for plotting."""
        return np.array([np.mean(self.angle_range[idx]) for idx in self._get_chunk_indices()])

    def plot_results(self, results):
        """Plot MAE vs angle position (or chunk center) for distance decoder.

        Parameters
        ----------
        results : dict
            Output from cross_validate().
        """
        actual = np.array(results['all_actual'])
        decoded = np.array(results['all_decoded'])
        angle_indices = np.array(results['all_angle_indices'])
        metrics = results['metrics']
        errors = decoded - actual

        fig, ax = plt.subplots(figsize=(10, 6))

        if self.chunk_size == 1:
            x_vals = self.angle_range
            n_positions = self.n_angles
        else:
            x_vals = self._get_chunk_centers()
            n_positions = self.n_chunks

        mae_by_pos = np.zeros(n_positions)
        for idx in range(n_positions):
            mask = angle_indices == idx
            if mask.sum() > 0:
                mae_by_pos[idx] = np.mean(np.abs(errors[mask]))

        ax.plot(x_vals, mae_by_pos, 'b-', linewidth=2)
        ax.axhline(metrics['mae'], color='g', linestyle='--', alpha=0.7,
                   label=f'Overall MAE = {metrics["mae"]:.2f}')
        ax.axhline(1.25, color='r', linestyle=':', alpha=0.5, label='Random guess MAE ~1.25')
        ax.set_xlabel('Angle Position (deg)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Distance Decoding Error vs Angle')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, metrics

    def plot_results_comparison(self, results_list, labels):
        """Plot MAE vs angle for multiple conditions overlaid on the same axes.

        Parameters
        ----------
        results_list : list of dict
            Each element is an output from cross_validate().
        labels : list of str
            Label for each result (e.g. ['Monocular', 'Binocular']).
        """
        colors = ['b', 'r', 'g', 'orange', 'purple']
        fig, ax = plt.subplots(figsize=(10, 6))

        if self.chunk_size == 1:
            x_vals = self.angle_range
            n_positions = self.n_angles
        else:
            x_vals = self._get_chunk_centers()
            n_positions = self.n_chunks

        for i, (results, label) in enumerate(zip(results_list, labels)):
            actual = np.array(results['all_actual'])
            decoded = np.array(results['all_decoded'])
            angle_indices = np.array(results['all_angle_indices'])
            metrics = results['metrics']
            errors = decoded - actual
            c = colors[i % len(colors)]

            mae_by_pos = np.zeros(n_positions)
            for idx in range(n_positions):
                mask = angle_indices == idx
                if mask.sum() > 0:
                    mae_by_pos[idx] = np.mean(np.abs(errors[mask]))

            ax.plot(x_vals, mae_by_pos, '-', color=c, linewidth=2,
                    label=f'{label} (MAE={metrics["mae"]:.2f}, r={metrics["correlation"]:.3f})')

        ax.axhline(1.25, color='k', linestyle=':', alpha=0.5, label='Random ~1.25')
        ax.set_xlabel('Angle Position (deg)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Distance Decoding Error vs Angle')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trial(self, trial_idx, trial_idxs=None):
        """Plot decoded distance vs angle for a single trial (leave-one-out).

        Refits the model on all trials except the test trial, then decodes
        and plots the result against the actual distance.

        Parameters
        ----------
        trial_idx : int
            Trial index to evaluate.
        trial_idxs : array-like, optional
            Full set of trial indices. If None, uses all non-control trials.
        """
        if trial_idxs is None:
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=None)
        trial_idxs = np.asarray(trial_idxs)

        train_by_dist = {}
        for dist in range(self.n_distances):
            train_by_dist[dist] = np.array([
                t for t in trial_idxs
                if t != trial_idx and self.device.distance_sequence[t] == dist
            ])

        old_means, old_variances = self.neuron_means, self.neuron_variances
        self.fit(train_by_dist)
        decoded = self.decode_trial(trial_idx)
        self.neuron_means, self.neuron_variances = old_means, old_variances

        actual_dist = self.device.distance_sequence[trial_idx]

        if self.chunk_size == 1:
            x_vals = self.angle_range
        else:
            x_vals = self._get_chunk_centers()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_vals, decoded, 'bo-', linewidth=2, markersize=4, label='Decoded')
        ax.axhline(actual_dist, color='r', linestyle='--', linewidth=1,
                   label=f'Actual = {actual_dist}')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Decoded distance')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlim(self.angle_range[0], self.angle_range[-1])
        condition = 'Monocular' if self.device.monocular_sequence[trial_idx] == 1 else 'Binocular'
        ax.set_title(f'Trial {trial_idx}, {condition}, Distance={actual_dist}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_trial_likelihood(self, trial_idx, trial_idxs=None):
        """Plot posterior probability matrix for a single trial (leave-one-out).

        Shows a heatmap of P(distance | response) at each angle position.
        Overlays the decoded (posterior mean) and actual distance.

        Parameters
        ----------
        trial_idx : int
            Trial index to evaluate.
        trial_idxs : array-like, optional
            Full set of trial indices. If None, uses all non-control trials.
        """
        if trial_idxs is None:
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=None)
        trial_idxs = np.asarray(trial_idxs)

        train_by_dist = {}
        for dist in range(self.n_distances):
            train_by_dist[dist] = np.array([
                t for t in trial_idxs
                if t != trial_idx and self.device.distance_sequence[t] == dist
            ])

        old_means, old_variances = self.neuron_means, self.neuron_variances
        self.fit(train_by_dist)

        responses = self.get_trial_responses(trial_idx)

        if self.chunk_size == 1:
            n_positions = self.n_angles
            x_vals = self.angle_range
            prob_matrix = np.zeros((self.n_distances, n_positions))
            decoded = np.zeros(n_positions)
            for a in range(n_positions):
                ll = np.array([self._log_likelihood(responses[:, a], a, d)
                               for d in range(self.n_distances)])
                ll = ll / self.n_neurons
                ll = ll - np.max(ll)
                probs = np.exp(ll)
                probs = probs / np.sum(probs)
                prob_matrix[:, a] = probs
                decoded[a] = np.sum(self.distance_values * probs)
        else:
            chunks = self._get_chunk_indices()
            n_positions = self.n_chunks
            x_vals = self._get_chunk_centers()
            prob_matrix = np.zeros((self.n_distances, n_positions))
            decoded = np.zeros(n_positions)
            for c, angle_idxs in enumerate(chunks):
                ll = np.zeros(self.n_distances)
                for d in range(self.n_distances):
                    for a in angle_idxs:
                        ll[d] += self._log_likelihood(responses[:, a], a, d)
                ll = ll / (self.n_neurons * len(angle_idxs))
                ll = ll - np.max(ll)
                probs = np.exp(ll)
                probs = probs / np.sum(probs)
                prob_matrix[:, c] = probs
                decoded[c] = np.sum(self.distance_values * probs)

        self.neuron_means, self.neuron_variances = old_means, old_variances

        actual_dist = self.device.distance_sequence[trial_idx]
        condition = 'Monocular' if self.device.monocular_sequence[trial_idx] == 1 else 'Binocular'

        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(prob_matrix, aspect='auto', origin='lower',
                       extent=[x_vals[0], x_vals[-1], -0.5, 3.5],
                       cmap='hot', vmin=0, vmax=1)
        ax.plot(x_vals, decoded, 'c-', linewidth=2, label='Decoded')
        ax.axhline(actual_dist, color='w', linestyle='--', linewidth=1,
                   label=f'Actual = {actual_dist}')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Distance')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_title(f'Trial {trial_idx}, {condition}, Distance={actual_dist}')
        ax.legend(loc='upper right')
        fig.colorbar(im, ax=ax, label='P(distance | response)')
        plt.tight_layout()
        return fig

    @staticmethod
    def print_metrics(metrics):
        print("=" * 50)
        print("DISTANCE DECODER PERFORMANCE")
        print("=" * 50)
        print(f"  Mean Absolute Error:  {metrics['mae']:.3f}")
        print(f"  RMSE:                 {metrics['rmse']:.3f}")
        print(f"  Bias:                 {metrics['bias']:.3f}")
        print(f"  Correlation:          {metrics['correlation']:.3f}")
        print()
        print("  Reference (random guess MAE ~ 1.25)")
        print("=" * 50)


class RidgeDistance:
    """Ridge (L2-regularized) linear regression decoder for distance."""

    def __init__(self, device, cells, monocular=None, chunk_size=1,
                 n_alphas=50, cv=5):
        """
        Parameters
        ----------
        device : RadialDevice
        cells : array-like
            Cell IDs to use for decoding.
        monocular : int or None
            Filter trials by monocular condition.
        chunk_size : int
            Number of angular positions pooled per model.
            Default 1 fits one Ridge per angle position.
        n_alphas : int
            Number of alpha values to try during CV. Default 50.
        cv : int
            Number of internal CV folds for alpha selection. Default 5.
        """
        self.device = device
        self.cells = np.asarray(cells)
        self.n_neurons = len(self.cells)
        self.angle_range = np.ravel(device.angle_range)
        self.n_angles = len(self.angle_range)
        self.distance_values = np.arange(4)
        self.n_distances = 4
        self.monocular = monocular
        self.chunk_size = chunk_size
        self.n_chunks = int(np.ceil(self.n_angles / chunk_size))
        self.n_alphas = n_alphas
        self.cv = cv

        self.models_ = None
        self.residual_stds_ = None

    def fit(self, train_trials_by_distance):
        """Fit one Ridge model per angle position (or chunk).

        Parameters
        ----------
        train_trials_by_distance : dict
            Maps distance (0-3) to array of trial indices for training.
        """
        # Collect all (trial_idx, distance_label) pairs
        all_trials = []
        for dist in range(self.n_distances):
            for t in train_trials_by_distance[dist]:
                all_trials.append((t, dist))

        # Pre-fetch all response matrices: (n_trials, n_neurons, n_angles)
        responses = np.array([self.get_trial_responses(t) for t, _ in all_trials])
        y = np.array([dist for _, dist in all_trials], dtype=float)

        n_models = self.n_angles if self.chunk_size == 1 else self.n_chunks
        self.models_ = [None] * n_models
        self.residual_stds_ = [0.0] * n_models

        if self.chunk_size == 1:
            for angle_idx in range(self.n_angles):
                X = responses[:, :, angle_idx]  # (n_trials, n_neurons)
                model = RidgeCV(alphas=np.logspace(-3, 3, self.n_alphas), cv=self.cv)
                model.fit(X, y)
                self.models_[angle_idx] = model
                self.residual_stds_[angle_idx] = np.std(y - model.predict(X))
        else:
            chunks = self._get_chunk_indices()
            for c, angle_idxs in enumerate(chunks):
                X = np.mean(responses[:, :, angle_idxs], axis=2)  # (n_trials, n_neurons)
                model = RidgeCV(alphas=np.logspace(-3, 3, self.n_alphas), cv=self.cv)
                model.fit(X, y)
                self.models_[c] = model
                self.residual_stds_[c] = np.std(y - model.predict(X))

    def get_trial_responses(self, trial_idx):
        """Get response matrix for a single trial.

        Returns
        -------
        array, shape (n_neurons, n_angles)
        """
        return np.array([self.device.get_repeat_trace(cell, trial_idx) for cell in self.cells])

    def decode(self, response_vector, angle_idx):
        """Decode distance at a single angle position (or chunk index).

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
        angle_idx : int
            Angle index (chunk_size=1) or chunk index (chunk_size>1).

        Returns
        -------
        float : predicted distance (raw, no clipping)
        """
        model = self.models_[angle_idx]
        return model.predict(response_vector.reshape(1, -1))[0]

    def _get_chunk_indices(self):
        """Return list of arrays, each containing angle indices for one chunk."""
        return [np.arange(i * self.chunk_size,
                          min((i + 1) * self.chunk_size, self.n_angles))
                for i in range(self.n_chunks)]

    def _get_chunk_centers(self):
        """Return the mean angle for each chunk, for plotting."""
        return np.array([np.mean(self.angle_range[idx]) for idx in self._get_chunk_indices()])

    def decode_trial(self, trial_idx):
        """Decode distance at each angle position (or chunk) for a single trial.

        Returns
        -------
        list of float : predicted distance per angle or per chunk
        """
        responses = self.get_trial_responses(trial_idx)

        if self.chunk_size == 1:
            return [
                self.decode(responses[:, angle_idx], angle_idx)
                for angle_idx in range(self.n_angles)
            ]

        chunks = self._get_chunk_indices()
        decoded = []
        for c, angle_idxs in enumerate(chunks):
            avg_response = np.mean(responses[:, angle_idxs], axis=1)
            decoded.append(self.decode(avg_response, c))
        return decoded

    def _build_kfold_splits(self, k_folds):
        """Build stratified k-fold splits (non-overlapping test sets).

        Returns
        -------
        list of (train_trials_by_distance, test_trials) for each fold
        """
        folds_by_dist = {}
        for dist in range(self.n_distances):
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=dist)
            trial_idxs = np.random.permutation(trial_idxs)
            folds_by_dist[dist] = np.array_split(trial_idxs, k_folds)

        splits = []
        for fold in range(k_folds):
            train_by_dist = {}
            test_trials = []
            for dist in range(self.n_distances):
                test_idxs = folds_by_dist[dist][fold]
                train_idxs = np.concatenate([
                    folds_by_dist[dist][f] for f in range(k_folds) if f != fold
                ])
                train_by_dist[dist] = train_idxs
                test_trials.extend([(t, dist) for t in test_idxs])
            splits.append((train_by_dist, test_trials))

        return splits

    def cross_validate(self, k_folds=10):
        """Run stratified k-fold cross-validation for distance decoding.

        Parameters
        ----------
        k_folds : int
            Number of cross-validation folds.

        Returns
        -------
        dict with keys:
            all_decoded       : array of decoded distances (flat)
            all_actual        : array of actual distances (flat)
            all_angle_indices : array of angle indices (flat)
            metrics           : dict with evaluation metrics
        """
        splits = self._build_kfold_splits(k_folds)
        all_decoded = []
        all_actual = []
        all_angle_indices = []
        n_output = self.n_angles if self.chunk_size == 1 else self.n_chunks

        for fold, (train_by_dist, test_trials) in enumerate(splits):
            print(f'=== Fold {fold + 1}/{k_folds} ===', end='\r', flush=True)

            self.fit(train_by_dist)

            for test_idx, actual_dist in test_trials:
                decoded_trajectory = self.decode_trial(test_idx)
                all_decoded.extend(decoded_trajectory)
                all_actual.extend([actual_dist] * len(decoded_trajectory))
                all_angle_indices.extend(range(n_output))

        all_decoded = np.array(all_decoded)
        all_actual = np.array(all_actual)
        all_angle_indices = np.array(all_angle_indices)

        metrics = self.evaluate(all_actual, all_decoded)

        print(f'\nTotal predictions: {len(all_decoded)}')
        print(f'Predictions per angle: {len(all_decoded) // n_output}')
        self.print_metrics(metrics)

        return {
            'all_decoded': all_decoded,
            'all_actual': all_actual,
            'all_angle_indices': all_angle_indices,
            'metrics': metrics,
        }

    def evaluate(self, actual, decoded):
        """Compute distance decoding metrics.

        Returns
        -------
        dict with keys: mae, rmse, bias, std, correlation
        """
        actual = np.array(actual)
        decoded = np.array(decoded)
        errors = decoded - actual
        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'bias': np.mean(errors),
            'std': np.std(errors),
            'correlation': np.corrcoef(actual.flatten(), decoded.flatten())[0, 1],
        }

    def plot_results(self, results):
        """Plot MAE vs angle position for distance decoder.

        Parameters
        ----------
        results : dict
            Output from cross_validate().
        """
        actual = np.array(results['all_actual'])
        decoded = np.array(results['all_decoded'])
        angle_indices = np.array(results['all_angle_indices'])
        metrics = results['metrics']
        errors = decoded - actual

        fig, ax = plt.subplots(figsize=(10, 6))

        if self.chunk_size == 1:
            x_vals = self.angle_range
            n_positions = self.n_angles
        else:
            x_vals = self._get_chunk_centers()
            n_positions = self.n_chunks

        mae_by_pos = np.zeros(n_positions)
        for idx in range(n_positions):
            mask = angle_indices == idx
            if mask.sum() > 0:
                mae_by_pos[idx] = np.mean(np.abs(errors[mask]))

        ax.plot(x_vals, mae_by_pos, 'b-', linewidth=2)
        ax.axhline(metrics['mae'], color='g', linestyle='--', alpha=0.7,
                   label=f'Overall MAE = {metrics["mae"]:.2f}')
        ax.axhline(1.25, color='r', linestyle=':', alpha=0.5, label='Random guess MAE ~1.25')
        ax.set_xlabel('Angle Position (deg)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Distance Decoding Error vs Angle (Ridge)')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, metrics

    def plot_results_comparison(self, results_list, labels):
        """Plot MAE vs angle for multiple conditions overlaid.

        Parameters
        ----------
        results_list : list of dict
            Each element is an output from cross_validate().
        labels : list of str
            Label for each result.
        """
        colors = ['b', 'r', 'g', 'orange', 'purple']
        fig, ax = plt.subplots(figsize=(10, 6))

        if self.chunk_size == 1:
            x_vals = self.angle_range
            n_positions = self.n_angles
        else:
            x_vals = self._get_chunk_centers()
            n_positions = self.n_chunks

        for i, (results, label) in enumerate(zip(results_list, labels)):
            actual = np.array(results['all_actual'])
            decoded = np.array(results['all_decoded'])
            angle_indices = np.array(results['all_angle_indices'])
            metrics = results['metrics']
            errors = decoded - actual
            c = colors[i % len(colors)]

            mae_by_pos = np.zeros(n_positions)
            for idx in range(n_positions):
                mask = angle_indices == idx
                if mask.sum() > 0:
                    mae_by_pos[idx] = np.mean(np.abs(errors[mask]))

            ax.plot(x_vals, mae_by_pos, '-', color=c, linewidth=2,
                    label=f'{label} (MAE={metrics["mae"]:.2f}, r={metrics["correlation"]:.3f})')

        ax.axhline(1.25, color='k', linestyle=':', alpha=0.5, label='Random ~1.25')
        ax.set_xlabel('Angle Position (deg)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Distance Decoding Error vs Angle (Ridge)')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_trial(self, trial_idx, trial_idxs=None):
        """Plot decoded distance vs angle for a single trial (leave-one-out).

        Parameters
        ----------
        trial_idx : int
            Trial index to evaluate.
        trial_idxs : array-like, optional
            Full set of trial indices. If None, uses all non-control trials.
        """
        if trial_idxs is None:
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=None)
        trial_idxs = np.asarray(trial_idxs)

        train_by_dist = {}
        for dist in range(self.n_distances):
            train_by_dist[dist] = np.array([
                t for t in trial_idxs
                if t != trial_idx and self.device.distance_sequence[t] == dist
            ])

        old_models = self.models_
        old_residual_stds = self.residual_stds_
        self.fit(train_by_dist)
        decoded = self.decode_trial(trial_idx)
        self.models_ = old_models
        self.residual_stds_ = old_residual_stds

        actual_dist = self.device.distance_sequence[trial_idx]

        if self.chunk_size == 1:
            x_vals = self.angle_range
        else:
            x_vals = self._get_chunk_centers()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_vals, decoded, 'bo-', linewidth=2, markersize=4, label='Decoded')
        ax.axhline(actual_dist, color='r', linestyle='--', linewidth=1,
                   label=f'Actual = {actual_dist}')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Decoded distance')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlim(self.angle_range[0], self.angle_range[-1])
        condition = 'Monocular' if self.device.monocular_sequence[trial_idx] == 1 else 'Binocular'
        ax.set_title(f'Trial {trial_idx}, {condition}, Distance={actual_dist} (Ridge)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def plot_trial_likelihood(self, trial_idx, trial_idxs=None):
        """Plot decoded distance with confidence band for a single trial.

        Shows the Lasso prediction at each angle with a shaded band
        of +/- 1 training residual standard deviation.

        Parameters
        ----------
        trial_idx : int
            Trial index to evaluate.
        trial_idxs : array-like, optional
            Full set of trial indices. If None, uses all non-control trials.
        """
        if trial_idxs is None:
            trial_idxs = self.device.get_condition_trials(monocular=self.monocular, distance=None)
        trial_idxs = np.asarray(trial_idxs)

        train_by_dist = {}
        for dist in range(self.n_distances):
            train_by_dist[dist] = np.array([
                t for t in trial_idxs
                if t != trial_idx and self.device.distance_sequence[t] == dist
            ])

        old_models = self.models_
        old_residual_stds = self.residual_stds_
        self.fit(train_by_dist)
        decoded = np.array(self.decode_trial(trial_idx))
        residual_stds = np.array(self.residual_stds_)
        self.models_ = old_models
        self.residual_stds_ = old_residual_stds

        actual_dist = self.device.distance_sequence[trial_idx]
        condition = 'Monocular' if self.device.monocular_sequence[trial_idx] == 1 else 'Binocular'

        if self.chunk_size == 1:
            x_vals = self.angle_range
        else:
            x_vals = self._get_chunk_centers()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x_vals, decoded, 'bo-', linewidth=2, markersize=4, label='Decoded')
        ax.fill_between(x_vals,
                        decoded - residual_stds,
                        decoded + residual_stds,
                        alpha=0.2, color='b', label='Training residual \u00b11 SD')
        ax.axhline(actual_dist, color='r', linestyle='--', linewidth=1,
                   label=f'Actual = {actual_dist}')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Decoded distance')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlim(self.angle_range[0], self.angle_range[-1])
        ax.set_title(f'Trial {trial_idx}, {condition}, Distance={actual_dist} (Ridge)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def print_metrics(metrics):
        print("=" * 50)
        print("DISTANCE DECODER PERFORMANCE (Ridge)")
        print("=" * 50)
        print(f"  Mean Absolute Error:  {metrics['mae']:.3f}")
        print(f"  RMSE:                 {metrics['rmse']:.3f}")
        print(f"  Bias:                 {metrics['bias']:.3f}")
        print(f"  Correlation:          {metrics['correlation']:.3f}")
        print()
        print("  Reference (random guess MAE ~ 1.25)")
        print("=" * 50)

