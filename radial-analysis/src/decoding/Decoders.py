import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedKFold, KFold


class MLE:
    """Maximum Likelihood Estimation decoder for angle from neural population responses.

    A general-purpose decoder that works directly with neural response arrays.
    Compatible with SessionAggregator.get_decoder_data() output.

    Parameters
    ----------
    angle_range : array-like
        Ground truth angle values for each position, shape (n_angles,).
    var_floor : float, optional
        Minimum variance to prevent numerical instability. Default 1e-6.

    Attributes
    ----------
    neuron_means : np.ndarray or None
        Fitted mean responses, shape (n_neurons, n_angles).
    neuron_variances : np.ndarray or None
        Fitted variances, shape (n_neurons, n_angles).
    n_neurons : int
        Number of neurons (set after fit).
    n_angles : int
        Number of angle positions.
    """

    def __init__(self, angle_range, var_floor=1e-6):
        self.angle_range = np.ravel(angle_range)
        self.n_angles = len(self.angle_range)
        self.var_floor = var_floor

        self.neuron_means = None
        self.neuron_variances = None
        self.n_neurons = None

    def fit(self, X_train):
        """Fit the decoder by computing mean and variance per neuron per angle.

        Parameters
        ----------
        X_train : np.ndarray
            Training data, shape (n_neurons, n_angles, n_trials).

        Returns
        -------
        self : MLE
            Fitted decoder instance.
        """
        # X_train: (n_neurons, n_angles, n_trials)
        self.n_neurons = X_train.shape[0]
        # Mean across trials for each neuron at each angle
        self.neuron_means = np.mean(X_train, axis=2)  # (n_neurons, n_angles)
        self.neuron_variances = np.var(X_train, axis=2)  # (n_neurons, n_angles)
        return self

    def _log_likelihood(self, response_vector, angle_idx):
        """Gaussian log-likelihood for a given angle."""
        expected = self.neuron_means[:, angle_idx]
        variances = np.maximum(self.neuron_variances[:, angle_idx], self.var_floor)
        return -0.5 * np.sum(np.log(variances) + (response_vector - expected) ** 2 / variances)

    def decode(self, response_vector):
        """Decode a single timepoint response vector to an angle.

        Parameters
        ----------
        response_vector : np.ndarray, shape (n_neurons,)
            Response of each neuron at one timepoint.

        Returns
        -------
        float
            Decoded angle.
        """
        ll = np.array([self._log_likelihood(response_vector, a) for a in range(self.n_angles)])
        return self.angle_range[np.argmax(ll)]

    def decode_trial(self, trial_data):
        """Decode angle at each timepoint for a single trial.

        Parameters
        ----------
        trial_data : np.ndarray, shape (n_neurons, n_angles)
            Neural responses for one trial.

        Returns
        -------
        np.ndarray, shape (n_angles,)
            Decoded angles for each timepoint.
        """
        return np.array([self.decode(trial_data[:, i]) for i in range(trial_data.shape[1])])

    def decode_trials(self, X):
        """Decode angle trajectories for multiple trials.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Neural responses for multiple trials.

        Returns
        -------
        np.ndarray, shape (n_trials, n_angles)
            Decoded angles for each trial.
        """
        n_trials = X.shape[2]
        return np.array([self.decode_trial(X[:, :, t]) for t in range(n_trials)])

    def cross_validate(self, X, n_folds=5, stratify_labels=None, random_state=None,
                       average_test=False):
        """Run k-fold cross-validation.

        Splits trials into n_folds, trains on k-1 folds, tests on 1.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Full dataset.
        n_folds : int
            Number of CV folds.
        stratify_labels : np.ndarray, optional
            Labels to stratify by (e.g., distance), shape (n_trials,).
            Ensures each fold has same proportion of each label.
        random_state : int, optional
            Random seed for reproducibility.
        average_test : bool, optional
            If True, average test trials within each fold and decode the mean.
            Returns one decoded trajectory per fold. Default False.

        Returns
        -------
        dict
            If average_test=False:
                decoded_angles : np.ndarray, shape (n_trials, n_angles)
                actual_angles : np.ndarray, shape (n_trials, n_angles)
            If average_test=True:
                decoded_angles : np.ndarray, shape (n_folds, n_angles)
                actual_angles : np.ndarray, shape (n_folds, n_angles)
            metrics : dict with mae, rmse, bias, correlation, r2, std
        """
        n_trials = X.shape[2]

        if stratify_labels is not None:
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kfold.split(np.zeros(n_trials), stratify_labels))
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
            splits = list(kfold.split(np.zeros(n_trials)))

        if average_test:
            # Average test trials per fold, decode the mean
            all_decoded = []
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f'Fold {fold_idx + 1}/{n_folds}', end='\r')

                X_train = X[:, :, train_idx]
                X_test = X[:, :, test_idx]

                self.fit(X_train)
                # Average across test trials -> (n_neurons, n_angles)
                X_test_mean = np.mean(X_test, axis=2)
                decoded = self.decode_trial(X_test_mean)
                all_decoded.append(decoded)

            decoded_angles = np.array(all_decoded)  # (n_folds, n_angles)
            actual_angles = np.tile(self.angle_range, (n_folds, 1))
        else:
            # Decode each trial individually
            all_decoded = np.zeros((n_trials, self.n_angles))
            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f'Fold {fold_idx + 1}/{n_folds}', end='\r')

                X_train = X[:, :, train_idx]
                X_test = X[:, :, test_idx]

                self.fit(X_train)
                decoded = self.decode_trials(X_test)
                all_decoded[test_idx] = decoded

            decoded_angles = all_decoded
            actual_angles = np.tile(self.angle_range, (n_trials, 1))

        metrics = self.evaluate(actual_angles, decoded_angles)

        print(f'\nTotal test samples: {len(decoded_angles)}')
        self.print_metrics(metrics)

        return {
            'decoded_angles': decoded_angles,
            'actual_angles': actual_angles,
            'metrics': metrics,
        }

    def cross_validate_loo(self, X):
        """Run leave-one-out cross-validation.

        Trains on all but 1 trial, tests on that 1 trial, repeats for every trial.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Full dataset.

        Returns
        -------
        dict
            Same format as cross_validate().
        """
        n_trials = X.shape[2]
        all_decoded = []

        for i in range(n_trials):
            print(f'Trial {i + 1}/{n_trials}', end='\r')

            # Leave one out
            train_idx = np.concatenate([np.arange(0, i), np.arange(i + 1, n_trials)])
            X_train = X[:, :, train_idx]
            X_test = X[:, :, i]  # (n_neurons, n_angles)

            self.fit(X_train)
            decoded = self.decode_trial(X_test)
            all_decoded.append(decoded)

        decoded_angles = np.array(all_decoded)
        actual_angles = np.tile(self.angle_range, (n_trials, 1))
        metrics = self.evaluate(actual_angles, decoded_angles)

        print(f'\nTotal test trials: {n_trials}')
        self.print_metrics(metrics)

        return {
            'decoded_angles': decoded_angles,
            'actual_angles': actual_angles,
            'metrics': metrics,
        }

    @staticmethod
    def evaluate(actual_angles, decoded_angles):
        """Compute decoding metrics.

        Parameters
        ----------
        actual_angles : np.ndarray
            True angle values.
        decoded_angles : np.ndarray
            Decoded angle values (same shape as actual_angles).

        Returns
        -------
        dict
            mae, rmse, bias, correlation, r2, std
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

    def plot_results(self, results):
        """Plot decoding results from cross_validate.

        Parameters
        ----------
        results : dict
            Output from cross_validate() or cross_validate_loo().
        """
        decoded = results['decoded_angles']
        actual = results['actual_angles']
        metrics = results['metrics']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Decoded vs actual (scatter density)
        ax = axes[0]
        ax.scatter(actual.flatten(), decoded.flatten(), alpha=0.5, s=1, color='k')
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

    @staticmethod
    def print_metrics(metrics):
        print("=" * 40)
        print(f"  MAE:          {metrics['mae']:.2f}")
        print(f"  RMSE:         {metrics['rmse']:.2f}")
        print(f"  Correlation:  {metrics['correlation']:.3f}")
        print(f"  R²:           {metrics['r2']:.3f}")
        print("=" * 40)

    @staticmethod
    def compare(results1, results2, labels=('Decoder 1', 'Decoder 2')):
        """Compare two decoder results with plots and statistical tests.

        Parameters
        ----------
        results1, results2 : dict
            Output from cross_validate() or cross_validate_loo().
            Must have same number of trials/samples.
        labels : tuple of str
            Names for the two decoders.

        Returns
        -------
        dict
            metrics1, metrics2: individual metrics dicts
            paired_ttest: (t_statistic, p_value) on per-trial MAE
            wilcoxon: (statistic, p_value) non-parametric test
        """
        from scipy import stats

        decoded1 = results1['decoded_angles']
        decoded2 = results2['decoded_angles']
        actual1 = results1['actual_angles']
        actual2 = results2['actual_angles']
        metrics1 = results1['metrics']
        metrics2 = results2['metrics']

        # Compute per-trial MAE
        mae_per_trial1 = np.mean(np.abs(decoded1 - actual1), axis=1)
        mae_per_trial2 = np.mean(np.abs(decoded2 - actual2), axis=1)

        # Statistical tests
        ttest = stats.ttest_rel(mae_per_trial1, mae_per_trial2)
        try:
            wilcoxon = stats.wilcoxon(mae_per_trial1, mae_per_trial2)
        except ValueError:
            # Wilcoxon fails if all differences are zero
            wilcoxon = (np.nan, np.nan)

        # Get angle range from results (assume same for both)
        n_angles = decoded1.shape[1]
        angle_range = actual1[0] if len(actual1) > 0 else np.arange(n_angles)

        # Plotting: 1 row, 3 columns
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. MAE per angle overlay
        ax = axes[0]
        abs_err1 = np.abs(decoded1 - actual1)
        abs_err2 = np.abs(decoded2 - actual2)
        mae_angle1 = np.mean(abs_err1, axis=0)
        mae_angle2 = np.mean(abs_err2, axis=0)
        sem1 = np.std(abs_err1, axis=0) / np.sqrt(abs_err1.shape[0])
        sem2 = np.std(abs_err2, axis=0) / np.sqrt(abs_err2.shape[0])
        ax.plot(angle_range, mae_angle1, 'b-', linewidth=2, label=labels[0])
        ax.fill_between(angle_range, mae_angle1 - sem1, mae_angle1 + sem1, alpha=0.2, color='blue')
        ax.plot(angle_range, mae_angle2, '-', color='orange', linewidth=2, label=labels[1])
        ax.fill_between(angle_range, mae_angle2 - sem2, mae_angle2 + sem2, alpha=0.2, color='orange')
        ax.set_xlabel('Actual angle (°)')
        ax.set_ylabel('MAE (°)')
        ax.set_title('MAE per angle')
        ax.legend()

        # 2. Error distribution histogram
        ax = axes[1]
        errors1 = (decoded1 - actual1).flatten()
        errors2 = (decoded2 - actual2).flatten()
        bins = np.linspace(min(errors1.min(), errors2.min()),
                          max(errors1.max(), errors2.max()), 50)
        ax.hist(errors1, bins=bins, alpha=0.5, color='blue', label=labels[0], density=True)
        ax.hist(errors2, bins=bins, alpha=0.5, color='orange', label=labels[1], density=True)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Decoding error (°)')
        ax.set_ylabel('Density')
        ax.set_title('Error distribution')
        ax.legend()

        # 3. Metrics panel
        ax = axes[2]
        ax.axis('off')
        textstr = (
            f"{labels[0]}:\n"
            f"  MAE = {metrics1['mae']:.2f}°\n"
            f"  RMSE = {metrics1['rmse']:.2f}°\n"
            f"  R² = {metrics1['r2']:.3f}\n\n"
            f"{labels[1]}:\n"
            f"  MAE = {metrics2['mae']:.2f}°\n"
            f"  RMSE = {metrics2['rmse']:.2f}°\n"
            f"  R² = {metrics2['r2']:.3f}\n\n"
            f"{'─' * 20}\n"
            f"t-test: p = {ttest.pvalue:.4f}\n"
            f"Wilcoxon: p = {wilcoxon[1]:.4f}"
        )
        ax.text(0.1, 0.5, textstr, fontsize=12, verticalalignment='center',
                fontfamily='monospace', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Metrics')

        plt.tight_layout()
        plt.show()

        return {
            'metrics1': metrics1,
            'metrics2': metrics2,
            'paired_ttest': (ttest.statistic, ttest.pvalue),
            'wilcoxon': (wilcoxon[0], wilcoxon[1]),
            'mae_per_trial1': mae_per_trial1,
            'mae_per_trial2': mae_per_trial2,
        }


class MLEOld:

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
    """Maximum likelihood decoder for distance (continuous output via posterior mean).

    A general-purpose decoder that works directly with neural response arrays.
    Compatible with SessionAggregator.get_decoder_data() output.

    Parameters
    ----------
    distance_values : array-like, optional
        Distance labels to decode. Default np.arange(4) for distances 0-3.
    var_floor : float, optional
        Minimum variance to prevent numerical instability. Default 1e-2.
    var_shrinkage : float, optional
        Shrinkage toward pooled (distance-averaged) variance. Default 0.0.

    Attributes
    ----------
    neuron_means : np.ndarray or None
        Fitted mean responses, shape (n_neurons, n_distances, n_angles).
    neuron_variances : np.ndarray or None
        Fitted variances, shape (n_neurons, n_distances, n_angles).
    """

    def __init__(self, distance_values=None, var_floor=1e-2, var_shrinkage=0.0, chunk_size=1):
        """
        Parameters
        ----------
        distance_values : array-like, optional
            Distance labels to decode. Default np.arange(4) for distances 0-3.
        var_floor : float, optional
            Minimum variance to prevent numerical instability. Default 1e-2.
        var_shrinkage : float, optional
            Shrinkage toward pooled (distance-averaged) variance. Default 0.0.
        chunk_size : int, optional
            Number of angles to pool together. Default 1 (no pooling).
            E.g., chunk_size=5 pools every 5 adjacent angles.
        """
        if distance_values is None:
            distance_values = np.arange(4)
        self.distance_values = np.asarray(distance_values)
        self.n_distances = len(self.distance_values)
        self.var_floor = var_floor
        self.var_shrinkage = var_shrinkage
        self.chunk_size = chunk_size

        self.neuron_means = None
        self.neuron_variances = None
        self.n_neurons = None
        self.n_angles = None
        self.n_chunks = None

    def _get_chunk_indices(self):
        """Return list of arrays, each containing angle indices for one chunk."""
        return [np.arange(i * self.chunk_size,
                          min((i + 1) * self.chunk_size, self.n_angles))
                for i in range(self.n_chunks)]

    def fit(self, X_train, y_train):
        """Fit the decoder by computing mean and variance per distance per chunk.

        Parameters
        ----------
        X_train : np.ndarray
            Training data, shape (n_neurons, n_angles, n_trials).
        y_train : np.ndarray
            Distance labels, shape (n_trials,).

        Returns
        -------
        self : MLEDistance
            Fitted decoder instance.
        """
        self.n_neurons = X_train.shape[0]
        self.n_angles = X_train.shape[1]
        self.n_chunks = int(np.ceil(self.n_angles / self.chunk_size))

        # Compute stats per chunk (not per angle)
        self.neuron_means = np.zeros((self.n_neurons, self.n_distances, self.n_chunks))
        self.neuron_variances = np.zeros((self.n_neurons, self.n_distances, self.n_chunks))

        chunk_indices = self._get_chunk_indices()

        for dist_idx, dist_val in enumerate(self.distance_values):
            mask = y_train == dist_val
            if mask.sum() > 0:
                X_dist = X_train[:, :, mask]  # (n_neurons, n_angles, n_trials_dist)
                for c, angle_idxs in enumerate(chunk_indices):
                    # Average responses within chunk, then compute stats
                    X_chunk = X_dist[:, angle_idxs, :].mean(axis=1)  # (n_neurons, n_trials)
                    self.neuron_means[:, dist_idx, c] = np.mean(X_chunk, axis=1)
                    self.neuron_variances[:, dist_idx, c] = np.var(X_chunk, axis=1)

        if self.var_shrinkage > 0:
            pooled = np.mean(self.neuron_variances, axis=1, keepdims=True)
            self.neuron_variances = ((1 - self.var_shrinkage) * self.neuron_variances
                                     + self.var_shrinkage * pooled)
        return self

    def _log_likelihood(self, response_vector, chunk_idx, dist_idx):
        """Gaussian log-likelihood for a given (chunk, distance)."""
        expected = self.neuron_means[:, dist_idx, chunk_idx]
        variances = np.maximum(self.neuron_variances[:, dist_idx, chunk_idx], self.var_floor)
        return -0.5 * np.sum(np.log(variances) + (response_vector - expected) ** 2 / variances)

    def decode(self, response_vector, chunk_idx):
        """Decode distance at a single chunk position using posterior mean.

        Parameters
        ----------
        response_vector : np.ndarray, shape (n_neurons,)
            Response averaged within the chunk.
        chunk_idx : int

        Returns
        -------
        float : predicted distance (continuous)
        """
        log_likelihoods = np.array([
            self._log_likelihood(response_vector, chunk_idx, d)
            for d in range(self.n_distances)
        ])
        log_likelihoods = log_likelihoods / self.n_neurons
        log_likelihoods = log_likelihoods - np.max(log_likelihoods)
        probabilities = np.exp(log_likelihoods)
        probabilities = probabilities / np.sum(probabilities)
        return np.sum(self.distance_values * probabilities)

    def decode_trial(self, trial_data):
        """Decode distance at each chunk position for a single trial.

        Parameters
        ----------
        trial_data : np.ndarray, shape (n_neurons, n_angles)
            Neural responses for one trial.

        Returns
        -------
        np.ndarray, shape (n_chunks,)
            Decoded distance at each chunk.
        """
        chunk_indices = self._get_chunk_indices()
        decoded = np.zeros(self.n_chunks)
        for c, angle_idxs in enumerate(chunk_indices):
            # Average response within chunk
            avg_response = trial_data[:, angle_idxs].mean(axis=1)
            decoded[c] = self.decode(avg_response, c)
        return decoded

    def decode_trials(self, X):
        """Decode distance trajectories for multiple trials.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)

        Returns
        -------
        np.ndarray, shape (n_trials, n_angles)
        """
        n_trials = X.shape[2]
        return np.array([self.decode_trial(X[:, :, t]) for t in range(n_trials)])

    def cross_validate(self, X, y_distance, n_folds=5, average_test=False, random_state=None):
        """Run stratified k-fold cross-validation for distance decoding.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Full dataset.
        y_distance : np.ndarray, shape (n_trials,)
            Distance labels for each trial.
        n_folds : int
            Number of CV folds.
        average_test : bool
            If True, average test trials BY DISTANCE CLASS before decoding.
            Returns one decoded trajectory per distance per fold.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            decoded_distances : np.ndarray
            actual_distances : np.ndarray
            metrics : dict
        """
        n_trials = X.shape[2]

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kfold.split(np.zeros(n_trials), y_distance))

        if average_test:
            # Average test trials by distance class, then decode
            all_decoded = []
            all_actual = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f'Fold {fold_idx + 1}/{n_folds}', end='\r')

                X_train = X[:, :, train_idx]
                y_train = y_distance[train_idx]
                X_test = X[:, :, test_idx]
                y_test = y_distance[test_idx]

                self.fit(X_train, y_train)

                # For each distance in test set, average and decode
                for dist_val in self.distance_values:
                    dist_mask = y_test == dist_val
                    if dist_mask.sum() > 0:
                        X_test_dist = X_test[:, :, dist_mask]
                        X_test_mean = np.mean(X_test_dist, axis=2)  # (n_neurons, n_angles)
                        decoded = self.decode_trial(X_test_mean)  # (n_angles,)
                        all_decoded.append(decoded)
                        all_actual.append(dist_val)

            # Stack into 2D: (n_samples, n_angles)
            decoded_distances = np.stack(all_decoded, axis=0)
            actual_distances = np.array(all_actual)
        else:
            # Decode each trial individually
            all_decoded = []
            all_actual = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f'Fold {fold_idx + 1}/{n_folds}', end='\r')

                X_train = X[:, :, train_idx]
                y_train = y_distance[train_idx]
                X_test = X[:, :, test_idx]
                y_test = y_distance[test_idx]

                self.fit(X_train, y_train)
                decoded = self.decode_trials(X_test)  # (n_test_trials, n_angles)

                all_decoded.append(decoded)
                all_actual.extend(y_test)

            # Stack into 2D: (n_trials, n_angles)
            decoded_distances = np.vstack(all_decoded)
            actual_distances = np.array(all_actual)

        metrics = self.evaluate(actual_distances, decoded_distances)

        print(f'\nTotal predictions: {len(decoded_distances)}')
        self.print_metrics(metrics)

        return {
            'decoded_distances': decoded_distances,
            'actual_distances': actual_distances,
            'metrics': metrics,
        }

    @staticmethod
    def evaluate(actual, decoded):
        """Compute distance decoding metrics.

        Parameters
        ----------
        actual : array, shape (n_trials,)
            True distance values per trial.
        decoded : array, shape (n_trials, n_angles)
            Decoded distance values at each angle for each trial.

        Returns
        -------
        dict with keys: mae, rmse, bias, std, correlation
        """
        actual = np.array(actual)
        decoded = np.array(decoded)

        # Broadcast actual to match decoded shape: (n_trials,) -> (n_trials, 1)
        if decoded.ndim == 2 and actual.ndim == 1:
            actual_broadcast = actual[:, np.newaxis]
        else:
            actual_broadcast = actual

        errors = decoded - actual_broadcast

        # For correlation, use mean decoded per trial vs actual
        if decoded.ndim == 2:
            mean_decoded = np.mean(decoded, axis=1)
            corr = np.corrcoef(actual.flatten(), mean_decoded.flatten())[0, 1]
        else:
            corr = np.corrcoef(actual.flatten(), decoded.flatten())[0, 1]

        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'bias': np.mean(errors),
            'std': np.std(errors),
            'correlation': corr,
        }

    def plot_results(self, results, angle_range=None):
        """Plot distance decoding results.

        Parameters
        ----------
        results : dict
            Output from cross_validate().
        angle_range : array, optional
            Angle values for x-axis. If None, uses angle indices.

        Returns
        -------
        fig : matplotlib figure
        """
        decoded_distances = results['decoded_distances']  # (n_trials, n_angles)
        actual_distances = results['actual_distances']  # (n_trials,)
        metrics = results['metrics']

        n_trials, n_angles = decoded_distances.shape
        if angle_range is None:
            angle_range = np.arange(n_angles)

        # Compute errors per angle
        # actual_distances is (n_trials,), need to broadcast
        errors = decoded_distances - actual_distances[:, np.newaxis]  # (n_trials, n_angles)
        mae_per_angle = np.mean(np.abs(errors), axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Plot 1: MAE per angle
        ax = axes[0]
        ax.plot(angle_range, mae_per_angle, 'b-', linewidth=2)
        ax.axhline(metrics['mae'], color='g', linestyle='--', alpha=0.7,
                   label=f'Overall MAE = {metrics["mae"]:.2f}')
        ax.axhline(1.25, color='r', linestyle=':', alpha=0.5, label='Random ~1.25')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('MAE')
        ax.set_title('MAE per Angle')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error distribution
        ax = axes[1]
        all_errors = errors.flatten()
        ax.hist(all_errors, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Error (decoded - actual)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)

        # Plot 3: Metrics panel
        ax = axes[2]
        ax.axis('off')
        metrics_text = (
            f"MAE: {metrics['mae']:.3f}\n"
            f"RMSE: {metrics['rmse']:.3f}\n"
            f"Bias: {metrics['bias']:.3f}\n"
            f"Std: {metrics['std']:.3f}\n"
            f"Correlation: {metrics['correlation']:.3f}\n"
            f"\nReference:\n"
            f"Random MAE ~ 1.25"
        )
        ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Metrics')

        plt.tight_layout()
        return fig

    @staticmethod
    def compare(results1, results2, labels=('Decoder 1', 'Decoder 2'), angle_range=None):
        """Compare two distance decoder results with plots and statistical tests.

        Parameters
        ----------
        results1, results2 : dict
            Output from cross_validate().
        labels : tuple of str
            Names for the two decoders.
        angle_range : array, optional
            Angle values for x-axis. If None, uses angle indices.

        Returns
        -------
        dict with comparison metrics and test results
        """
        from scipy import stats

        decoded1 = results1['decoded_distances']  # (n_trials, n_angles)
        decoded2 = results2['decoded_distances']
        actual1 = results1['actual_distances']
        actual2 = results2['actual_distances']
        metrics1 = results1['metrics']
        metrics2 = results2['metrics']

        n_trials, n_angles = decoded1.shape
        if angle_range is None:
            angle_range = np.arange(n_angles)

        # Compute errors
        errors1 = decoded1 - actual1[:, np.newaxis]
        errors2 = decoded2 - actual2[:, np.newaxis]

        # MAE per angle for both
        mae1_per_angle = np.mean(np.abs(errors1), axis=0)
        mae2_per_angle = np.mean(np.abs(errors2), axis=0)

        # MAE per trial (averaged across angles)
        mae1_per_trial = np.mean(np.abs(errors1), axis=1)
        mae2_per_trial = np.mean(np.abs(errors2), axis=1)

        # Statistical tests on per-trial MAE
        t_stat, t_pval = stats.ttest_rel(mae1_per_trial, mae2_per_trial)
        try:
            w_stat, w_pval = stats.wilcoxon(mae1_per_trial, mae2_per_trial)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Plot 1: MAE per angle comparison
        ax = axes[0]
        ax.plot(angle_range, mae1_per_angle, 'b-', linewidth=2, label=labels[0])
        ax.plot(angle_range, mae2_per_angle, 'r-', linewidth=2, label=labels[1])
        ax.axhline(1.25, color='k', linestyle=':', alpha=0.5, label='Random ~1.25')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('MAE')
        ax.set_title('MAE per Angle')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error distribution comparison
        ax = axes[1]
        ax.hist(errors1.flatten(), bins=30, alpha=0.5, label=labels[0], edgecolor='blue')
        ax.hist(errors2.flatten(), bins=30, alpha=0.5, label=labels[1], edgecolor='red')
        ax.axvline(0, color='k', linestyle='--', alpha=0.7)
        ax.set_xlabel('Error (decoded - actual)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Metrics comparison
        ax = axes[2]
        ax.axis('off')
        metrics_text = (
            f"{labels[0]:>12}  {labels[1]:>12}\n"
            f"{'='*28}\n"
            f"MAE:  {metrics1['mae']:>8.3f}  {metrics2['mae']:>12.3f}\n"
            f"RMSE: {metrics1['rmse']:>8.3f}  {metrics2['rmse']:>12.3f}\n"
            f"Bias: {metrics1['bias']:>8.3f}  {metrics2['bias']:>12.3f}\n"
            f"Corr: {metrics1['correlation']:>8.3f}  {metrics2['correlation']:>12.3f}\n"
            f"{'='*28}\n"
            f"Paired t-test p: {t_pval:.4f}\n"
            f"Wilcoxon p:      {w_pval:.4f}"
        )
        ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Comparison')

        plt.tight_layout()

        return {
            'metrics1': metrics1,
            'metrics2': metrics2,
            'paired_ttest': (t_stat, t_pval),
            'wilcoxon': (w_stat, w_pval),
            'fig': fig,
        }

    @staticmethod
    def print_metrics(metrics):
        """Print distance decoder metrics."""
        print("=" * 50)
        print("DISTANCE DECODER PERFORMANCE")
        print("=" * 50)
        print(f"  Mean Absolute Error:  {metrics['mae']:.3f}")
        print(f"  RMSE:                 {metrics['rmse']:.3f}")
        print(f"  Bias:                 {metrics['bias']:.3f}")
        print(f"  Std:                  {metrics['std']:.3f}")
        print(f"  Correlation:          {metrics['correlation']:.3f}")
        print()
        print("  Reference (random guess MAE ~ 1.25)")
        print("=" * 50)


class RidgeDistanceCV:
    """Ridge regression decoder for distance (array-based interface).

    Compatible with SessionAggregator.get_decoder_data() output.
    Uses RidgeCV for automatic regularization strength selection.

    Parameters
    ----------
    distance_values : array-like, optional
        Distance labels. Default np.arange(4) for distances 0-3.
    chunk_size : int, optional
        Number of angles to pool together. Default 1 (no pooling).
    alphas : array-like, optional
        Alpha values for RidgeCV. Default logspace(-3, 3, 50).
    cv : int, optional
        Internal CV folds for alpha selection. Default 5.

    Attributes
    ----------
    models_ : list
        Fitted RidgeCV models, one per chunk.
    n_neurons : int
        Number of neurons (set after fit).
    n_angles : int
        Number of angle positions (set after fit).
    n_chunks : int
        Number of chunks (set after fit).
    """

    def __init__(self, distance_values=None, chunk_size=1, alphas=None, cv=5):
        if distance_values is None:
            distance_values = np.arange(4)
        self.distance_values = np.asarray(distance_values)
        self.n_distances = len(self.distance_values)
        self.chunk_size = chunk_size
        self.alphas = alphas if alphas is not None else np.logspace(-3, 3, 50)
        self.cv = cv

        self.models_ = None
        self.n_neurons = None
        self.n_angles = None
        self.n_chunks = None

    def _get_chunk_indices(self):
        """Return list of arrays, each containing angle indices for one chunk."""
        return [np.arange(i * self.chunk_size,
                          min((i + 1) * self.chunk_size, self.n_angles))
                for i in range(self.n_chunks)]

    def fit(self, X_train, y_train):
        """Fit one Ridge model per chunk position.

        Parameters
        ----------
        X_train : np.ndarray
            Training data, shape (n_neurons, n_angles, n_trials).
        y_train : np.ndarray
            Distance labels, shape (n_trials,).

        Returns
        -------
        self : RidgeDistanceCV
            Fitted decoder instance.
        """
        self.n_neurons = X_train.shape[0]
        self.n_angles = X_train.shape[1]
        n_trials = X_train.shape[2]
        self.n_chunks = int(np.ceil(self.n_angles / self.chunk_size))

        chunk_indices = self._get_chunk_indices()
        self.models_ = []

        for c, angle_idxs in enumerate(chunk_indices):
            # Average responses within chunk: (n_neurons, n_trials)
            X_chunk = X_train[:, angle_idxs, :].mean(axis=1)
            # Transpose to (n_trials, n_neurons) for sklearn
            X_chunk = X_chunk.T

            model = RidgeCV(alphas=self.alphas, cv=self.cv)
            model.fit(X_chunk, y_train)
            self.models_.append(model)

        return self

    def decode(self, response_vector, chunk_idx):
        """Decode distance at a single chunk position.

        Parameters
        ----------
        response_vector : np.ndarray, shape (n_neurons,)
            Response averaged within the chunk.
        chunk_idx : int
            Chunk index.

        Returns
        -------
        float : predicted distance
        """
        model = self.models_[chunk_idx]
        return model.predict(response_vector.reshape(1, -1))[0]

    def decode_trial(self, trial_data):
        """Decode distance at each chunk position for a single trial.

        Parameters
        ----------
        trial_data : np.ndarray, shape (n_neurons, n_angles)
            Neural responses for one trial.

        Returns
        -------
        np.ndarray, shape (n_chunks,)
            Decoded distance at each chunk.
        """
        chunk_indices = self._get_chunk_indices()
        decoded = np.zeros(self.n_chunks)

        for c, angle_idxs in enumerate(chunk_indices):
            # Average response within chunk
            avg_response = trial_data[:, angle_idxs].mean(axis=1)
            decoded[c] = self.decode(avg_response, c)

        return decoded

    def decode_trials(self, X):
        """Decode distance trajectories for multiple trials.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)

        Returns
        -------
        np.ndarray, shape (n_trials, n_chunks)
        """
        n_trials = X.shape[2]
        return np.array([self.decode_trial(X[:, :, t]) for t in range(n_trials)])

    def cross_validate(self, X, y_distance, n_folds=5, average_test=False, random_state=None):
        """Run stratified k-fold cross-validation for distance decoding.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Full dataset.
        y_distance : np.ndarray, shape (n_trials,)
            Distance labels for each trial.
        n_folds : int
            Number of CV folds.
        average_test : bool
            If True, average test trials BY DISTANCE CLASS before decoding.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict
            decoded_distances : np.ndarray
            actual_distances : np.ndarray
            metrics : dict
        """
        n_trials = X.shape[2]

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(kfold.split(np.zeros(n_trials), y_distance))

        if average_test:
            all_decoded = []
            all_actual = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f'Fold {fold_idx + 1}/{n_folds}', end='\r')

                X_train = X[:, :, train_idx]
                y_train = y_distance[train_idx]
                X_test = X[:, :, test_idx]
                y_test = y_distance[test_idx]

                self.fit(X_train, y_train)

                # For each distance in test set, average and decode
                for dist_val in self.distance_values:
                    dist_mask = y_test == dist_val
                    if dist_mask.sum() > 0:
                        X_test_dist = X_test[:, :, dist_mask]
                        X_test_mean = np.mean(X_test_dist, axis=2)
                        decoded = self.decode_trial(X_test_mean)
                        all_decoded.append(decoded)
                        all_actual.append(dist_val)

            decoded_distances = np.stack(all_decoded, axis=0)
            actual_distances = np.array(all_actual)
        else:
            all_decoded = []
            all_actual = []

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f'Fold {fold_idx + 1}/{n_folds}', end='\r')

                X_train = X[:, :, train_idx]
                y_train = y_distance[train_idx]
                X_test = X[:, :, test_idx]
                y_test = y_distance[test_idx]

                self.fit(X_train, y_train)
                decoded = self.decode_trials(X_test)

                all_decoded.append(decoded)
                all_actual.extend(y_test)

            decoded_distances = np.vstack(all_decoded)
            actual_distances = np.array(all_actual)

        metrics = self.evaluate(actual_distances, decoded_distances)

        print(f'\nTotal predictions: {len(decoded_distances)}')
        self.print_metrics(metrics)

        return {
            'decoded_distances': decoded_distances,
            'actual_distances': actual_distances,
            'metrics': metrics,
        }

    @staticmethod
    def evaluate(actual, decoded):
        """Compute distance decoding metrics.

        Parameters
        ----------
        actual : array, shape (n_trials,)
            True distance values per trial.
        decoded : array, shape (n_trials, n_chunks)
            Decoded distance values at each chunk for each trial.

        Returns
        -------
        dict with keys: mae, rmse, bias, std, correlation
        """
        actual = np.array(actual)
        decoded = np.array(decoded)

        # Broadcast actual to match decoded shape
        if decoded.ndim == 2 and actual.ndim == 1:
            actual_broadcast = actual[:, np.newaxis]
        else:
            actual_broadcast = actual

        errors = decoded - actual_broadcast

        # For correlation, use mean decoded per trial vs actual
        if decoded.ndim == 2:
            mean_decoded = np.mean(decoded, axis=1)
            corr = np.corrcoef(actual.flatten(), mean_decoded.flatten())[0, 1]
        else:
            corr = np.corrcoef(actual.flatten(), decoded.flatten())[0, 1]

        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'bias': np.mean(errors),
            'std': np.std(errors),
            'correlation': corr,
        }

    def plot_results(self, results, angle_range=None):
        """Plot distance decoding results.

        Parameters
        ----------
        results : dict
            Output from cross_validate().
        angle_range : array, optional
            Angle values for x-axis. If None, uses angle indices.

        Returns
        -------
        fig : matplotlib figure
        """
        decoded_distances = results['decoded_distances']
        actual_distances = results['actual_distances']
        metrics = results['metrics']

        n_trials, n_chunks = decoded_distances.shape
        if angle_range is None:
            angle_range = np.arange(n_chunks)

        errors = decoded_distances - actual_distances[:, np.newaxis]
        mae_per_chunk = np.mean(np.abs(errors), axis=0)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Plot 1: MAE per chunk
        ax = axes[0]
        ax.plot(angle_range, mae_per_chunk, 'b-', linewidth=2)
        ax.axhline(metrics['mae'], color='g', linestyle='--', alpha=0.7,
                   label=f'Overall MAE = {metrics["mae"]:.2f}')
        ax.axhline(1.25, color='r', linestyle=':', alpha=0.5, label='Random ~1.25')
        ax.set_xlabel('Angle (deg)' if len(angle_range) > n_chunks else 'Chunk')
        ax.set_ylabel('MAE')
        ax.set_title('MAE per Position (Ridge)')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error distribution
        ax = axes[1]
        all_errors = errors.flatten()
        ax.hist(all_errors, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Error (decoded - actual)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.grid(True, alpha=0.3)

        # Plot 3: Metrics panel
        ax = axes[2]
        ax.axis('off')
        metrics_text = (
            f"MAE: {metrics['mae']:.3f}\n"
            f"RMSE: {metrics['rmse']:.3f}\n"
            f"Bias: {metrics['bias']:.3f}\n"
            f"Std: {metrics['std']:.3f}\n"
            f"Correlation: {metrics['correlation']:.3f}\n"
            f"\nReference:\n"
            f"Random MAE ~ 1.25"
        )
        ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Metrics (Ridge)')

        plt.tight_layout()
        return fig

    @staticmethod
    def compare(results1, results2, labels=('Decoder 1', 'Decoder 2'), angle_range=None):
        """Compare two distance decoder results with plots and statistical tests.

        Parameters
        ----------
        results1, results2 : dict
            Output from cross_validate().
        labels : tuple of str
            Names for the two decoders.
        angle_range : array, optional
            Angle values for x-axis. If None, uses indices.

        Returns
        -------
        dict with comparison metrics and test results
        """
        from scipy import stats

        decoded1 = results1['decoded_distances']
        decoded2 = results2['decoded_distances']
        actual1 = results1['actual_distances']
        actual2 = results2['actual_distances']
        metrics1 = results1['metrics']
        metrics2 = results2['metrics']

        n_trials, n_chunks = decoded1.shape
        if angle_range is None:
            angle_range = np.arange(n_chunks)

        # Compute errors
        errors1 = decoded1 - actual1[:, np.newaxis]
        errors2 = decoded2 - actual2[:, np.newaxis]

        # MAE per chunk for both
        mae1_per_chunk = np.mean(np.abs(errors1), axis=0)
        mae2_per_chunk = np.mean(np.abs(errors2), axis=0)

        # MAE per trial (averaged across chunks)
        mae1_per_trial = np.mean(np.abs(errors1), axis=1)
        mae2_per_trial = np.mean(np.abs(errors2), axis=1)

        # Statistical tests on per-trial MAE
        t_stat, t_pval = stats.ttest_rel(mae1_per_trial, mae2_per_trial)
        try:
            w_stat, w_pval = stats.wilcoxon(mae1_per_trial, mae2_per_trial)
        except ValueError:
            w_stat, w_pval = np.nan, np.nan

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Plot 1: MAE per chunk comparison
        ax = axes[0]
        ax.plot(angle_range, mae1_per_chunk, 'b-', linewidth=2, label=labels[0])
        ax.plot(angle_range, mae2_per_chunk, 'r-', linewidth=2, label=labels[1])
        ax.axhline(1.25, color='k', linestyle=':', alpha=0.5, label='Random ~1.25')
        ax.set_xlabel('Position')
        ax.set_ylabel('MAE')
        ax.set_title('MAE per Position')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Error distribution comparison
        ax = axes[1]
        ax.hist(errors1.flatten(), bins=30, alpha=0.5, label=labels[0], edgecolor='blue')
        ax.hist(errors2.flatten(), bins=30, alpha=0.5, label=labels[1], edgecolor='red')
        ax.axvline(0, color='k', linestyle='--', alpha=0.7)
        ax.set_xlabel('Error (decoded - actual)')
        ax.set_ylabel('Count')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Metrics comparison
        ax = axes[2]
        ax.axis('off')
        metrics_text = (
            f"{labels[0]:>12}  {labels[1]:>12}\n"
            f"{'='*28}\n"
            f"MAE:  {metrics1['mae']:>8.3f}  {metrics2['mae']:>12.3f}\n"
            f"RMSE: {metrics1['rmse']:>8.3f}  {metrics2['rmse']:>12.3f}\n"
            f"Bias: {metrics1['bias']:>8.3f}  {metrics2['bias']:>12.3f}\n"
            f"Corr: {metrics1['correlation']:>8.3f}  {metrics2['correlation']:>12.3f}\n"
            f"{'='*28}\n"
            f"Paired t-test p: {t_pval:.4f}\n"
            f"Wilcoxon p:      {w_pval:.4f}"
        )
        ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax.set_title('Comparison')

        plt.tight_layout()

        return {
            'metrics1': metrics1,
            'metrics2': metrics2,
            'paired_ttest': (t_stat, t_pval),
            'wilcoxon': (w_stat, w_pval),
            'fig': fig,
        }

    @staticmethod
    def print_metrics(metrics):
        """Print distance decoder metrics."""
        print("=" * 50)
        print("DISTANCE DECODER PERFORMANCE (Ridge)")
        print("=" * 50)
        print(f"  Mean Absolute Error:  {metrics['mae']:.3f}")
        print(f"  RMSE:                 {metrics['rmse']:.3f}")
        print(f"  Bias:                 {metrics['bias']:.3f}")
        print(f"  Std:                  {metrics['std']:.3f}")
        print(f"  Correlation:          {metrics['correlation']:.3f}")
        print()
        print("  Reference (random guess MAE ~ 1.25)")
        print("=" * 50)


def plot_mae_by_angle_range(agg, area=None, n_folds=10):
    """Continuous MAE-per-angle comparison and boxplot by region,
    monocular vs binocular.

    Parameters
    ----------
    agg : SessionAggregator
        Aggregator object for the animal/session.
    area : str, optional
        Brain area to decode from (e.g. 'V1'). None uses all areas.
    n_folds : int
        Number of cross-validation folds. Default 10.
    """
    from matplotlib.patches import Patch
    import pandas as pd

    ranges = {
        'Left Eye Only': (-30, -20),
        'Both Eyes': (-20, 20),
        'Right Eye Only': (20, 60),
    }

    conditions = {'Monocular': 1, 'Binocular': 0}
    plot_colors = {'Monocular': 'coral', 'Binocular': 'blue'}
    box_colors = {'Monocular': 'coral', 'Binocular': 'lightblue'}

    abs_errors_by_cond = {}
    angle_range = None
    data = []

    for cond_label, mono_val in conditions.items():
        X, y = agg.get_decoder_data(monocular=mono_val, area=area)
        angle_range = np.linspace(-30, 60, X.shape[1])
        decoder = MLE(angle_range)
        results = decoder.cross_validate(X, n_folds=n_folds)

        decoded = results['decoded_angles']
        actual = results['actual_angles']
        abs_errors = np.abs(decoded - actual)  # (n_trials, n_angles)
        abs_errors_by_cond[cond_label] = abs_errors

        for range_label, (lo, hi) in ranges.items():
            mask = (angle_range >= lo) & (angle_range <= hi)
            trial_maes = np.mean(abs_errors[:, mask], axis=1)
            for mae_val in trial_maes:
                data.append({
                    'Angle Range': range_label,
                    'Condition': cond_label,
                    'MAE': mae_val,
                })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: continuous MAE per angle
    ax = axes[0]
    for cond_label, abs_errors in abs_errors_by_cond.items():
        mae_per_angle = np.mean(abs_errors, axis=0)
        sem_per_angle = np.std(abs_errors, axis=0) / np.sqrt(abs_errors.shape[0])
        c = plot_colors[cond_label]
        mean_mae = np.mean(mae_per_angle)
        ax.plot(angle_range, mae_per_angle, '-', color=c, linewidth=2, label=f'{cond_label} (mean={mean_mae:.2f}°)')
        ax.fill_between(angle_range, mae_per_angle - sem_per_angle,
                        mae_per_angle + sem_per_angle, alpha=0.2, color=c)
    ax.axvspan(-20, 20, color='purple', alpha=0.3, label='Binocular overlap')
    ax.axvline(-20, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(20, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('MAE (deg)')
    ax.set_title(f'MAE per Angle — {area or "All Areas"}')
    ax.legend()

    # Right panel: boxplot by region
    ax = axes[1]
    range_labels = list(ranges.keys())
    x_positions = np.arange(len(range_labels))
    width = 0.35

    for i, cond in enumerate(['Monocular', 'Binocular']):
        bp_data = [df[(df['Angle Range'] == r) & (df['Condition'] == cond)]['MAE'].values
                   for r in range_labels]
        positions = x_positions + (i - 0.5) * width
        bp = ax.boxplot(bp_data, positions=positions, widths=width * 0.8,
                        patch_artist=True, manage_ticks=False)
        for patch in bp['boxes']:
            patch.set_facecolor(box_colors[cond])
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set_color('black')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(range_labels)
    ax.set_ylabel('MAE (deg)')
    ax.set_title(f'MAE by Region — {area or "All Areas"}')
    legend_elements = [Patch(facecolor=box_colors[c], alpha=0.7, label=c)
                       for c in ['Monocular', 'Binocular']]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()


def plot_mae_by_angle_range_all_areas(agg, n_folds=10):
    """Compare MAE by angle range across all areas, monocular vs binocular.

    Creates a 1x3 figure (one panel per angle range). In each panel the
    x-axis shows brain areas with monocular/binocular boxplots side by side.

    Parameters
    ----------
    agg : SessionAggregator
        Aggregator object for the animal/session.
    n_folds : int
        Number of cross-validation folds. Default 10.
    """
    from matplotlib.patches import Patch
    import pandas as pd

    ranges = {
        'Left Eye Only': (-30, -20),
        'Both Eyes': (-20, 20),
        'Right Eye Only': (20, 40),
    }

    conditions = {'Monocular': 1, 'Binocular': 0}
    colors = {'Monocular': 'coral', 'Binocular': 'blue'}
    min_cells = 50

    # Filter to areas with >= 50 reliable cells
    all_areas = agg.available_areas
    areas = [a for a in all_areas
             if a is not None and str(a) != 'None'
             and len(agg._get_cell_indices(a)) >= min_cells]
    print(f"Areas with >= {min_cells} cells: {areas}")

    data = []
    for area in areas:
        for cond_label, mono_val in conditions.items():
            X, y = agg.get_decoder_data(monocular=mono_val, area=area)
            angle_range = np.linspace(-30, 60, X.shape[1])
            decoder = MLE(angle_range)
            results = decoder.cross_validate(X, n_folds=n_folds)

            decoded = results['decoded_angles']
            actual = results['actual_angles']
            abs_errors = np.abs(decoded - actual)

            for range_label, (lo, hi) in ranges.items():
                mask = (angle_range >= lo) & (angle_range <= hi)
                trial_maes = np.mean(abs_errors[:, mask], axis=1)
                for mae_val in trial_maes:
                    data.append({
                        'Area': area,
                        'Angle Range': range_label,
                        'Condition': cond_label,
                        'MAE': mae_val,
                    })

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    range_labels = list(ranges.keys())
    width = 0.35

    for ax, range_label in zip(axes, range_labels):
        x_positions = np.arange(len(areas))
        for i, cond in enumerate(['Monocular', 'Binocular']):
            bp_data = [
                df[(df['Area'] == a) & (df['Angle Range'] == range_label)
                   & (df['Condition'] == cond)]['MAE'].values
                for a in areas
            ]
            positions = x_positions + (i - 0.5) * width
            bp = ax.boxplot(bp_data, positions=positions, widths=width * 0.8,
                            patch_artist=True, manage_ticks=False)
            for patch in bp['boxes']:
                patch.set_facecolor(colors[cond])
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set_color('black')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(areas)
        ax.set_ylabel('MAE (deg)')
        ax.set_title(range_label)

    # Sync y-axis across all panels
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)

    # Statistical tests: Wilcoxon within each area x range (mono vs bino)
    from scipy.stats import ranksums
    sig_y = y_max * 0.95
    for ax, range_label in zip(axes, range_labels):
        x_positions = np.arange(len(areas))
        for j, area in enumerate(areas):
            mono_vals = df[(df['Area'] == area) & (df['Angle Range'] == range_label)
                           & (df['Condition'] == 'Monocular')]['MAE'].values
            bino_vals = df[(df['Area'] == area) & (df['Angle Range'] == range_label)
                           & (df['Condition'] == 'Binocular')]['MAE'].values
            if len(mono_vals) > 0 and len(bino_vals) > 0:
                _, p = ranksums(mono_vals, bino_vals)
                if p < 0.001:
                    label = '***'
                elif p < 0.01:
                    label = '**'
                elif p < 0.05:
                    label = '*'
                else:
                    continue
                x1 = x_positions[j] - 0.5 * width
                x2 = x_positions[j] + 0.5 * width
                ax.plot([x1, x1, x2, x2], [sig_y * 0.92, sig_y * 0.94, sig_y * 0.94, sig_y * 0.92],
                        'k-', linewidth=1)
                ax.text((x1 + x2) / 2, sig_y * 0.94, label,
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    legend_elements = [Patch(facecolor=colors[c], alpha=0.7, label=c)
                       for c in ['Monocular', 'Binocular']]
    axes[-1].legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()


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

