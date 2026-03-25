import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp, softmax
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut


def _resolve_session(agg, session):
    """Resolve an integer session number to the full session name string.

    Parameters
    ----------
    agg     : SessionAggregator
    session : int or None
        Integer N matching the session whose name contains '_N_'
        (e.g. 9 → '251217_SAM043_tiling_9_120um').

    Returns
    -------
    str or None
    """
    if session is None:
        return session
    if isinstance(session, str):
        raise TypeError(f"session must be an integer, not a string. Got: '{session}'")
    token = f'_{int(session)}_'
    matches = [s for s in agg.trial_info if token in s]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise ValueError(f"No session containing '{token}' found. Available: {list(agg.trial_info.keys())}")
    raise ValueError(f"Ambiguous session number {session}: matched {matches}")


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

    def __init__(self, angle_range=None, var_floor=1e-6):
        self.angle_range = np.ravel(angle_range) if angle_range is not None else None
        self.n_angles = len(self.angle_range) if self.angle_range is not None else None
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
        if self.n_angles is None:
            self.n_angles = X_train.shape[1]
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
                       average_test=False, angle_ranges=None):
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
        angle_ranges : list of (lo, hi) tuples, optional
            Degree ranges for per-range breakdown, e.g. [(-30, -15), (-15, 15), (15, 45)].
            If provided, ``range_results`` in the output will have one entry per range
            with its angle MAE. If None, ``range_results`` is an empty dict.

        Returns
        -------
        dict
            a_pred      : np.ndarray, shape (1, n_rows, n_angles) — decoded degrees
            a_true      : np.ndarray, shape (n_rows, n_angles)    — true degrees
            angle_axis  : np.ndarray, shape (n_angles,)           — degree at each position
            angle_mae   : float — overall MAE in degrees
            range_results : dict — per-range angle MAE keyed by label like "[-30°, -15°)"
            metrics     : dict with mae, rmse, bias, correlation, r2, std
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

        return self._build_result(decoded_angles, actual_angles, metrics, angle_ranges)

    def _build_result(self, decoded_angles, actual_angles, metrics, angle_ranges):
        """Convert 2-D degree arrays to the shared result dict.

        Output shape convention matches plot_angle_decoding_comparison:
          a_pred  : (1, n_rows, n_angles)  decoded degrees  [1 repeat, n_rows as "conditions"]
          a_true  : (n_rows, n_angles)     true degrees
          angle_axis : (n_angles,)         degree at each position
        """
        angle_axis = self.angle_range  # (n_angles,)

        # decoded_angles / actual_angles already in degrees, shape (n_rows, n_angles)
        a_pred = decoded_angles[np.newaxis]  # (1, n_rows, n_angles)
        a_true = actual_angles               # (n_rows, n_angles)

        if angle_ranges is not None:
            range_results = {}
            for lo, hi in angle_ranges:
                pos_mask = (angle_axis >= lo) & (angle_axis < hi)
                label = f"[{lo}°, {hi}°)"
                if pos_mask.any():
                    err = np.abs(decoded_angles[:, pos_mask] - actual_angles[:, pos_mask])
                    range_results[label] = {'angle_mae': float(err.mean())}
                else:
                    range_results[label] = {'angle_mae': np.nan}
        else:
            range_results = {}

        return {
            'a_pred': a_pred,
            'a_true': a_true,
            'angle_axis': angle_axis,
            'angle_mae': metrics['mae'],
            'range_results': range_results,
            'metrics': metrics,
        }

    def cross_validate_loo(self, X, angle_ranges=None):
        """Run leave-one-out cross-validation.

        Trains on all but 1 trial, tests on that 1 trial, repeats for every trial.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Full dataset.
        angle_ranges : list of (lo, hi) tuples, optional
            Degree ranges for per-range breakdown. See cross_validate().

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

        return self._build_result(decoded_angles, actual_angles, metrics, angle_ranges)

    @staticmethod
    def cross_validate_averaged(agg, monocular, area=None, n_repeats=10, cell_idx=None):
        """Cross-validated MLE on condition-averaged data.

        For each repeat, splits trials into train/test (stratified by distance),
        fits MLE on train trials, then averages test trials per (distance, angle)
        condition and decodes the mean response.

        Same calling interface as FactoredMLE.cross_validate_averaged, enabling
        direct comparison of angle decoding performance between the two decoders.

        Parameters
        ----------
        agg       : SessionAggregator  (must have .angle_axis and .distances)
        monocular : int   0 = binocular, 1 = monocular
        area      : str    visual area (None = all areas)
        n_repeats : int
        cell_idx  : array-like or None   neuron subset (None = all)

        Returns
        -------
        dict with keys:
            a_pred         shape (n_repeats, n_distances, n_positions)  decoded angle (degrees)
            a_true         shape (n_distances, n_positions)             true angle grid
            angle_axis     shape (n_positions,)   degrees for each position index
            distances      shape (n_distances,)   distance values
            a_error_by_pos shape (n_positions,)   mean abs angle error per position (deg)
        """
        from sklearn.model_selection import StratifiedShuffleSplit

        X, y_distance = agg.get_decoder_data(monocular=monocular, area=area)
        distances  = agg.distances
        angle_axis = agg.angle_axis
        n_positions  = len(angle_axis)
        n_distances  = len(distances)

        if cell_idx is not None:
            X = X[cell_idx]

        a_pred = np.zeros((n_repeats, n_distances, n_positions))

        splitter = StratifiedShuffleSplit(n_splits=n_repeats, test_size=0.5)
        for rep, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(X.shape[2]), y_distance)):
            print(f'Repeat {rep + 1}/{n_repeats}', end='\r')

            dec = MLE(angle_axis).fit(X[:, :, train_idx])

            for d_idx, d_val in enumerate(distances):
                mask = y_distance[test_idx] == d_val
                test_d_idx = test_idx[mask]
                if len(test_d_idx) == 0:
                    a_pred[rep, d_idx, :] = np.nan
                    continue
                r_mean = X[:, :, test_d_idx].mean(axis=2)   # (n_neurons, n_angles)
                a_pred[rep, d_idx, :] = dec.decode_trial(r_mean)

        print()
        a_true = np.tile(angle_axis[None, :], (n_distances, 1))   # (n_distances, n_positions)
        a_error_by_pos = np.nanmean(np.abs(a_pred - a_true[np.newaxis]), axis=(0, 1))

        return dict(
            a_pred         = a_pred,
            a_true         = a_true,
            angle_axis     = angle_axis,
            distances      = distances,
            a_error_by_pos = a_error_by_pos,
        )

    @staticmethod
    def cross_validate_single_trial(agg, monocular, area=None, session=None,
                                    n_folds=5, random_state=None,
                                    cell_idx=None, angle_ranges=None):
        """Cross-validated MLE decoding of single trials.

        Same interface as cross_validate_averaged but decodes individual trials
        rather than condition averages.

        Parameters
        ----------
        agg       : SessionAggregator
        monocular : int   0 = binocular, 1 = monocular
        area      : str   visual area filter (None = all)
        session   : str   session filter — use to keep trials within a single session
        n_folds   : int
        random_state : int or None
        cell_idx  : array-like or None
        angle_ranges : list of (lo, hi) tuples in degrees, or None
        """
        session = _resolve_session(agg, session)
        X, y_distance = agg.get_decoder_data(monocular=monocular, area=area, session=session)
        if cell_idx is not None:
            X = X[cell_idx]
        return MLE(agg.angle_axis).cross_validate(
            X, n_folds=n_folds, stratify_labels=y_distance,
            random_state=random_state, angle_ranges=angle_ranges,
        )

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
        angle_axis = results['angle_axis']
        decoded = results['a_pred'][0]   # (n_rows, n_angles) — drop the repeat dim
        actual  = results['a_true']      # (n_rows, n_angles)
        metrics = results['metrics']

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # 1. Decoded vs actual (scatter density)
        ax = axes[0]
        ax.scatter(actual.flatten(), decoded.flatten(), alpha=0.5, s=1, color='k')
        ax.plot([angle_axis[0], angle_axis[-1]], [angle_axis[0], angle_axis[-1]], 'r--', linewidth=1)
        ax.set_xlabel('Actual angle (°)')
        ax.set_ylabel('Decoded angle (°)')
        ax.set_title(f'Decoded vs Actual (R²={metrics["r2"]:.3f})')
        ax.set_aspect('equal')

        # 2. Mean decoded trajectory
        ax = axes[1]
        mean_decoded = np.mean(decoded, axis=0)
        std_decoded = np.std(decoded, axis=0)
        ax.plot(angle_axis, angle_axis, 'r--', linewidth=1, label='Perfect')
        ax.plot(angle_axis, mean_decoded, 'b-', linewidth=2, label='Mean decoded')
        ax.fill_between(angle_axis, mean_decoded - std_decoded, mean_decoded + std_decoded,
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
        ax.plot(angle_axis, mae_per_angle, 'b-', linewidth=2)
        ax.fill_between(angle_axis, mae_per_angle - sem_per_angle, mae_per_angle + sem_per_angle,
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

        angle_axis = results1['angle_axis']
        metrics1 = results1['metrics']
        metrics2 = results2['metrics']

        # a_pred shape: (1, n_rows, n_angles) degrees; drop repeat dim
        decoded1 = results1['a_pred'][0]   # (n_rows, n_angles)
        actual1  = results1['a_true']      # (n_rows, n_angles)
        decoded2 = results2['a_pred'][0]
        actual2  = results2['a_true']

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
        ax.plot(angle_axis, mae_angle1, 'b-', linewidth=2, label=labels[0])
        ax.fill_between(angle_axis, mae_angle1 - sem1, mae_angle1 + sem1, alpha=0.2, color='blue')
        ax.plot(angle_axis, mae_angle2, '-', color='orange', linewidth=2, label=labels[1])
        ax.fill_between(angle_axis, mae_angle2 - sem2, mae_angle2 + sem2, alpha=0.2, color='orange')
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
        ax.text(0.1, 0.5, textstr, fontsize=20, verticalalignment='center',
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

class FactoredMLE:
    """Joint angle+distance decoder using a factored template model.

    Template: mu(c, a, d) = mu_angle(c, a) * gain(c, d)
      - mu_angle(c, a): RF profile — mean response of cell c at position a across distances
      - gain(c, d)    : distance gain at RF peak, mean-normalised

    Distance-invariant cells have gain ≈ [1,1,1,1] → templates identical across d
    → contribute nothing to the distance likelihood → drop out naturally.
    """

    def fit(self, X_cond, sigma2=None):
        """Fit from condition-averaged data.

        Parameters
        ----------
        X_cond : (n_neurons, n_positions, n_distances)
        sigma2 : (n_neurons,) or None
            Per-neuron noise variance. If None, uniform variance (equivalent to sigma2=1)
            is assumed. Use build_noise_variance to estimate from training trials.
        """
        n_neurons = X_cond.shape[0]
        self.mu_angle  = X_cond.mean(axis=2)                              # (n_neurons, n_positions)
        rf_pos         = self.mu_angle.argmax(axis=1)                     # (n_neurons,)
        rf_resp        = X_cond[np.arange(n_neurons), rf_pos, :]          # (n_neurons, n_distances)
        rf_mean        = rf_resp.mean(axis=1, keepdims=True) + 1e-10
        self.gain      = rf_resp / rf_mean                                # (n_neurons, n_distances)
        self.templates = self.mu_angle[:, :, np.newaxis] * self.gain[:, np.newaxis, :]
        # templates: (n_neurons, n_positions, n_distances)
        # self.sigma2    = sigma2 if sigma2 is not None else np.ones(n_neurons)
        self.sigma2    = np.ones(n_neurons)
        
        return self

    def decode(self, r):
        """Decode a single population snapshot.

        Parameters
        ----------
        r : (n_neurons,)

        Returns
        -------
        a_hat  : int   decoded position index
        d_hat  : int   decoded distance index (hard argmax)
        scores : (n_positions, n_distances)  log-likelihood grid
        """
        residuals = r[:, np.newaxis, np.newaxis] - self.templates
        scores    = -0.5 * (residuals ** 2 / self.sigma2[:, np.newaxis, np.newaxis]).sum(axis=0)
        a_hat, d_hat = np.unravel_index(scores.argmax(), scores.shape)
        return int(a_hat), int(d_hat), scores

    def decode_all(self, X_cond_test):
        """Decode every (position, distance) condition snapshot."""
        _, n_pos, n_dist = X_cond_test.shape
        a_pred, d_pred, a_true, d_true = [], [], [], []
        for a in range(n_pos):
            for d in range(n_dist):
                ah, dh, _ = self.decode(X_cond_test[:, a, d])
                a_pred.append(ah); d_pred.append(dh)
                a_true.append(a);  d_true.append(d)
        return (np.array(a_pred), np.array(d_pred),
                np.array(a_true), np.array(d_true))


class TrialDecoder:
    """Cross-validated single-trial decoder.

    Parameters
    ----------
    model : class, default FactoredMLE
        A class whose instances expose .fit(X) and .decode(r) ->
        (angle_idx, dist_idx, scores).
    """

    def __init__(self, model=None):
        self.model = model if model is not None else FactoredMLE

    @staticmethod
    def _default_ranges(angle_axis):
        lo, hi = float(angle_axis[0]), float(angle_axis[-1])
        t1 = lo + (hi - lo) / 3
        t2 = lo + 2 * (hi - lo) / 3
        return [(lo, t1), (t1, t2), (t2, hi)]

    def cross_validate(self, agg, monocular, area=None, session=None,
                       n_folds=5, loo=False, random_state=None,
                       cell_idx=None, verbose=True):
        """Cross-validated single-trial decoding.

        Parameters
        ----------
        agg          : SessionAggregator  (must have .angle_axis and .distances)
        monocular    : int   0 = binocular, 1 = monocular
        area         : str   visual area (None = all areas)
        session      : int or None   integer session number
        n_folds      : int   number of CV folds (default 5); ignored when loo=True
        loo          : bool  if True, use leave-one-out CV instead of k-fold (default False)
        random_state : int or None
        cell_idx     : array-like or None   neuron subset (None = all)
        verbose      : bool  print progress and summary (default True)

        Returns
        -------
        dict with keys:
            pred_dist        shape (n_trials, n_positions)
            pred_angle       shape (n_trials, n_positions)
            true_dist        shape (n_trials,)
            true_angle       shape (n_trials, n_positions)
            angle_axis       shape (n_positions,)
            distances        shape (n_distances,)
            angle_mae_by_pos shape (n_positions,)
            dist_acc_by_pos  shape (n_positions,)
        """
        session = _resolve_session(agg, session)
        X, y_distance = agg.get_decoder_data(monocular=monocular, area=area,
                                              session=session)
        distances   = agg.distances
        angle_axis  = agg.angle_axis
        n_positions = len(angle_axis)
        n_trials    = X.shape[2]

        if cell_idx is not None:
            X = X[cell_idx]

        pred_dist_idx  = np.zeros((n_trials, n_positions), dtype=int)
        pred_angle_idx = np.zeros((n_trials, n_positions), dtype=int)

        if loo:
            splitter = LeaveOneOut()
            splits   = list(splitter.split(np.zeros(n_trials)))
            n_splits = n_trials
        else:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                                       random_state=random_state)
            splits   = list(splitter.split(np.zeros(n_trials), y_distance))
            n_splits = n_folds

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if verbose:
                label = f"Trial {fold_idx + 1}/{n_splits}" if loo else f"Fold {fold_idx + 1}/{n_splits}"
                print(label, end="\r")

            X_cond_train = np.stack(
                [X[:, :, train_idx[y_distance[train_idx] == d]].mean(axis=2)
                 for d in distances],
                axis=2,
            )  # (n_neurons, n_positions, n_distances)

            dec = self.model().fit(X_cond_train)

            for t in test_idx:
                for angle_idx in range(n_positions):
                    r = X[:, angle_idx, t]
                    pred_a, pred_d, _ = dec.decode(r)
                    pred_angle_idx[t, angle_idx] = pred_a
                    pred_dist_idx[t, angle_idx]  = pred_d

        true_dist  = y_distance
        true_angle = np.tile(angle_axis[None, :], (n_trials, 1))

        pred_dist  = distances[pred_dist_idx]
        pred_angle = angle_axis[pred_angle_idx]

        angle_mae_by_pos = np.mean(np.abs(pred_angle - true_angle), axis=0)
        dist_acc_by_pos  = np.mean(pred_dist == true_dist[:, np.newaxis], axis=0)

        res = dict(
            pred_dist        = pred_dist,
            pred_angle       = pred_angle,
            true_dist        = true_dist,
            true_angle       = true_angle,
            angle_axis       = angle_axis,
            distances        = distances,
            angle_mae_by_pos = angle_mae_by_pos,
            dist_acc_by_pos  = dist_acc_by_pos,
        )
        if verbose:
            print()
            self._print_results(res)
        return res

    @staticmethod
    def _print_results(res):
        print("=" * 40)
        if "angle_mae_by_pos" in res:
            print(f"  Angle MAE:    {res['angle_mae_by_pos'].mean():.2f}°")
        if "dist_acc_by_pos" in res:
            print(f"  Dist Acc:     {res['dist_acc_by_pos'].mean():.1%}")
        print("=" * 40)

    def plot_distance_results(self, bino_res, mono_res, title=''):
        """Distance decoding summary: accuracy by position (row 1) and confusion matrices by angle range (row 2).

        Mirrors AveragedDecoder.plot_distance_results but for single-trial data.

        Row 1 : accuracy vs angle position for both conditions overlaid,
                with SEM band across trials.
        Row 2 : confusion matrices — rows = condition (bino / mono), cols = angle range.

        Parameters
        ----------
        bino_res : dict  output of self.cross_validate (binocular)
        mono_res : dict  output of self.cross_validate (monocular)
        title    : str   optional figure suptitle

        Returns
        -------
        matplotlib.figure.Figure
        """
        angle_ranges = [(-30, -15), (-15, 15), (15, 45)]
        range_colors = ['firebrick', 'mediumpurple', 'steelblue']
        range_cmaps  = ['Reds',      'Purples',      'Blues'    ]
        cond_colors  = ['steelblue', 'firebrick']
        cond_labels  = ['Binocular', 'Monocular']

        angle_axis  = bino_res['angle_axis']
        distances   = bino_res['distances']
        n_distances = len(distances)
        chance      = 100 / n_distances

        fig, axes = plt.subplot_mosaic(
            [['top', 'top', 'top'],
             ['b0',  'b1',  'b2' ],
             ['m0',  'm1',  'm2' ]],
            figsize=(15, 15),
            gridspec_kw={'height_ratios': [1.5, 1, 1], 'hspace': 0.5, 'wspace': 0},
        )

        # Row 0: accuracy by angle position (both conditions on one plot)
        ax = axes['top']
        for res, color, label in zip([bino_res, mono_res], cond_colors, cond_labels):
            pred     = res['pred_dist']   # (n_trials, n_positions)
            true     = res['true_dist']   # (n_trials,)
            n_trials = len(true)

            # per-trial accuracy at each position: shape (n_trials, n_positions)
            correct_by_trial = (pred == true[:, np.newaxis]).astype(float) * 100
            acc_mean = correct_by_trial.mean(axis=0)
            acc_sem  = correct_by_trial.std(axis=0) / np.sqrt(n_trials)

            ax.plot(angle_axis, acc_mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(angle_axis, acc_mean - acc_sem, acc_mean + acc_sem,
                            color=color, alpha=0.2)

        for i, (lo, hi) in enumerate(angle_ranges):
            ax.axvspan(lo, hi, color=range_colors[i], alpha=0.08, zorder=0)
        ax.axhline(chance, color='gray', linestyle=':', linewidth=1,
                   label=f'Chance ({chance:.0f}%)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set(xlabel='Angle (°)', ylabel='Accuracy (%)', ylim=(0, 100))
        ax.set_title('Distance accuracy by position', pad=25)
        ax.legend(fontsize=12)

        # Rows 1-2: confusion matrices (row=condition, col=range)
        cm_keys = [['b0', 'b1', 'b2'], ['m0', 'm1', 'm2']]
        for res, cond_label, keys in zip([bino_res, mono_res], cond_labels, cm_keys):
            pred = res['pred_dist']   # (n_trials, n_positions)
            true = res['true_dist']   # (n_trials,)

            for col, ((lo, hi), color, cmap, key) in enumerate(
                    zip(angle_ranges, range_colors, range_cmaps, keys)):
                ax        = axes[key]
                pos_mask  = (angle_axis >= lo) & (angle_axis < hi)
                d_true_flat = np.repeat(true, pos_mask.sum())
                d_pred_flat = pred[:, pos_mask].ravel()

                cm = confusion_matrix(d_true_flat, d_pred_flat,
                                      labels=distances, normalize='true')
                disp = ConfusionMatrixDisplay(cm, display_labels=[f'{d}' for d in distances])
                disp.plot(ax=ax, colorbar=False, cmap=cmap)
                for text in disp.text_.ravel():
                    text.set_text(f'{float(text.get_text()):.1f}' if text.get_text() else '')

                acc = (d_pred_flat == d_true_flat).mean() * 100
                ax.set_title(f'{cond_label}  {lo}° to {hi}°\nacc={acc:.1f}%', color=color)
                if col > 0:
                    ax.set_ylabel('')

        fig.suptitle(title, fontsize=22, y=0.95)
        plt.tight_layout()
        return fig

    def plot_distance_comparison(self, bino_res, mono_res, title="",
                                  metric="accuracy", angle_ranges=None):
        """Compare binocular vs monocular distance decoding as a function of angle.

        1x2 layout:
          Left  : dist_acc_by_pos (or MAE) line plot per condition with zone bands
          Right : per-trial accuracy per zone - grouped boxplot + t-test bars

        Parameters
        ----------
        bino_res     : dict  output of self.cross_validate (binocular)
        mono_res     : dict  output of self.cross_validate (monocular)
        title        : str
        metric       : "accuracy" (default) or "mae"
        angle_ranges : list of (lo, hi) degree tuples or None (3 equal zones)

        Returns
        -------
        matplotlib.figure.Figure
        """
        from matplotlib.patches import Patch
        from scipy.stats import ttest_ind

        angle_axis  = bino_res["angle_axis"]
        distances   = bino_res["distances"]
        n_distances = len(distances)

        if angle_ranges is None:
            angle_ranges = self._default_ranges(angle_axis)

        range_colors = ["firebrick", "mediumpurple", "steelblue"]
        range_labels = [f"[{lo:.0f}°, {hi:.0f}°)" for lo, hi in angle_ranges]
        conditions   = [("Binocular", bino_res, "steelblue"),
                        ("Monocular", mono_res, "firebrick")]

        def _sig_label(p):
            if p > 0.05:  return "ns"
            if p > 0.01:  return "*"
            if p > 0.001: return "**"
            return "***"

        fig, (ax_line, ax_box) = plt.subplots(
            1, 2, figsize=(16, 4), gridspec_kw={"width_ratios": [2, 1.2]})

        for i, (lo, hi) in enumerate(angle_ranges):
            ax_line.axvspan(lo, hi,
                            color=range_colors[i % len(range_colors)],
                            alpha=0.08, zorder=0)

        n_conds = len(conditions)
        x_pos   = np.arange(1, len(angle_ranges) + 1)
        width   = 0.6 / n_conds
        step    = width * 1.1
        starts  = np.arange(-(n_conds - 1), n_conds, 2) * step / 2
        box_data_all = []

        for k, (name, res, color) in enumerate(conditions):
            pred_dist = res["pred_dist"]   # (n_trials, n_positions)
            true_dist = res["true_dist"]   # (n_trials,)
            n_trials  = len(true_dist)

            if metric == "accuracy":
                metric_by_pos = res["dist_acc_by_pos"] * 100
            else:
                metric_by_pos = np.mean(
                    np.abs(pred_dist - true_dist[:, np.newaxis]), axis=0)

            ax_line.plot(angle_axis, metric_by_pos, color=color,
                         linewidth=1.8, label=name)

            zone_data = []
            for lo, hi in angle_ranges:
                zmask = (angle_axis >= lo) & (angle_axis < hi)
                if metric == "accuracy":
                    data = np.array(
                        [(pred_dist[t, zmask] == true_dist[t]).mean() * 100
                         for t in range(n_trials)])
                else:
                    data = np.array(
                        [np.abs(pred_dist[t, zmask] - true_dist[t]).mean()
                         for t in range(n_trials)])
                zone_data.append(data)
            box_data_all.append(zone_data)

            for i, data in enumerate(zone_data):
                bp = ax_box.boxplot(
                    data, positions=[x_pos[i] + starts[k]], widths=width * 0.9,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker="o", markersize=3, color=color,
                                   alpha=0.5))
                bp["boxes"][0].set(facecolor=color, alpha=0.7)

        if metric == "accuracy":
            chance = 100 / n_distances
            ax_line.axhline(chance, color="gray", linestyle=":", linewidth=1,
                            label=f"Chance ({chance:.0f}%)")
            ax_line.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax_line.set(xlabel="Angle (°)", ylabel="Accuracy (%)",
                        title="Distance accuracy by position", ylim=(0, 100))
            ax_box.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
            ax_box.set(ylabel="Accuracy (%)", title="Distance accuracy by zone",
                       ylim=(0, 100))
        else:
            chance = np.mean(np.abs(distances[:, None] - distances[None, :]))
            ax_line.axhline(chance, color="gray", linestyle=":", linewidth=1,
                            label=f"Chance ({chance:.2f})")
            ax_line.set(xlabel="Angle (°)", ylabel="Distance MAE",
                        title="Distance MAE by position")
            ax_box.set(ylabel="Distance MAE", title="Distance MAE by zone")

        ax_line.legend()

        for i in range(len(angle_ranges)):
            d_bino = box_data_all[0][i]
            d_mono = box_data_all[1][i]
            _, p   = ttest_ind(d_bino, d_mono)
            y_max  = max(d_bino.max(), d_mono.max())
            y_step = y_max * 0.18
            y      = y_max + y_step
            x1     = x_pos[i] + starts[0]
            x2     = x_pos[i] + starts[1]
            ax_box.plot([x1, x1, x2, x2],
                        [y - y_step * 0.2, y, y, y - y_step * 0.2],
                        color="k", linewidth=1.2)
            ax_box.text((x1 + x2) / 2, y + y_step * 0.05, _sig_label(p),
                        ha="center", va="bottom", fontsize=10)

        ax_box.set_xticks(x_pos)
        ax_box.set_xticklabels(range_labels, rotation=15)
        ax_box.legend(
            handles=[Patch(facecolor=c, alpha=0.7, label=n)
                     for n, _, c in conditions], fontsize=9)

        if title:
            fig.suptitle(title, fontsize=20)
        plt.tight_layout()
        return fig

    def plot_confusion_by_angle_range(self, results, labels=None, title="",
                                       angle_ranges=None):
        """Grid of distance confusion matrices split by angle range.

        Layout:
          rows  - one per result set
          col 0...n_ranges-1 - one zone each
          last col           - all positions pooled (overall)

        Parameters
        ----------
        results      : dict or list of dict  output of self.cross_validate
        labels       : str or list of str, optional
        title        : str, optional
        angle_ranges : list of (lo, hi) degree tuples or None (3 equal zones)

        Returns
        -------
        matplotlib.figure.Figure
        """
        if isinstance(results, dict):
            results = [results]
        if labels is None:
            labels = [f"Result {i + 1}" for i in range(len(results))]
        elif isinstance(labels, str):
            labels = [labels]

        angle_axis = results[0]["angle_axis"]

        if angle_ranges is None:
            angle_ranges = self._default_ranges(angle_axis)

        range_cmaps  = ["Reds", "Purples", "Blues", "Oranges"]
        range_labels = [f"[{lo:.0f}°, {hi:.0f}°)" for lo, hi in angle_ranges]
        n_ranges     = len(angle_ranges)
        n_rows       = len(results)
        n_cols       = n_ranges + 1

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 4.5 * n_rows),
            squeeze=False,
            gridspec_kw={"hspace": 0.85},
        )

        for r, (res, label) in enumerate(zip(results, labels)):
            pred_dist  = res["pred_dist"]   # (n_trials, n_positions)
            true_dist  = res["true_dist"]   # (n_trials,)
            distances  = res["distances"]
            dist_labels = [f"{d}" for d in distances]

            for c, ((lo, hi), cmap, rlabel) in enumerate(
                    zip(angle_ranges, range_cmaps, range_labels)):
                ax    = axes[r, c]
                zmask = (angle_axis >= lo) & (angle_axis < hi)
                d_true_flat = np.repeat(true_dist, zmask.sum())
                d_pred_flat = pred_dist[:, zmask].ravel()

                cm = confusion_matrix(d_true_flat, d_pred_flat,
                                      labels=distances, normalize="true")
                ConfusionMatrixDisplay(cm, display_labels=dist_labels).plot(
                    ax=ax, colorbar=False, cmap=cmap)
                acc = (d_pred_flat == d_true_flat).mean() * 100
                ax.set_title(f"{rlabel}\nacc={acc:.1f}%")
                ax.set_ylabel(label if c == 0 else "", fontsize=12, labelpad=8)

            # Overall (all positions pooled)
            ax = axes[r, -1]
            d_true_all = np.repeat(true_dist, len(angle_axis))
            d_pred_all = pred_dist.ravel()
            cm = confusion_matrix(d_true_all, d_pred_all,
                                  labels=distances, normalize="true")
            ConfusionMatrixDisplay(cm, display_labels=dist_labels).plot(
                ax=ax, colorbar=False, cmap="Greens")
            acc = (d_pred_all == d_true_all).mean() * 100
            ax.set_title(f"Overall\nacc={acc:.1f}%")
            ax.set_ylabel("")

        if title:
            fig.suptitle(title, fontsize=24, y=1.02)
        plt.tight_layout()
        return fig


class AveragedDecoder:
    """Cross-validated averaged-trial decoder.

    Parameters
    ----------
    model : class, default FactoredMLE
        A class whose instances expose .fit(X) and .decode(r) ->
        (angle_idx, dist_idx, scores).
    """

    def __init__(self, model=None):
        self.model = model if model is not None else FactoredMLE

    def cross_validate(self, agg, monocular, area=None, n_repeats=10,
                       cell_idx=None, verbose=True):
        """Cross-validated averaged-condition decoding.

        Parameters
        ----------
        agg       : SessionAggregator  (must have .angle_axis and .distances)
        monocular : int   0 = binocular, 1 = monocular
        area      : str   visual area (None = all areas)
        n_repeats : int
        cell_idx  : array-like or None   neuron subset (None = all)
        verbose   : bool  print progress and summary (default True)

        Returns
        -------
        dict with keys:
            pred_dist        shape (n_repeats, n_distances, n_positions)
            pred_angle       shape (n_repeats, n_distances, n_positions)
            true_dist        shape (n_distances, n_positions)
            true_angle       shape (n_distances, n_positions)
            angle_axis       shape (n_positions,)
            distances        shape (n_distances,)
            angle_mae_by_pos shape (n_positions,)
            dist_acc_by_pos  shape (n_positions,)
        """
        distances   = agg.distances
        angle_axis  = agg.angle_axis
        n_positions = agg.n_positions
        n_distances = len(distances)

        pred_dist_idx_grid  = np.zeros((n_repeats, n_distances, n_positions), dtype=int)
        pred_angle_idx_grid = np.zeros((n_repeats, n_distances, n_positions), dtype=int)

        for rep in range(n_repeats):
            if verbose:
                print(f"Repeat {rep + 1}/{n_repeats}", end="\r")
            X_train, X_test, _ = agg.get_split_decoder_data(monocular=monocular,
                                                             area=area)
            if cell_idx is not None:
                X_train = X_train[cell_idx]
                X_test  = X_test[cell_idx]

            dec = self.model().fit(X_train)

            for dist_idx in range(n_distances):
                for angle_idx in range(n_positions):
                    r = X_test[:, angle_idx, dist_idx]
                    pred_a_idx, pred_d_idx, _ = dec.decode(r)
                    pred_dist_idx_grid[rep, dist_idx, angle_idx]  = pred_d_idx
                    pred_angle_idx_grid[rep, dist_idx, angle_idx] = pred_a_idx

        true_dist  = np.tile(distances[:, None],  (1, n_positions))
        true_angle = np.tile(angle_axis[None, :], (n_distances, 1))

        pred_dist  = distances[pred_dist_idx_grid]
        pred_angle = angle_axis[pred_angle_idx_grid]

        angle_mae_by_pos = np.mean(
            np.abs(pred_angle - true_angle[np.newaxis]), axis=(0, 1))
        dist_acc_by_pos  = np.mean(
            pred_dist == true_dist[np.newaxis], axis=(0, 1))

        res = dict(
            pred_dist        = pred_dist,
            pred_angle       = pred_angle,
            true_dist        = true_dist,
            true_angle       = true_angle,
            angle_axis       = angle_axis,
            distances        = distances,
            angle_mae_by_pos = angle_mae_by_pos,
            dist_acc_by_pos  = dist_acc_by_pos,
        )
        if verbose:
            print()
            self._print_results(res)
        return res

    @staticmethod
    def _print_results(res):
        print("=" * 40)
        if "angle_mae_by_pos" in res:
            print(f"  Angle MAE:    {res['angle_mae_by_pos'].mean():.2f}°")
        if "dist_acc_by_pos" in res:
            print(f"  Dist Acc:     {res['dist_acc_by_pos'].mean():.1%}")
        print("=" * 40)

    def plot_distance_results(self, bino_res, mono_res, title=''):
        """Distance decoding summary: accuracy by position (row 1) and confusion matrices by angle range (row 2).

        Row 1 : single plot with both conditions overlaid showing distance
                classification accuracy vs angle position, with SEM band across repeats.
        Row 2 : confusion matrices — rows = condition (bino / mono), cols = angle range.

        Parameters
        ----------
        bino_res : dict  output of self.cross_validate (binocular)
        mono_res : dict  output of self.cross_validate (monocular)
        title    : str   optional figure suptitle

        Returns
        -------
        matplotlib.figure.Figure
        """
        angle_ranges = [(-30, -15), (-15, 15), (15, 45)]
        range_colors = ['firebrick', 'mediumpurple', 'steelblue']
        range_cmaps  = ['Reds',      'Purples',      'Blues'    ]
        cond_colors  = ['steelblue', 'firebrick']
        cond_labels  = ['Binocular', 'Monocular']

        angle_axis  = bino_res['angle_axis']
        distances   = bino_res['distances']
        n_distances = len(distances)
        n_repeats   = bino_res['pred_dist'].shape[0]
        chance      = 100 / n_distances

        fig, axes = plt.subplot_mosaic(
            [['top', 'top', 'top'],
             ['b0',  'b1',  'b2' ],
             ['m0',  'm1',  'm2' ]],
            figsize=(15, 15),
            gridspec_kw={'height_ratios': [1.5, 1, 1], 'hspace': 0.5, 'wspace': 0},
        )

        # Row 0: accuracy by angle position (both conditions on one plot)
        ax = axes['top']
        for res, color, label in zip([bino_res, mono_res], cond_colors, cond_labels):
            pred = res['pred_dist']   # (n_repeats, n_distances, n_positions)
            true = res['true_dist']   # (n_distances, n_positions)

            acc_per_rep = (pred == true[np.newaxis]).mean(axis=1) * 100
            acc_mean    = acc_per_rep.mean(axis=0)
            acc_sem     = acc_per_rep.std(axis=0) / np.sqrt(n_repeats)

            ax.plot(angle_axis, acc_mean, color=color, linewidth=1.8, label=label)
            ax.fill_between(angle_axis, acc_mean - acc_sem, acc_mean + acc_sem,
                            color=color, alpha=0.2)

        for i, (lo, hi) in enumerate(angle_ranges):
            ax.axvspan(lo, hi, color=range_colors[i], alpha=0.08, zorder=0)
        ax.axhline(chance, color='gray', linestyle=':', linewidth=1,
                   label=f'Chance ({chance:.0f}%)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.set(xlabel='Angle (°)', ylabel='Accuracy (%)', ylim=(0, 100))
        ax.set_title('Distance accuracy by position', pad=25)
        ax.legend(fontsize=12)

        # Rows 1-2: confusion matrices (row=condition, col=range)
        cm_keys = [['b0', 'b1', 'b2'], ['m0', 'm1', 'm2']]
        for res, cond_label, keys in zip([bino_res, mono_res], cond_labels, cm_keys):
            pred = res['pred_dist']   # (n_repeats, n_distances, n_positions)
            true = res['true_dist']   # (n_distances, n_positions)

            for col, ((lo, hi), color, cmap, key) in enumerate(
                    zip(angle_ranges, range_colors, range_cmaps, keys)):
                ax        = axes[key]
                pos_mask  = (angle_axis >= lo) & (angle_axis < hi)
                pred_flat = pred[:, :, pos_mask].ravel()
                true_flat = np.tile(true[:, pos_mask][np.newaxis], (n_repeats, 1, 1)).ravel()

                cm = confusion_matrix(true_flat, pred_flat,
                                      labels=distances, normalize='true')
                disp = ConfusionMatrixDisplay(cm, display_labels=[f'{d}' for d in distances])
                disp.plot(ax=ax, colorbar=False, cmap=cmap)
                for text in disp.text_.ravel():
                    text.set_text(f'{float(text.get_text()):.1f}' if text.get_text() else '')

                acc = (pred_flat == true_flat).mean() * 100
                ax.set_title(f'{cond_label}  {lo}° to {hi}°\nacc={acc:.1f}%', color=color)
                if col > 0:
                    ax.set_ylabel('')

        fig.suptitle(title, fontsize=22, y=0.95)
        plt.tight_layout()
        return fig

    def plot_distance_comparison(self, bino_res, mono_res, title="", metric="accuracy",
                                  angle_ranges=None):
        raise NotImplementedError(
            "plot_distance_comparison is not yet implemented for AveragedDecoder")

    def plot_confusion_by_angle_range(self, results, labels=None, title="",
                                       angle_ranges=None):
        raise NotImplementedError(
            "plot_confusion_by_angle_range is not yet implemented for AveragedDecoder")
