import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold


class _MultiHeadMLP(nn.Module):
    """Angle-aware multi-head MLP for joint distance/angle decoding."""

    def __init__(self, input_size, hidden_sizes=(128, 64), head_size=32,
                 n_distances=4, dropout=0.3):
        super().__init__()

        # Shared backbone
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # Distance classification head
        self.dist_head = nn.Sequential(
            nn.Linear(in_dim, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, n_distances),
        )

        # Angle regression head
        self.angle_head = nn.Sequential(
            nn.Linear(in_dim, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        dist_logits = self.dist_head(features)
        angle_pred = self.angle_head(features).squeeze(-1)
        return dist_logits, angle_pred


class NNDecoder:
    """Multi-head neural network decoder for joint distance and angle decoding.

    Compatible with SessionAggregator.get_decoder_data() output.
    Uses an angle-aware MLP: input is [neural_response, normalized_angle].
    Distance head outputs 4-class probabilities (posterior mean for continuous output).
    Angle head outputs angle regression.
    """

    def __init__(self, chunk_size=1,
                 hidden_sizes=(128, 64), head_size=32, dropout=0.3,
                 lr=1e-3, epochs=200, patience=15, angle_weight=1.0):
        """
        Parameters
        ----------
        chunk_size : int
            Number of angular positions to pool per sample.
        hidden_sizes : tuple of int
            Hidden layer sizes for shared backbone.
        head_size : int
            Hidden size in each output head.
        dropout : float
            Dropout rate.
        lr : float
            Learning rate for Adam optimizer.
        epochs : int
            Maximum training epochs.
        patience : int
            Early stopping patience (epochs without val improvement).
        angle_weight : float
            Weight for angle loss relative to distance loss.
        """
        self.distance_values = np.arange(4)
        self.n_distances = 4
        self.chunk_size = chunk_size

        # NN hyperparameters
        self.hidden_sizes = hidden_sizes
        self.head_size = head_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.angle_weight = angle_weight

        # Torch device
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Set after fit
        self.model = None
        self.n_neurons = None
        self.n_angles = None
        self.n_chunks = None
        self.norm_angles = None

    def _build_model(self):
        input_size = self.n_neurons + 1  # response + normalized angle
        model = _MultiHeadMLP(
            input_size=input_size,
            hidden_sizes=self.hidden_sizes,
            head_size=self.head_size,
            n_distances=self.n_distances,
            dropout=self.dropout,
        ).to(self.torch_device)
        return model

    def _get_chunk_indices(self):
        return [np.arange(i * self.chunk_size,
                          min((i + 1) * self.chunk_size, self.n_angles))
                for i in range(self.n_chunks)]

    def _get_chunk_centers(self):
        """Return center indices for each chunk (normalized 0-1)."""
        chunk_indices = self._get_chunk_indices()
        chunk_centers = np.array([np.mean(idx) for idx in chunk_indices])
        return chunk_centers / (self.n_angles - 1) if self.n_angles > 1 else np.array([0.5])

    def _build_training_data(self, X, y):
        """Build X, y_dist, y_angle arrays from training data.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
        y : np.ndarray, shape (n_trials,)

        Returns
        -------
        X_samples : array, shape (n_samples, n_neurons + 1)
        y_dist : array, shape (n_samples,) int
        y_angle : array, shape (n_samples,) float (normalized 0-1)
        """
        n_trials = X.shape[2]
        X_list, y_dist_list, y_angle_list = [], [], []

        for t in range(n_trials):
            responses = X[:, :, t]  # (n_neurons, n_angles)
            dist = y[t]

            if self.chunk_size == 1:
                for a_idx in range(self.n_angles):
                    feat = np.append(responses[:, a_idx], self.norm_angles[a_idx])
                    X_list.append(feat)
                    y_dist_list.append(dist)
                    y_angle_list.append(self.norm_angles[a_idx])
            else:
                for c, chunk_idxs in enumerate(self._get_chunk_indices()):
                    avg_response = np.mean(responses[:, chunk_idxs], axis=1)
                    feat = np.append(avg_response, self.norm_angles[c])
                    X_list.append(feat)
                    y_dist_list.append(dist)
                    y_angle_list.append(self.norm_angles[c])

        return (np.array(X_list, dtype=np.float32),
                np.array(y_dist_list, dtype=np.int64),
                np.array(y_angle_list, dtype=np.float32))

    def fit(self, X_train, y_train):
        """Train the multi-head MLP on training data.

        Parameters
        ----------
        X_train : np.ndarray
            Training data, shape (n_neurons, n_angles, n_trials).
        y_train : np.ndarray
            Distance labels, shape (n_trials,).

        Returns
        -------
        self : NNDecoder
            Fitted decoder instance.
        """
        self.n_neurons = X_train.shape[0]
        self.n_angles = X_train.shape[1]
        self.n_chunks = int(np.ceil(self.n_angles / self.chunk_size))

        # Normalized angle values for each chunk (0 to 1)
        self.norm_angles = self._get_chunk_centers()

        X, y_dist, y_angle = self._build_training_data(X_train, y_train)

        # Stratified 80/20 split for early stopping
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train_t = torch.tensor(X[train_idx], device=self.torch_device)
        y_dist_train = torch.tensor(y_dist[train_idx], device=self.torch_device)
        y_angle_train = torch.tensor(y_angle[train_idx], device=self.torch_device)

        X_val = torch.tensor(X[val_idx], device=self.torch_device)
        y_dist_val = torch.tensor(y_dist[val_idx], device=self.torch_device)
        y_angle_val = torch.tensor(y_angle[val_idx], device=self.torch_device)

        train_ds = TensorDataset(X_train_t, y_dist_train, y_angle_train)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        dist_criterion = nn.CrossEntropyLoss()
        angle_criterion = nn.MSELoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
            self.model.train()
            for X_batch, y_d_batch, y_a_batch in train_loader:
                dist_logits, angle_pred = self.model(X_batch)
                loss = dist_criterion(dist_logits, y_d_batch) + \
                       self.angle_weight * angle_criterion(angle_pred, y_a_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validate
            self.model.eval()
            with torch.no_grad():
                dist_logits, angle_pred = self.model(X_val)
                val_loss = dist_criterion(dist_logits, y_dist_val) + \
                           self.angle_weight * angle_criterion(angle_pred, y_angle_val)
                val_loss = val_loss.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.torch_device) for k, v in best_state.items()})
        self.model.eval()

        return self

    def predict_probability_map(self, response_vector):
        """Get distance probability at every angle for a single neural response.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
            Neural activity from a single time-point.

        Returns
        -------
        prob_matrix : np.ndarray, shape (n_angles, n_distances)
            prob_matrix[a, d] = P(distance=d | response, angle=a).
        angle_values : np.ndarray, shape (n_angles,)
            Normalized angle values (0 to 1) for each row.
        predicted_angles : np.ndarray, shape (n_angles,)
            Angle head regression output (normalized 0-1) at each queried angle.
        """
        n = len(self.norm_angles)
        response = np.asarray(response_vector, dtype=np.float32)

        feats = np.column_stack([
            np.tile(response, (n, 1)),
            self.norm_angles.reshape(-1, 1),
        ]).astype(np.float32)

        x = torch.tensor(feats, device=self.torch_device)
        with torch.no_grad():
            dist_logits, angle_preds = self.model(x)

        prob_matrix = torch.softmax(dist_logits, dim=-1).cpu().numpy()
        predicted_angles = angle_preds.cpu().numpy()
        return prob_matrix, self.norm_angles.copy(), predicted_angles

    def plot_probability_map(self, response_vector, angle_values_deg=None,
                             actual_distance=None, actual_angle_idx=None):
        """Plot the distance×angle probability matrix for a single response.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
            Neural activity from a single time-point.
        angle_values_deg : array-like, optional
            Angle values in degrees for y-axis labels. If None, uses indices.
        actual_distance : int, optional
            True distance (0-3) to mark on the plot.
        actual_angle_idx : int, optional
            True angle index to mark on the plot.

        Returns
        -------
        fig : matplotlib Figure
        prob_matrix : np.ndarray, shape (n_angles, n_distances)
        """
        prob_matrix, norm_angles, predicted_angles = self.predict_probability_map(response_vector)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                                 gridspec_kw={'width_ratios': [2, 1, 1.2]})

        # Left: heatmap (angles × distances)
        ax = axes[0]
        im = ax.imshow(prob_matrix, aspect='auto', cmap='hot',
                       origin='lower', vmin=0, vmax=1)
        ax.set_xlabel('Distance class')
        ax.set_xticks(range(self.n_distances))
        if angle_values_deg is not None:
            n_ticks = min(10, len(angle_values_deg))
            tick_idx = np.linspace(0, len(angle_values_deg) - 1, n_ticks, dtype=int)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([f'{angle_values_deg[i]:.0f}°' for i in tick_idx])
            ax.set_ylabel('Angle (deg)')
        else:
            ax.set_ylabel('Angle index')
        ax.set_title('P(distance | response, angle)')
        plt.colorbar(im, ax=ax, label='Probability')

        if actual_distance is not None and actual_angle_idx is not None:
            ax.plot(actual_distance, actual_angle_idx, 'c*', markersize=15,
                    markeredgecolor='white', markeredgewidth=1)

        # Center: marginal distance probability (averaged over angles)
        ax = axes[1]
        mean_probs = prob_matrix.mean(axis=0)
        ax.bar(range(self.n_distances), mean_probs, color='steelblue')
        ax.set_xlabel('Distance class')
        ax.set_ylabel('Mean probability')
        ax.set_title('Marginal P(distance)')
        ax.set_xticks(range(self.n_distances))
        ax.set_ylim([0, 1])
        if actual_distance is not None:
            ax.axvline(actual_distance, color='red', linestyle='--',
                       label=f'True dist={actual_distance}')
            ax.legend()

        # Right: predicted angle vs input angle
        ax = axes[2]
        if angle_values_deg is not None:
            input_angles = np.array(angle_values_deg)
            pred_angles_deg = predicted_angles * (input_angles[-1] - input_angles[0]) + input_angles[0]
            ax.plot(input_angles, pred_angles_deg, 'b-', linewidth=2, label='Predicted')
            ax.plot(input_angles, input_angles, 'r--', linewidth=1, label='Perfect')
            if actual_angle_idx is not None:
                ax.axvline(input_angles[actual_angle_idx], color='green', linestyle=':',
                           alpha=0.7, label=f'True angle={input_angles[actual_angle_idx]:.0f}°')
            ax.set_xlabel('Input angle (deg)')
            ax.set_ylabel('Predicted angle (deg)')
        else:
            ax.plot(norm_angles, predicted_angles, 'b-', linewidth=2, label='Predicted')
            ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Perfect')
            if actual_angle_idx is not None:
                ax.axvline(norm_angles[actual_angle_idx], color='green', linestyle=':',
                           alpha=0.7, label=f'True idx={actual_angle_idx}')
            ax.set_xlabel('Input angle (normalized)')
            ax.set_ylabel('Predicted angle (normalized)')
        ax.set_title('Angle prediction')
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()
        return fig, prob_matrix, predicted_angles

    def decode(self, response_vector, angle_idx):
        """Decode distance and angle from a single response vector.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
        angle_idx : int

        Returns
        -------
        decoded_distance : float (0-3 continuous via posterior mean)
        decoded_angle : float (normalized 0-1)
        """
        feat = np.append(response_vector, self.norm_angles[angle_idx]).astype(np.float32)
        x = torch.tensor(feat, device=self.torch_device).unsqueeze(0)

        with torch.no_grad():
            dist_logits, angle_pred = self.model(x)

        # Distance: softmax → posterior mean
        probs = torch.softmax(dist_logits, dim=-1).cpu().numpy()[0]
        decoded_dist = float(np.sum(self.distance_values * probs))

        # Angle: already normalized
        decoded_angle = float(angle_pred.cpu().numpy()[0])

        return decoded_dist, decoded_angle

    def decode_response_matrix(self, responses):
        """Batch decode distance and angle from a response matrix.

        Parameters
        ----------
        responses : array, shape (n_neurons, n_angles)

        Returns
        -------
        decoded_distances : list of float
        decoded_angles : list of float (normalized 0-1)
        """
        if self.chunk_size == 1:
            # Build batch: (n_angles, n_neurons + 1)
            feats = np.column_stack([
                responses.T,  # (n_angles, n_neurons)
                self.norm_angles.reshape(-1, 1)
            ]).astype(np.float32)
        else:
            chunks = self._get_chunk_indices()
            feat_list = []
            for c, chunk_idxs in enumerate(chunks):
                avg_response = np.mean(responses[:, chunk_idxs], axis=1)
                feat_list.append(np.append(avg_response, self.norm_angles[c]))
            feats = np.array(feat_list, dtype=np.float32)

        x = torch.tensor(feats, device=self.torch_device)
        with torch.no_grad():
            dist_logits, angle_preds = self.model(x)

        probs = torch.softmax(dist_logits, dim=-1).cpu().numpy()
        decoded_distances = [float(np.sum(self.distance_values * p)) for p in probs]

        angle_vals = angle_preds.cpu().numpy()
        decoded_angles = [float(a) for a in angle_vals]

        return decoded_distances, decoded_angles

    def decode_trial(self, trial_data):
        """Decode distance and angle at each position for a single trial.

        Parameters
        ----------
        trial_data : np.ndarray, shape (n_neurons, n_angles)
            Neural responses for one trial.

        Returns
        -------
        decoded_distances : list of float
        decoded_angles : list of float
        """
        return self.decode_response_matrix(trial_data)

    def cross_validate(self, X, y_distance, k_folds=10, random_state=None):
        """Run stratified k-fold cross-validation for distance and angle decoding.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Full dataset.
        y_distance : np.ndarray, shape (n_trials,)
            Distance labels for each trial.
        k_folds : int
            Number of CV folds.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict with keys:
            all_decoded_dist, all_actual_dist, all_angle_indices, dist_metrics,
            all_decoded_angle, all_actual_angle, angle_metrics
        """
        n_trials = X.shape[2]

        if random_state is not None:
            np.random.seed(random_state)

        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        splits = list(kfold.split(np.zeros(n_trials), y_distance))

        all_decoded_dist = []
        all_actual_dist = []
        all_decoded_angle = []
        all_actual_angle = []
        all_angle_indices = []

        # Initialize n_angles from data
        self.n_angles = X.shape[1]
        self.n_chunks = int(np.ceil(self.n_angles / self.chunk_size))
        self.norm_angles = self._get_chunk_centers()

        n_output = self.n_angles if self.chunk_size == 1 else self.n_chunks
        actual_angle_values = self.norm_angles

        for fold, (train_idx, test_idx) in enumerate(splits):
            print(f'=== Fold {fold + 1}/{k_folds} ===', end='\r', flush=True)

            X_train = X[:, :, train_idx]
            y_train = y_distance[train_idx]
            X_test = X[:, :, test_idx]
            y_test = y_distance[test_idx]

            self.fit(X_train, y_train)

            for t in range(len(test_idx)):
                trial_data = X_test[:, :, t]
                actual_dist = y_test[t]

                dec_dist, dec_angle = self.decode_trial(trial_data)
                all_decoded_dist.extend(dec_dist)
                all_actual_dist.extend([actual_dist] * len(dec_dist))
                all_decoded_angle.extend(dec_angle)
                all_actual_angle.extend(actual_angle_values)
                all_angle_indices.extend(range(n_output))

        all_decoded_dist = np.array(all_decoded_dist)
        all_actual_dist = np.array(all_actual_dist)
        all_decoded_angle = np.array(all_decoded_angle)
        all_actual_angle = np.array(all_actual_angle)
        all_angle_indices = np.array(all_angle_indices)

        dist_metrics = self.evaluate_distance(all_actual_dist, all_decoded_dist)
        angle_metrics = self.evaluate_angle(all_actual_angle, all_decoded_angle)

        print(f'\nTotal predictions: {len(all_decoded_dist)}')
        self.print_metrics(dist_metrics, angle_metrics)

        return {
            'all_decoded_dist': all_decoded_dist,
            'all_actual_dist': all_actual_dist,
            'all_decoded_angle': all_decoded_angle,
            'all_actual_angle': all_actual_angle,
            'all_angle_indices': all_angle_indices,
            'dist_metrics': dist_metrics,
            'angle_metrics': angle_metrics,
        }

    def cross_validate_mean(self, X, y_distance, n_repeats=100, random_state=None):
        """Run repeated 80/20 split CV, decoding from mean test responses per distance.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
            Full dataset.
        y_distance : np.ndarray, shape (n_trials,)
            Distance labels for each trial.
        n_repeats : int
            Number of random splits.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict (same format as cross_validate)
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize n_angles from data
        self.n_angles = X.shape[1]
        self.n_chunks = int(np.ceil(self.n_angles / self.chunk_size))
        self.norm_angles = self._get_chunk_centers()

        all_decoded_dist = []
        all_actual_dist = []
        all_decoded_angle = []
        all_actual_angle = []
        all_angle_indices = []
        n_output = self.n_angles if self.chunk_size == 1 else self.n_chunks
        actual_angle_values = self.norm_angles

        # Group trials by distance
        trials_by_dist = {}
        for dist in range(self.n_distances):
            trials_by_dist[dist] = np.where(y_distance == dist)[0]

        for rep in range(n_repeats):
            print(f'Repeat {rep + 1}/{n_repeats}', end='\r', flush=True)

            train_idx_list = []
            test_idx_by_dist = {}

            for dist in range(self.n_distances):
                idxs = np.random.permutation(trials_by_dist[dist])
                split = int(0.8 * len(idxs))
                train_idx_list.extend(idxs[:split])
                test_idx_by_dist[dist] = idxs[split:]

            train_idx = np.array(train_idx_list)
            X_train = X[:, :, train_idx]
            y_train = y_distance[train_idx]

            self.fit(X_train, y_train)

            for dist in range(self.n_distances):
                test_idx = test_idx_by_dist[dist]
                test_responses = X[:, :, test_idx]  # (n_neurons, n_angles, n_test)
                mean_responses = np.mean(test_responses, axis=2)  # (n_neurons, n_angles)

                dec_dist, dec_angle = self.decode_response_matrix(mean_responses)
                all_decoded_dist.extend(dec_dist)
                all_actual_dist.extend([dist] * len(dec_dist))
                all_decoded_angle.extend(dec_angle)
                all_actual_angle.extend(actual_angle_values)
                all_angle_indices.extend(range(n_output))

        all_decoded_dist = np.array(all_decoded_dist)
        all_actual_dist = np.array(all_actual_dist)
        all_decoded_angle = np.array(all_decoded_angle)
        all_actual_angle = np.array(all_actual_angle)
        all_angle_indices = np.array(all_angle_indices)

        dist_metrics = self.evaluate_distance(all_actual_dist, all_decoded_dist)
        angle_metrics = self.evaluate_angle(all_actual_angle, all_decoded_angle)

        print(f'\nTotal predictions: {len(all_decoded_dist)}')
        print(f'Repeats: {n_repeats}, Distances: {self.n_distances}')
        self.print_metrics(dist_metrics, angle_metrics)

        return {
            'all_decoded_dist': all_decoded_dist,
            'all_actual_dist': all_actual_dist,
            'all_decoded_angle': all_decoded_angle,
            'all_actual_angle': all_actual_angle,
            'all_angle_indices': all_angle_indices,
            'dist_metrics': dist_metrics,
            'angle_metrics': angle_metrics,
        }

    @staticmethod
    def evaluate_distance(actual, decoded):
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

    @staticmethod
    def evaluate_angle(actual, decoded):
        actual = np.array(actual)
        decoded = np.array(decoded)
        errors = decoded - actual
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'bias': np.mean(errors),
            'correlation': np.corrcoef(actual.flatten(), decoded.flatten())[0, 1],
            'r2': r2,
            'std': np.std(errors),
        }

    @staticmethod
    def print_metrics(dist_metrics, angle_metrics):
        print("=" * 50)
        print("NN DECODER PERFORMANCE")
        print("=" * 50)
        print("  DISTANCE:")
        print(f"    MAE:          {dist_metrics['mae']:.3f}")
        print(f"    RMSE:         {dist_metrics['rmse']:.3f}")
        print(f"    Bias:         {dist_metrics['bias']:.3f}")
        print(f"    Correlation:  {dist_metrics['correlation']:.3f}")
        print()
        print("  ANGLE:")
        print(f"    MAE:          {angle_metrics['mae']:.2f}")
        print(f"    RMSE:         {angle_metrics['rmse']:.2f}")
        print(f"    R²:           {angle_metrics['r2']:.3f}")
        print(f"    Correlation:  {angle_metrics['correlation']:.3f}")
        print()
        print("  Reference (random distance MAE ~ 1.25)")
        print("=" * 50)

    def plot_results(self, results):
        """Plot distance MAE vs angle and angle decoded vs actual.

        Parameters
        ----------
        results : dict
            Output from cross_validate() or cross_validate_mean().
        """
        actual_dist = np.array(results['all_actual_dist'])
        decoded_dist = np.array(results['all_decoded_dist'])
        actual_angle = np.array(results['all_actual_angle'])
        decoded_angle = np.array(results['all_decoded_angle'])
        angle_indices = np.array(results['all_angle_indices'])
        dist_metrics = results['dist_metrics']
        angle_metrics = results['angle_metrics']

        n_positions = self.n_angles if self.chunk_size == 1 else self.n_chunks
        x_vals = np.arange(n_positions)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Distance MAE vs angle
        ax = axes[0]
        dist_errors = decoded_dist - actual_dist
        mae_by_pos = np.zeros(n_positions)
        for idx in range(n_positions):
            mask = angle_indices == idx
            if mask.sum() > 0:
                mae_by_pos[idx] = np.mean(np.abs(dist_errors[mask]))
        ax.plot(x_vals, mae_by_pos, 'b-', linewidth=2)
        ax.axhline(dist_metrics['mae'], color='g', linestyle='--', alpha=0.7,
                   label=f'Overall MAE = {dist_metrics["mae"]:.2f}')
        ax.axhline(1.25, color='r', linestyle=':', alpha=0.5, label='Random ~1.25')
        ax.set_xlabel('Angle Position')
        ax.set_ylabel('Distance MAE')
        ax.set_title('Distance Decoding Error vs Angle (NN)')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Decoded vs actual angle (scatter)
        ax = axes[1]
        ax.scatter(actual_angle, decoded_angle, alpha=0.05, s=1, color='k')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1)
        ax.set_xlabel('Actual angle (normalized)')
        ax.set_ylabel('Decoded angle (normalized)')
        ax.set_title(f'Angle Decoding (R²={angle_metrics["r2"]:.3f})')
        ax.set_aspect('equal')

        # 3. Mean decoded angle trajectory
        ax = axes[2]
        mean_decoded_by_pos = np.zeros(n_positions)
        std_decoded_by_pos = np.zeros(n_positions)
        for idx in range(n_positions):
            mask = angle_indices == idx
            if mask.sum() > 0:
                mean_decoded_by_pos[idx] = np.mean(decoded_angle[mask])
                std_decoded_by_pos[idx] = np.std(decoded_angle[mask])
        ax.plot(self.norm_angles, self.norm_angles, 'r--', linewidth=1, label='Perfect')
        ax.plot(self.norm_angles, mean_decoded_by_pos, 'b-', linewidth=2, label='Mean decoded')
        ax.fill_between(self.norm_angles,
                        mean_decoded_by_pos - std_decoded_by_pos,
                        mean_decoded_by_pos + std_decoded_by_pos,
                        alpha=0.2, color='b')
        ax.set_xlabel('Actual angle (normalized)')
        ax.set_ylabel('Decoded angle (normalized)')
        ax.set_title('Mean Angle Trajectory (NN)')
        ax.legend()

        plt.tight_layout()
        plt.show()
        return fig

    def plot_results_comparison(self, results_list, labels):
        """Plot distance MAE vs angle for multiple conditions.

        Parameters
        ----------
        results_list : list of dict
        labels : list of str
        """
        colors = ['b', 'r', 'g', 'orange', 'purple']
        fig, ax = plt.subplots(figsize=(10, 6))

        n_positions = self.n_angles if self.chunk_size == 1 else self.n_chunks
        x_vals = np.arange(n_positions)

        for i, (results, label) in enumerate(zip(results_list, labels)):
            actual = np.array(results['all_actual_dist'])
            decoded = np.array(results['all_decoded_dist'])
            angle_indices = np.array(results['all_angle_indices'])
            metrics = results['dist_metrics']
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
        ax.set_xlabel('Angle Position')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Distance Decoding Error vs Angle (NN)')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def compare_results(results_list, labels):
        """
        Compare NN decoder results across different conditions.

        Parameters
        ----------
        results_list : list of dict
            List of results from cross_validate() or cross_validate_mean().
        labels : list of str
            Labels for each condition (e.g., ['V1', 'LM'] or ['Mono', 'Bino']).

        Returns
        -------
        dict with comparison metrics
        """
        n = len(results_list)
        metrics_list = []

        for results in results_list:
            actual = np.array(results['all_actual_dist'])
            decoded = np.array(results['all_decoded_dist'])

            mae = np.mean(np.abs(decoded - actual))
            rmse = np.sqrt(np.mean((decoded - actual) ** 2))
            corr = np.corrcoef(actual, decoded)[0, 1]

            # Classification accuracy
            predicted = np.clip(np.round(decoded), 0, 3).astype(int)
            accuracy = np.mean(predicted == actual)

            metrics_list.append({
                'mae': mae, 'rmse': rmse, 'corr': corr, 'accuracy': accuracy
            })

        # Print comparison table
        print("=" * (20 + 15 * n))
        print(f"{'NN DECODER COMPARISON':^{20 + 15 * n}}")
        print("=" * (20 + 15 * n))
        header = f"{'Metric':<20}" + "".join([f"{label:<15}" for label in labels])
        print(header)
        print("-" * (20 + 15 * n))

        for metric_name in ['mae', 'rmse', 'corr', 'accuracy']:
            row = f"{metric_name.upper():<20}"
            for m in metrics_list:
                if metric_name == 'accuracy':
                    row += f"{m[metric_name]:<15.1%}"
                else:
                    row += f"{m[metric_name]:<15.3f}"
            print(row)

        print("=" * (20 + 15 * n))

        return {'labels': labels, 'metrics': metrics_list}

    @staticmethod
    def plot_compare_results(results_list, labels, angle_values=None, figsize=(10, 5)):
        """
        Plot MAE vs angle comparison for multiple conditions.

        Parameters
        ----------
        results_list : list of dict
            List of results from cross_validate() or cross_validate_mean().
        labels : list of str
            Labels for each condition.
        angle_values : array-like, optional
            Actual angle values in degrees for x-axis. If None, uses indices.
        figsize : tuple

        Returns
        -------
        fig, comparison_dict
        """
        import matplotlib.pyplot as plt

        # First print the comparison table
        comparison = NNDecoder.compare_results(results_list, labels)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        modulation_colors = {'Binocular': 'blue', 'Monocular': 'coral'}
        colors = [modulation_colors.get(l, plt.cm.tab10(i / max(len(results_list), 1)))
                  for i, l in enumerate(labels)]

        for i, (results, label) in enumerate(zip(results_list, labels)):
            actual = np.array(results['all_actual_dist'])
            decoded = np.array(results['all_decoded_dist'])
            angle_indices = np.array(results['all_angle_indices'])

            n_positions = int(angle_indices.max()) + 1
            errors = decoded - actual

            mae_by_pos = np.zeros(n_positions)
            for idx in range(n_positions):
                mask = angle_indices == idx
                if mask.sum() > 0:
                    mae_by_pos[idx] = np.mean(np.abs(errors[mask]))

            # Use actual angle values if provided, otherwise use indices
            if angle_values is not None:
                x_vals = np.array(angle_values)[:n_positions]
            else:
                x_vals = np.arange(n_positions)

            metrics = comparison['metrics'][i]
            ax.plot(x_vals, mae_by_pos, '-', color=colors[i],
                    linewidth=2, label=f'{label} (MAE={metrics["mae"]:.2f})')

        ax.axhline(1.25, color='k', linestyle=':', alpha=0.5, label='Chance ~1.25')

        if angle_values is not None:
            ax.set_xlabel('Angle (degrees)')
        else:
            ax.set_xlabel('Angle Position')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Distance Decoding Error vs Angle')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, comparison

    @staticmethod
    def plot_compare_results_multi(binocular_results, monocular_results,
                                   areas=('V1', 'RL', 'LM'),
                                   angle_values=None, figsize=(18, 5)):
        """
        Plot binocular vs monocular MAE comparison across areas in one figure.

        Parameters
        ----------
        binocular_results : dict
            Dict mapping area name -> results dict from cross_validate().
        monocular_results : dict
            Dict mapping area name -> results dict from cross_validate().
        areas : list of str
            Areas to plot (one subplot column each).
        angle_values : array-like, optional
            Actual angle values in degrees for x-axis.
        figsize : tuple

        Returns
        -------
        fig, comparisons : dict mapping area -> comparison dict
        """
        import matplotlib.pyplot as plt

        n_areas = len(areas)
        fig, axes = plt.subplots(1, n_areas, figsize=figsize, sharey=True)
        if n_areas == 1:
            axes = [axes]

        labels = ['Binocular', 'Monocular']
        colors = ['blue', 'coral']
        comparisons = {}

        for col, area in enumerate(areas):
            ax = axes[col]
            results_list = [binocular_results[area], monocular_results[area]]
            comparison = NNDecoder.compare_results(results_list, labels)
            comparisons[area] = comparison

            for i, (results, label) in enumerate(zip(results_list, labels)):
                actual = np.array(results['all_actual_dist'])
                decoded = np.array(results['all_decoded_dist'])
                angle_indices = np.array(results['all_angle_indices'])

                n_positions = int(angle_indices.max()) + 1
                errors = decoded - actual

                mae_by_pos = np.zeros(n_positions)
                for idx in range(n_positions):
                    mask = angle_indices == idx
                    if mask.sum() > 0:
                        mae_by_pos[idx] = np.mean(np.abs(errors[mask]))

                if angle_values is not None:
                    x_vals = np.array(angle_values)[:n_positions]
                else:
                    x_vals = np.arange(n_positions)

                metrics = comparison['metrics'][i]
                ax.plot(x_vals, mae_by_pos, '-', color=colors[i],
                        linewidth=4, label=f'{label} (MAE={metrics["mae"]:.2f})')

            ax.axhline(1.25, color='k', linestyle=':', alpha=0.5, label='Chance ~1.25')
            ax.axvspan(-20, 20, color='purple', alpha=0.3)
            ax.set_title(area, fontsize=18)
            ax.set_xlabel('Angle (degrees)' if angle_values is not None else 'Angle Position')
            if col == 0:
                ax.set_ylabel('Mean Absolute Error')
            ax.set_ylim([0, 2])
            ax.legend(fontsize=14)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, comparisons


class _JointClassifierMLP(nn.Module):
    """MLP that classifies both distance and angle from neural response alone."""

    def __init__(self, input_size, hidden_sizes=(128, 64), head_size=32,
                 n_distances=4, n_angles=61, dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        self.dist_head = nn.Sequential(
            nn.Linear(in_dim, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, n_distances),
        )

        self.angle_head = nn.Sequential(
            nn.Linear(in_dim, head_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_size, n_angles),
        )

    def forward(self, x):
        features = self.backbone(x)
        dist_logits = self.dist_head(features)
        angle_logits = self.angle_head(features)
        return dist_logits, angle_logits


class JointDecoder:
    """Joint angle+distance decoder that outputs full probability maps.

    Given a single neural response vector (one frame/time-point), outputs
    P(distance) and P(angle) as proper probability distributions, and their
    outer product as the joint P(angle, distance | response).

    Compatible with SessionAggregator.get_decoder_data() output.
    """

    def __init__(self, hidden_sizes=(128, 64), head_size=32, dropout=0.3,
                 lr=1e-3, epochs=200, patience=15, angle_weight=1.0):
        self.n_distances = 4
        self.hidden_sizes = hidden_sizes
        self.head_size = head_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.angle_weight = angle_weight

        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None
        self.n_neurons = None
        self.n_angles = None

    def _build_model(self):
        return _JointClassifierMLP(
            input_size=self.n_neurons,
            hidden_sizes=self.hidden_sizes,
            head_size=self.head_size,
            n_distances=self.n_distances,
            n_angles=self.n_angles,
            dropout=self.dropout,
        ).to(self.torch_device)

    def _build_training_data(self, X, y_distance):
        """Build flat training arrays from 3D data.

        Parameters
        ----------
        X : np.ndarray, shape (n_neurons, n_angles, n_trials)
        y_distance : np.ndarray, shape (n_trials,)

        Returns
        -------
        X_flat : array, shape (n_samples, n_neurons)
        y_dist : array, shape (n_samples,) int
        y_angle : array, shape (n_samples,) int
        """
        n_neurons, n_angles, n_trials = X.shape
        X_flat = X.transpose(2, 1, 0).reshape(-1, n_neurons).astype(np.float32)
        y_dist = np.repeat(y_distance, n_angles).astype(np.int64)
        y_angle = np.tile(np.arange(n_angles), n_trials).astype(np.int64)
        return X_flat, y_dist, y_angle

    def fit(self, X_train, y_train):
        """Train the joint classifier.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_neurons, n_angles, n_trials)
        y_train : np.ndarray, shape (n_trials,)

        Returns
        -------
        self
        """
        self.n_neurons = X_train.shape[0]
        self.n_angles = X_train.shape[1]

        X, y_dist, y_angle = self._build_training_data(X_train, y_train)

        # 80/20 split for early stopping
        n = len(X)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train_t = torch.tensor(X[train_idx], device=self.torch_device)
        y_dist_train = torch.tensor(y_dist[train_idx], device=self.torch_device)
        y_angle_train = torch.tensor(y_angle[train_idx], device=self.torch_device)

        X_val = torch.tensor(X[val_idx], device=self.torch_device)
        y_dist_val = torch.tensor(y_dist[val_idx], device=self.torch_device)
        y_angle_val = torch.tensor(y_angle[val_idx], device=self.torch_device)

        train_ds = TensorDataset(X_train_t, y_dist_train, y_angle_train)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        dist_criterion = nn.CrossEntropyLoss()
        angle_criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for X_batch, y_d_batch, y_a_batch in train_loader:
                dist_logits, angle_logits = self.model(X_batch)
                loss = dist_criterion(dist_logits, y_d_batch) + \
                       self.angle_weight * angle_criterion(angle_logits, y_a_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                dist_logits, angle_logits = self.model(X_val)
                val_loss = dist_criterion(dist_logits, y_dist_val) + \
                           self.angle_weight * angle_criterion(angle_logits, y_angle_val)
                val_loss = val_loss.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.torch_device) for k, v in best_state.items()})
        self.model.eval()

        return self

    def predict_joint(self, response_vector):
        """Predict joint P(angle, distance) for a single neural response.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)

        Returns
        -------
        joint : np.ndarray, shape (n_angles, n_distances)
            Joint probability matrix.
        p_distance : np.ndarray, shape (n_distances,)
            Marginal distance probabilities.
        p_angle : np.ndarray, shape (n_angles,)
            Marginal angle probabilities.
        """
        feat = np.asarray(response_vector, dtype=np.float32)
        x = torch.tensor(feat, device=self.torch_device).unsqueeze(0)

        with torch.no_grad():
            dist_logits, angle_logits = self.model(x)

        p_distance = torch.softmax(dist_logits, dim=-1).cpu().numpy()[0]
        p_angle = torch.softmax(angle_logits, dim=-1).cpu().numpy()[0]

        joint = np.outer(p_distance, p_angle)  # (n_distances, n_angles)
        return joint, p_distance, p_angle

    def plot_joint(self, response_vector, angle_values_deg=None,
                   actual_distance=None, actual_angle_idx=None):
        """Plot the joint probability map as a radial polar plot.

        Angles are distributed radially, distance is in the r direction,
        matching the single-cell radial plot convention.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
        angle_values_deg : array-like, optional
            Angle values in degrees. If None, uses indices as degrees.
        actual_distance : int, optional
            True distance (0-3) to mark.
        actual_angle_idx : int, optional
            True angle index to mark.

        Returns
        -------
        fig, joint, p_distance, p_angle
        """
        joint, p_distance, p_angle = self.predict_joint(response_vector)

        if angle_values_deg is not None:
            angles_deg = np.array(angle_values_deg)
        else:
            angles_deg = np.linspace(0, 90, self.n_angles)

        # Build polar grid edges
        theta_edges = np.linspace(np.radians(angles_deg[0]),
                                  np.radians(angles_deg[-1]),
                                  self.n_angles + 1)
        r_edges = np.linspace(0, self.n_distances, self.n_distances + 1)
        theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)

        fig = plt.figure(figsize=(14, 5))

        # Left: radial polar heatmap
        ax_polar = fig.add_subplot(1, 3, 1, projection='polar')
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_thetamin(angles_deg[0])
        ax_polar.set_thetamax(angles_deg[-1])
        pcm = ax_polar.pcolormesh(theta_grid, r_grid, joint,
                                  shading='auto', cmap='hot')
        ax_polar.set_title('P(angle, distance | response)', pad=15)
        plt.colorbar(pcm, ax=ax_polar, label='Probability', shrink=0.8)

        # Mark true position
        if actual_distance is not None and actual_angle_idx is not None:
            true_theta = np.radians(angles_deg[actual_angle_idx])
            true_r = actual_distance + 0.5
            ax_polar.plot(true_theta, true_r, 'c*', markersize=15,
                          markeredgecolor='white', markeredgewidth=1)

        # Center: marginal P(angle)
        ax = fig.add_subplot(1, 3, 2)
        ax.bar(angles_deg, p_angle, width=np.diff(angles_deg).mean(), color='coral')
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Probability')
        ax.set_title('P(angle)')
        ax.set_ylim([0, max(0.3, p_angle.max() * 1.2)])
        if actual_angle_idx is not None:
            ax.axvline(angles_deg[actual_angle_idx], color='green', linestyle='--',
                       label=f'True={angles_deg[actual_angle_idx]:.0f}°')
            ax.legend(fontsize=8)

        # Right: marginal P(distance)
        ax = fig.add_subplot(1, 3, 3)
        ax.barh(range(self.n_distances), p_distance, color='steelblue')
        ax.set_ylabel('Distance class')
        ax.set_xlabel('Probability')
        ax.set_title('P(distance)')
        ax.set_yticks(range(self.n_distances))
        ax.set_xlim([0, 1])
        if actual_distance is not None:
            ax.axhline(actual_distance, color='red', linestyle='--',
                       label=f'True dist={actual_distance}')
            ax.legend()

        plt.tight_layout()
        plt.show()
        return fig, joint, p_distance, p_angle

    def animate_trial(self, trial_data, angle_values_deg=None,
                      actual_distance=None, save_path=None, fps=5):
        """Create a video animating predictions across all time-points of a trial.

        Parameters
        ----------
        trial_data : np.ndarray, shape (n_neurons, n_angles)
            Neural responses for one trial across all angle positions.
        angle_values_deg : array-like, optional
            Angle values in degrees.
        actual_distance : int, optional
            True distance (0-3) to mark.
        save_path : str, optional
            Path to save the video (e.g., 'trial_prediction.mp4').
            If None, saves to 'joint_decoder_trial.mp4'.
        fps : int
            Frames per second for the video.

        Returns
        -------
        str : path to the saved video file
        """
        from matplotlib.animation import FuncAnimation, PillowWriter

        if angle_values_deg is not None:
            angles_deg = np.array(angle_values_deg)
        else:
            angles_deg = np.linspace(0, 90, self.n_angles)

        if save_path is None:
            save_path = 'joint_decoder_trial.gif'

        # Precompute all predictions
        all_joints, all_p_dist, all_p_angle = [], [], []
        for t in range(trial_data.shape[1]):
            joint, p_dist, p_ang = self.predict_joint(trial_data[:, t])
            all_joints.append(joint)
            all_p_dist.append(p_dist)
            all_p_angle.append(p_ang)

        # Find global vmax for consistent colormap
        vmax = max(j.max() for j in all_joints)

        # Build polar grid
        theta_edges = np.linspace(np.radians(angles_deg[0]),
                                  np.radians(angles_deg[-1]),
                                  self.n_angles + 1)
        r_edges = np.linspace(0, self.n_distances, self.n_distances + 1)
        theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)

        # Create figure
        fig = plt.figure(figsize=(14, 5))

        # Polar axis
        ax_polar = fig.add_subplot(1, 3, 1, projection='polar')
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_thetamin(angles_deg[0])
        ax_polar.set_thetamax(angles_deg[-1])

        # Angle axis
        ax_angle = fig.add_subplot(1, 3, 2)

        # Distance axis
        ax_dist = fig.add_subplot(1, 3, 3)

        # Initial plot
        pcm = ax_polar.pcolormesh(theta_grid, r_grid, all_joints[0],
                                  shading='auto', cmap='hot', vmin=0, vmax=vmax)
        plt.colorbar(pcm, ax=ax_polar, label='Probability', shrink=0.8)

        # True position marker
        star_marker = None
        if actual_distance is not None:
            true_r = actual_distance + 0.5

        bar_width = np.diff(angles_deg).mean()
        suptitle = fig.suptitle('', fontsize=12)

        def update(frame_idx):
            nonlocal star_marker

            joint = all_joints[frame_idx]
            p_angle = all_p_angle[frame_idx]
            p_dist = all_p_dist[frame_idx]

            # Update polar heatmap
            pcm.set_array(joint.ravel())

            # Remove old star
            if star_marker is not None:
                star_marker.remove()
                star_marker = None

            # Draw new star at current angle
            if actual_distance is not None:
                true_theta = np.radians(angles_deg[frame_idx])
                star_marker, = ax_polar.plot(true_theta, true_r, 'c*', markersize=15,
                                             markeredgecolor='white', markeredgewidth=1)

            # Update angle bar chart
            ax_angle.clear()
            ax_angle.bar(angles_deg, p_angle, width=bar_width, color='coral')
            ax_angle.axvline(angles_deg[frame_idx], color='green', linestyle='--',
                             label=f'Current={angles_deg[frame_idx]:.0f}°')
            ax_angle.set_xlabel('Angle (deg)')
            ax_angle.set_ylabel('Probability')
            ax_angle.set_title('P(angle)')
            ax_angle.set_ylim([0, max(0.3, max(a.max() for a in all_p_angle) * 1.2)])
            ax_angle.legend(fontsize=8)

            # Update distance bar chart
            ax_dist.clear()
            ax_dist.barh(range(self.n_distances), p_dist, color='steelblue')
            ax_dist.set_ylabel('Distance class')
            ax_dist.set_xlabel('Probability')
            ax_dist.set_title('P(distance)')
            ax_dist.set_yticks(range(self.n_distances))
            ax_dist.set_xlim([0, 1])
            if actual_distance is not None:
                ax_dist.axhline(actual_distance, color='red', linestyle='--',
                                label=f'True dist={actual_distance}')
                ax_dist.legend()

            suptitle.set_text(f'Frame {frame_idx + 1}/{self.n_angles}  |  '
                              f'Angle = {angles_deg[frame_idx]:.0f}°')

            return [pcm, suptitle]

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        anim = FuncAnimation(fig, update, frames=self.n_angles,
                             interval=1000 // fps, blit=False)

        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        print(f'Video saved to {save_path}')
        return save_path
