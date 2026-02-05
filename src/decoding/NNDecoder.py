import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


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

    Uses an angle-aware MLP: input is [neural_response, normalized_angle].
    Distance head outputs 4-class probabilities (posterior mean for continuous output).
    Angle head outputs angle regression.
    """

    def __init__(self, device, cells, monocular=None, chunk_size=1,
                 hidden_sizes=(128, 64), head_size=32, dropout=0.3,
                 lr=1e-3, epochs=200, patience=15, angle_weight=1.0):
        """
        Parameters
        ----------
        device : RadialDevice
        cells : array-like
            Cell IDs to use for decoding.
        monocular : int or None
            Filter trials by monocular condition.
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

        # NN hyperparameters
        self.hidden_sizes = hidden_sizes
        self.head_size = head_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.angle_weight = angle_weight

        # Normalize angle range to [0, 1]
        self.angle_min = self.angle_range[0]
        self.angle_max = self.angle_range[-1]
        self.norm_angles = (self.angle_range - self.angle_min) / (self.angle_max - self.angle_min)

        # Torch device
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = None

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
        return np.array([np.mean(self.angle_range[idx]) for idx in self._get_chunk_indices()])

    def get_trial_responses(self, trial_idx):
        return np.array([self.device.get_repeat_trace(cell, trial_idx) for cell in self.cells])

    def _build_training_data(self, train_trials_by_distance):
        """Build X, y_dist, y_angle arrays from training trials.

        Returns
        -------
        X : array, shape (n_samples, n_neurons + 1)
        y_dist : array, shape (n_samples,) int
        y_angle : array, shape (n_samples,) float (normalized 0-1)
        """
        X_list, y_dist_list, y_angle_list = [], [], []

        for dist in range(self.n_distances):
            for t in train_trials_by_distance[dist]:
                responses = self.get_trial_responses(t)  # (n_neurons, n_angles)

                if self.chunk_size == 1:
                    for a_idx in range(self.n_angles):
                        feat = np.append(responses[:, a_idx], self.norm_angles[a_idx])
                        X_list.append(feat)
                        y_dist_list.append(dist)
                        y_angle_list.append(self.norm_angles[a_idx])
                else:
                    for chunk_idxs in self._get_chunk_indices():
                        avg_response = np.mean(responses[:, chunk_idxs], axis=1)
                        avg_norm_angle = np.mean(self.norm_angles[chunk_idxs])
                        feat = np.append(avg_response, avg_norm_angle)
                        X_list.append(feat)
                        y_dist_list.append(dist)
                        y_angle_list.append(avg_norm_angle)

        return (np.array(X_list, dtype=np.float32),
                np.array(y_dist_list, dtype=np.int64),
                np.array(y_angle_list, dtype=np.float32))

    def fit(self, train_trials_by_distance):
        """Train the multi-head MLP on training trials.

        Parameters
        ----------
        train_trials_by_distance : dict
            Maps distance (0-3) to array of trial indices for training.
        """
        X, y_dist, y_angle = self._build_training_data(train_trials_by_distance)

        # Stratified 80/20 split for early stopping
        n = len(X)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(0.8 * n)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train = torch.tensor(X[train_idx], device=self.torch_device)
        y_dist_train = torch.tensor(y_dist[train_idx], device=self.torch_device)
        y_angle_train = torch.tensor(y_angle[train_idx], device=self.torch_device)

        X_val = torch.tensor(X[val_idx], device=self.torch_device)
        y_dist_val = torch.tensor(y_dist[val_idx], device=self.torch_device)
        y_angle_val = torch.tensor(y_angle[val_idx], device=self.torch_device)

        train_ds = TensorDataset(X_train, y_dist_train, y_angle_train)
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

    def decode(self, response_vector, angle_idx):
        """Decode distance and angle from a single response vector.

        Parameters
        ----------
        response_vector : array, shape (n_neurons,)
        angle_idx : int

        Returns
        -------
        decoded_distance : float (0-3 continuous via posterior mean)
        decoded_angle : float (degrees)
        """
        feat = np.append(response_vector, self.norm_angles[angle_idx]).astype(np.float32)
        x = torch.tensor(feat, device=self.torch_device).unsqueeze(0)

        with torch.no_grad():
            dist_logits, angle_pred = self.model(x)

        # Distance: softmax → posterior mean
        probs = torch.softmax(dist_logits, dim=-1).cpu().numpy()[0]
        decoded_dist = float(np.sum(self.distance_values * probs))

        # Angle: denormalize
        decoded_angle = float(angle_pred.cpu().numpy()[0]) * (self.angle_max - self.angle_min) + self.angle_min

        return decoded_dist, decoded_angle

    def decode_response_matrix(self, responses):
        """Batch decode distance and angle from a response matrix.

        Parameters
        ----------
        responses : array, shape (n_neurons, n_angles)

        Returns
        -------
        decoded_distances : list of float
        decoded_angles : list of float
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
            for chunk_idxs in chunks:
                avg_response = np.mean(responses[:, chunk_idxs], axis=1)
                avg_norm_angle = np.mean(self.norm_angles[chunk_idxs])
                feat_list.append(np.append(avg_response, avg_norm_angle))
            feats = np.array(feat_list, dtype=np.float32)

        x = torch.tensor(feats, device=self.torch_device)
        with torch.no_grad():
            dist_logits, angle_preds = self.model(x)

        probs = torch.softmax(dist_logits, dim=-1).cpu().numpy()
        decoded_distances = [float(np.sum(self.distance_values * p)) for p in probs]

        angle_vals = angle_preds.cpu().numpy()
        decoded_angles = [float(a) * (self.angle_max - self.angle_min) + self.angle_min for a in angle_vals]

        return decoded_distances, decoded_angles

    def decode_trial(self, trial_idx):
        """Decode distance and angle at each position for a single trial.

        Returns
        -------
        decoded_distances : list of float
        decoded_angles : list of float
        """
        responses = self.get_trial_responses(trial_idx)
        return self.decode_response_matrix(responses)

    def _build_kfold_splits(self, k_folds):
        """Build stratified k-fold splits by distance."""
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
        """Run stratified k-fold cross-validation for distance and angle decoding.

        Parameters
        ----------
        k_folds : int

        Returns
        -------
        dict with keys:
            all_decoded_dist, all_actual_dist, all_angle_indices, dist_metrics,
            all_decoded_angle, all_actual_angle, angle_metrics
        """
        splits = self._build_kfold_splits(k_folds)
        all_decoded_dist = []
        all_actual_dist = []
        all_decoded_angle = []
        all_actual_angle = []
        all_angle_indices = []
        n_output = self.n_angles if self.chunk_size == 1 else self.n_chunks

        if self.chunk_size == 1:
            actual_angle_values = self.angle_range
        else:
            actual_angle_values = self._get_chunk_centers()

        for fold, (train_by_dist, test_trials) in enumerate(splits):
            print(f'=== Fold {fold + 1}/{k_folds} ===', end='\r', flush=True)

            self.fit(train_by_dist)

            for test_idx, actual_dist in test_trials:
                dec_dist, dec_angle = self.decode_trial(test_idx)
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

    def cross_validate_mean(self, n_repeats=100):
        """Run repeated 50/50 split CV, decoding from mean test responses per distance.

        Parameters
        ----------
        n_repeats : int

        Returns
        -------
        dict (same format as cross_validate)
        """
        all_decoded_dist = []
        all_actual_dist = []
        all_decoded_angle = []
        all_actual_angle = []
        all_angle_indices = []
        n_output = self.n_angles if self.chunk_size == 1 else self.n_chunks

        if self.chunk_size == 1:
            actual_angle_values = self.angle_range
        else:
            actual_angle_values = self._get_chunk_centers()

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
                test_responses = np.array([
                    self.get_trial_responses(t) for t in test_by_dist[dist]
                ])
                mean_responses = np.mean(test_responses, axis=0)

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

        if self.chunk_size == 1:
            x_vals = self.angle_range
            n_positions = self.n_angles
        else:
            x_vals = self._get_chunk_centers()
            n_positions = self.n_chunks

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
        ax.set_xlabel('Angle Position (deg)')
        ax.set_ylabel('Distance MAE')
        ax.set_title('Distance Decoding Error vs Angle (NN)')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Decoded vs actual angle (scatter)
        ax = axes[1]
        ax.scatter(actual_angle, decoded_angle, alpha=0.05, s=1, color='k')
        ax.plot([self.angle_range[0], self.angle_range[-1]],
                [self.angle_range[0], self.angle_range[-1]], 'r--', linewidth=1)
        ax.set_xlabel('Actual angle (deg)')
        ax.set_ylabel('Decoded angle (deg)')
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
        ax.plot(x_vals, x_vals, 'r--', linewidth=1, label='Perfect')
        ax.plot(x_vals, mean_decoded_by_pos, 'b-', linewidth=2, label='Mean decoded')
        ax.fill_between(x_vals,
                        mean_decoded_by_pos - std_decoded_by_pos,
                        mean_decoded_by_pos + std_decoded_by_pos,
                        alpha=0.2, color='b')
        ax.set_xlabel('Actual angle (deg)')
        ax.set_ylabel('Decoded angle (deg)')
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

        if self.chunk_size == 1:
            x_vals = self.angle_range
            n_positions = self.n_angles
        else:
            x_vals = self._get_chunk_centers()
            n_positions = self.n_chunks

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
        ax.set_xlabel('Angle Position (deg)')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Distance Decoding Error vs Angle (NN)')
        ax.set_ylim([0, 2])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
