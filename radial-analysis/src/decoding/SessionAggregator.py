"""
SessionAggregator: Aggregates neural traces from all sessions for an experiment.

Enables decoder training on combined data from multiple sessions with filtering
by distance, monocular condition, and visual area.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Iterator, Optional

from sklearn.model_selection import train_test_split, StratifiedKFold

from src.data.RadialDevice import RadialDevice
from src.workflows import load_dataframe

class SessionAggregator:
    """
    Aggregates neural traces from all sessions for a given experiment.

    Loads traces from multiple sessions, filters by reliability, and provides
    methods to extract data for decoder training with various cross-validation
    strategies.

    Parameters
    ----------
    mouse_name : str
        Mouse identifier (e.g., 'SAM040')
    experiment_name : str
        Experiment name (e.g., 'tiling_120um')
    base_data_dir : str
        Base directory for processed single-session data
    multi_session_dir : str
        Base directory for multi-session combined data
    reliability_threshold : float
        Threshold for filtering reliable cells (default: 0.85)
    reliable_only : bool
        Whether to filter to reliable cells only (default: True)
    force_recompute : bool
        If True, recompute traces even if cache exists (default: False)

    Attributes
    ----------
    traces : Dict[int, np.ndarray]
        Dictionary mapping DataFrame index to (n_trials, n_positions) array
    trial_info : Dict[str, Dict]
        Dictionary mapping session_name to trial metadata
    df : pd.DataFrame
        Multi-session DataFrame with cell metadata

    Examples
    --------
    >>> agg = SessionAggregator('SAM040', 'tiling_120um')
    >>> X, y_dist, y_angle = agg.get_decoder_data(monocular=0, area='V1')
    >>> train_idx, test_idx = agg.get_train_test_split()
    """

    def __init__(
        self,
        mouse_name: str,
        experiment_name: str,
        base_data_dir: str = r'D:\data-distance-coding-project\processed-data',
        multi_session_dir: str = r'D:\data-distance-coding-project\multi-session-data',
        reliability_threshold: float = 0.85,
        reliable_only: bool = True,
        force_recompute: bool = False
    ):
        self.mouse_name = mouse_name
        self.experiment_name = experiment_name
        self.base_data_dir = base_data_dir
        self.multi_session_dir = multi_session_dir
        self.reliability_threshold = reliability_threshold
        self.reliable_only = reliable_only

        self._traces: Dict[int, np.ndarray] = {}
        self._trial_info: Dict[str, Dict] = {}
        self._df: Optional[pd.DataFrame] = None
        self._target_length: Optional[int] = None
        self._start_angle: Optional[float] = None
        self._end_angle: Optional[float] = None
        self._distances: Optional[np.ndarray] = None

        # Try to load from cache first
        if not force_recompute and self._load_cache():
            print(f"[ok] Loaded aggregated traces from cache")
        else:
            self._build_traces()
            self._save_cache()

    @property
    def traces(self) -> Dict[int, np.ndarray]:
        """Dictionary mapping DataFrame index to (n_trials, n_positions) array."""
        return self._traces

    @property
    def trial_info(self) -> Dict[str, Dict]:
        """Dictionary mapping session_name to trial metadata."""
        return self._trial_info

    @property
    def df(self) -> pd.DataFrame:
        """Multi-session DataFrame with cell metadata."""
        return self._df

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return len(self._traces)

    @property
    def n_positions(self) -> int:
        """Number of angular positions (normalized across sessions)."""
        return self._target_length

    @property
    def available_areas(self) -> List[str]:
        """List of available visual areas."""
        if 'area' in self._df.columns:
            return [a for a in self._df['area'].unique() if a is not None]
        return []

    @property
    def angle_axis(self) -> np.ndarray:
        """Angle in degrees for each position, shape (n_positions,)."""
        return np.linspace(self._start_angle, self._end_angle, self._target_length)

    @property
    def distances(self) -> np.ndarray:
        """Unique stimulus distances (excluding control), sorted ascending."""
        return self._distances

    @property
    def cache_path(self) -> str:
        """Default cache path."""
        return os.path.join(
            self.multi_session_dir,
            f'{self.mouse_name}_{self.experiment_name}',
            f'{self.mouse_name}_{self.experiment_name}_aggregated.pkl'
        )

    def resolve_session(self, session):
        """Resolve session argument to a full session name string.

        Parameters
        ----------
        session : str, int, or None
            Full session name, or an integer N that matches the session whose
            name contains the token '_N_'
            (e.g. 9 → '251217_SAM043_tiling_9_120um').

        Returns
        -------
        str or None
        """
        if session is None or isinstance(session, str):
            return session
        token = f'_{int(session)}_'
        matches = [s for s in self.trial_info if token in s]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise ValueError(
                f"No session containing '{token}' found. "
                f"Available: {list(self.trial_info.keys())}"
            )
        raise ValueError(f"Ambiguous session number {session}: matched {matches}")

    def _build_traces(self):
        """Load and aggregate traces from all sessions."""
        print(f"[...] Building traces for {self.mouse_name}_{self.experiment_name}")

        # Load multi-session DataFrame
        multi_df = load_dataframe(
            mouse_name=self.mouse_name,
            experiment_name=self.experiment_name,
            base_data_dir=self.base_data_dir,
            base_multi_session_dir=self.multi_session_dir
        )

        self._df = multi_df

        # First pass: determine minimum trace length across all sessions
        min_length = float('inf')
        for session_name in self._df['session_name'].unique():
            parts = session_name.split('_')
            record_date = parts[0]
            mouse = parts[1]
            exp = '_'.join(parts[2:])

            device = RadialDevice(
                mouse_name=mouse,
                record_date=record_date,
                experiment_name=exp,
                base_data_dir=self.base_data_dir
            )

            # Check trace length from first cell, first trial
            sample_trace = device.get_on_trace(0, 0)
            min_length = min(min_length, len(sample_trace))

            # Capture angle range (common across sessions); force scalar
            self._start_angle = float(np.asarray(device.start_angle).flat[0])
            self._end_angle   = float(np.asarray(device.end_angle).flat[0])

        self._target_length = int(min_length)
        print(f"Normalizing traces to {self._target_length} positions")

        # Second pass: extract traces
        for session_name, session_group in self._df.groupby('session_name'):
            print(f"Loading traces for {session_name} ({len(session_group)} cells)...")

            parts = session_name.split('_')
            record_date = parts[0]
            mouse = parts[1]
            exp = '_'.join(parts[2:])

            device = RadialDevice(
                mouse_name=mouse,
                record_date=record_date,
                experiment_name=exp,
                base_data_dir=self.base_data_dir
            )

            # Store trial info for this session
            n_trials = len(device.distance_sequence)
            all_repeats = np.arange(n_trials)

            self._trial_info[session_name] = {
                'repeat_indices': all_repeats,
                'distances': device.distance_sequence.copy(),
                'monocular': device.monocular_sequence.copy()
            }

            # Get traces for each cell
            for idx, row in session_group.iterrows():
                cell_id = row['cell_id']
                cell_idx = np.where(device.suite2p_id == cell_id)[0][0]

                all_traces = []
                for repeat in all_repeats:
                    on_trace = device.get_on_trace(cell_idx, repeat)
                    # Truncate to target length (on_trace is 1D)
                    truncated = on_trace[:self._target_length]
                    all_traces.append(truncated)

                self._traces[idx] = np.array(all_traces)

        print(f"[ok] Built traces for {len(self._traces)} cells")

        # Derive unique stimulus distances (exclude control condition = 4)
        all_dists = np.concatenate([info['distances'] for info in self._trial_info.values()])
        self._distances = np.unique(all_dists[all_dists != 4])

    def _save_cache(self):
        """Save traces to cache."""
        cache_data = {
            'traces': self._traces,
            'trial_info': self._trial_info,
            'df': self._df,
            'target_length': self._target_length,
            'start_angle': self._start_angle,
            'end_angle': self._end_angle,
            'distances': self._distances,
            'metadata': {
                'mouse_name': self.mouse_name,
                'experiment_name': self.experiment_name,
            }
        }

        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"[ok] Saved traces to {self.cache_path}")

    def _load_cache(self) -> bool:
        """Load traces from cache if it exists. Returns True if successful."""
        if not os.path.exists(self.cache_path):
            return False

        try:
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Invalidate old caches that lack angle/distance info
            if 'start_angle' not in cache_data:
                print("Cache missing angle/distance info, recomputing...")
                return False

            self._traces = cache_data['traces']
            self._trial_info = cache_data['trial_info']
            self._df = cache_data['df']
            self._target_length = cache_data['target_length']
            self._start_angle = cache_data['start_angle']
            self._end_angle = cache_data['end_angle']
            self._distances = cache_data['distances']
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def save_cache(self, path: str = None):
        """
        Save traces to a custom path.

        Parameters
        ----------
        path : str, optional
            Custom path to save to. If None, uses default cache path.
        """
        if path is None:
            self._save_cache()
        else:
            cache_data = {
                'traces': self._traces,
                'trial_info': self._trial_info,
                'df': self._df,
                'target_length': self._target_length,
                'start_angle': self._start_angle,
                'end_angle': self._end_angle,
                'distances': self._distances,
                'metadata': {
                    'mouse_name': self.mouse_name,
                    'experiment_name': self.experiment_name,
                }
            }
            with open(path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"[ok] Saved traces to {path}")

    def load_cache(self, path: str = None) -> bool:
        """
        Load traces from a custom path.

        Parameters
        ----------
        path : str, optional
            Custom path to load from. If None, uses default cache path.

        Returns
        -------
        bool
            True if loading was successful.
        """
        load_path = path if path is not None else self.cache_path

        if not os.path.exists(load_path):
            return False

        try:
            with open(load_path, 'rb') as f:
                cache_data = pickle.load(f)

            self._traces = cache_data['traces']
            self._trial_info = cache_data['trial_info']
            self._df = cache_data['df']
            self._target_length = cache_data['target_length']
            self._start_angle = cache_data.get('start_angle')
            self._end_angle = cache_data.get('end_angle')
            self._distances = cache_data.get('distances')
            return True
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False

    def _get_cell_indices(self, area: str = None, session: str = None) -> List[int]:
        """Get DataFrame indices for cells matching area, session, and reliability filters."""
        df = self._df

        if self.reliable_only:
            df = df[
                ((df['monocular'].apply(lambda x: x['reliable']) == True) |
                (df['binocular'].apply(lambda x: x['reliable']) == True)) &
                ((df['monocular'].apply(lambda x: x['reliable_corr']) >= self.reliability_threshold) |
                 (df['binocular'].apply(lambda x: x['reliable_corr']) >= self.reliability_threshold))
            ]

        if area is not None:
            if 'area' not in df.columns:
                raise ValueError("Area filtering requires 'area' column in DataFrame")
            df = df[df['area'] == area]

        if session is not None:
            session = self.resolve_session(session)
            df = df[df['session_name'] == session]

        return list(df.index)

    def _get_trial_mask(
        self,
        session_name: str,
        distance: int = None,
        monocular: int = None,
        include_control: bool = False
    ) -> np.ndarray:
        """Get boolean mask for trials matching conditions."""
        info = self._trial_info[session_name]
        distances = info['distances']
        mono = info['monocular']

        # Start with all trials
        mask = np.ones(len(distances), dtype=bool)

        # Filter by distance
        if distance is not None:
            mask &= (distances == distance)
        elif not include_control:
            # Exclude control (distance=4) by default
            mask &= (distances != 4)

        # Filter by monocular
        if monocular is not None:
            mask &= (mono == monocular)

        return mask

    def get_traces_for_condition(
        self,
        distance: int = None,
        monocular: int = None,
        area: str = None,
        include_control: bool = False
    ) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """
        Get traces filtered by conditions.

        Parameters
        ----------
        distance : int, optional
            Distance value (0-3, or 4 for control). None for all distances.
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.
        include_control : bool
            Whether to include control trials (distance=4). Default False.

        Returns
        -------
        filtered_traces : Dict[int, np.ndarray]
            Dictionary mapping cell index to filtered (n_filtered_trials, n_positions) array
        cell_indices : List[int]
            List of cell indices included
        """
        cell_indices = self._get_cell_indices(area)
        filtered_traces = {}

        for idx in cell_indices:
            session_name = self._df.loc[idx, 'session_name']
            mask = self._get_trial_mask(session_name, distance, monocular, include_control)

            if mask.sum() > 0:
                filtered_traces[idx] = self._traces[idx][mask]

        return filtered_traces, cell_indices

    def get_cell_traces(
        self,
        cell_id: int,
        session_name: str = None,
        distance: int = None,
        monocular: int = None,
        include_control: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get traces for a specific cell.

        Parameters
        ----------
        cell_id : int
            The cell ID (from suite2p)
        session_name : str, optional
            Session name. If None and cell_id is unique, finds it automatically.
        distance : int, optional
            Filter by distance (0-3, or 4 for control). None for all.
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        include_control : bool
            Whether to include control trials. Default False.

        Returns
        -------
        traces : np.ndarray
            Traces for this cell, shape (n_trials, n_angles)
        distances : np.ndarray
            Distance label for each trial, shape (n_trials,)
        """
        # Find the cell in the DataFrame
        if session_name is not None:
            mask = (self._df['cell_id'] == cell_id) & (self._df['session_name'] == session_name)
        else:
            mask = self._df['cell_id'] == cell_id

        matches = self._df[mask]

        if len(matches) == 0:
            raise ValueError(f"Cell {cell_id} not found" +
                           (f" in session {session_name}" if session_name else ""))
        if len(matches) > 1 and session_name is None:
            sessions = matches['session_name'].tolist()
            raise ValueError(f"Cell {cell_id} found in multiple sessions: {sessions}. "
                           "Please specify session_name.")

        idx = matches.index[0]
        session = matches.loc[idx, 'session_name']

        # Get trial mask
        trial_mask = self._get_trial_mask(session, distance, monocular, include_control)

        # Get traces and distances
        traces = self._traces[idx][trial_mask]
        distances = self._trial_info[session]['distances'][trial_mask]

        return traces, distances

    def _filter_modulated_cells(
        self,
        cell_indices: List[int],
        modulation_type: str
    ) -> List[int]:
        """
        Filter cell indices to only include modulated cells.

        Parameters
        ----------
        cell_indices : List[int]
            DataFrame indices of cells to filter.
        modulation_type : str
            Type of modulation: 'monocular', 'binocular', 'both', or 'any'.

        Returns
        -------
        List[int]
            Filtered cell indices.
        """
        filtered = []
        for idx in cell_indices:
            row = self._df.loc[idx]
            mono_pval = row['monocular']['bootstrap_pvalue']
            bino_pval = row['binocular']['bootstrap_pvalue']

            mono_mod = mono_pval < 0.05
            bino_mod = bino_pval < 0.05

            if modulation_type == 'monocular':
                if mono_mod and not bino_mod:
                    filtered.append(idx)
            elif modulation_type == 'binocular':
                if bino_mod and not mono_mod:
                    filtered.append(idx)
            elif modulation_type == 'both':
                if mono_mod and bino_mod:
                    filtered.append(idx)
            elif modulation_type == 'any':
                if mono_mod or bino_mod:
                    filtered.append(idx)

        return filtered

    def get_decoder_data(
        self,
        monocular: int = None,
        area: str = None,
        session: str = None,
        include_control: bool = False,
        modulated_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data formatted for decoder training.

        Format: 3D array preserving angular structure.

        Parameters
        ----------
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.
        session : str, optional
            Session name to filter by. None for all sessions.
        include_control : bool
            Whether to include control trials (distance=4). Default False.
        modulated_only : bool
            If True, only include cells modulated for the specified condition.
            When monocular=0, filters for binocular modulated cells.
            When monocular=1, filters for monocular modulated cells.
            When monocular=None, filters for cells modulated in either condition.
            Default False.

        Returns
        -------
        X : np.ndarray
            Neural responses, shape (n_neurons, n_angles, n_trials)
        y_distance : np.ndarray
            Distance labels per trial, shape (n_trials,)
        """
        cell_indices = self._get_cell_indices(area)

        # Filter by modulation if specified
        if modulated_only:
            if monocular == 0:
                modulation_type = 'binocular'
            elif monocular == 1:
                modulation_type = 'monocular'
            else:
                modulation_type = 'any'
            cell_indices = self._filter_modulated_cells(cell_indices, modulation_type)

        # Filter by session if specified
        session = self.resolve_session(session)
        if session is not None:
            if session not in self._trial_info:
                available = list(self._trial_info.keys())
                raise ValueError(f"Session '{session}' not found. Available: {available}")
            cell_indices = [idx for idx in cell_indices
                           if self._df.loc[idx, 'session_name'] == session]

        # First pass: find minimum number of trials across all cells
        min_trials = float('inf')
        valid_cells = []

        for idx in cell_indices:
            session_name = self._df.loc[idx, 'session_name']
            mask = self._get_trial_mask(session_name, None, monocular, include_control)

            if mask.sum() > 0:
                valid_cells.append((idx, session_name, mask))
                min_trials = min(min_trials, mask.sum())

        if len(valid_cells) == 0:
            raise ValueError("No valid cells found for the given conditions")

        min_trials = int(min_trials)

        # Second pass: collect data with consistent trial count
        all_responses = []
        y_distance = None

        for idx, session_name, mask in valid_cells:
            cell_traces = self._traces[idx][mask]  # (n_trials, n_angles)

            # Truncate to min_trials
            cell_traces = cell_traces[:min_trials]  # (min_trials, n_angles)

            # Transpose to (n_angles, n_trials)
            all_responses.append(cell_traces.T)

            # Get distance labels (same for all cells since we truncate consistently)
            if y_distance is None:
                distances = self._trial_info[session_name]['distances'][mask]
                y_distance = distances[:min_trials]

        # Stack: (n_neurons, n_angles, n_trials)
        X = np.stack(all_responses, axis=0)

        return X, y_distance

    def get_train_test_split(
        self,
        test_fraction: float = 0.2,
        stratify_by_distance: bool = True,
        monocular: int = None,
        area: str = None,
        random_state: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get stratified train/test indices into the trial dimension.

        Parameters
        ----------
        test_fraction : float
            Fraction of data for testing (default: 0.2)
        stratify_by_distance : bool
            Whether to stratify by distance (default: True)
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        train_indices : np.ndarray
            Indices into trial dimension for training
        test_indices : np.ndarray
            Indices into trial dimension for testing
        """
        _, y_distance = self.get_decoder_data(monocular=monocular, area=area)

        n_trials = len(y_distance)
        indices = np.arange(n_trials)

        stratify = y_distance if stratify_by_distance else None

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_fraction,
            stratify=stratify,
            random_state=random_state
        )

        return train_idx, test_idx

    def get_kfold_splits(
        self,
        n_folds: int = 5,
        stratify_by_distance: bool = True,
        monocular: int = None,
        area: str = None,
        random_state: int = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generator yielding train/test indices for k-fold CV.

        Parameters
        ----------
        n_folds : int
            Number of folds (default: 5)
        stratify_by_distance : bool
            Whether to stratify by distance (default: True)
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.
        random_state : int, optional
            Random seed for reproducibility.

        Yields
        ------
        train_indices : np.ndarray
            Indices into trial dimension for training
        test_indices : np.ndarray
            Indices into trial dimension for testing
        """
        _, y_distance = self.get_decoder_data(monocular=monocular, area=area)

        n_trials = len(y_distance)
        indices = np.arange(n_trials)

        if stratify_by_distance:
            kfold = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state
            )
            splits = kfold.split(indices, y_distance)
        else:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=random_state
            )
            splits = kfold.split(indices)

        for train_idx, test_idx in splits:
            yield train_idx, test_idx

    def compute_rf_positions(
        self,
        monocular: int = None,
        area: str = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Compute receptive field (RF) positions for all cells.

        RF is defined as the angle position with maximum mean response
        across all trials for each cell.

        Parameters
        ----------
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.

        Returns
        -------
        rf_positions : np.ndarray, shape (n_cells,)
            RF position (angle index) for each cell.
        cell_indices : List[int]
            DataFrame indices of included cells.
        """
        X, _ = self.get_decoder_data(monocular=monocular, area=area)
        # X shape: (n_neurons, n_angles, n_trials)

        # Mean across trials: (n_neurons, n_angles)
        mean_responses = np.mean(X, axis=2)

        # RF at peak of mean response
        rf_positions = np.argmax(mean_responses, axis=1)

        # Get cell indices
        cell_indices = self._get_cell_indices(area)

        return rf_positions, cell_indices

    def get_rf_distribution(
        self,
        monocular: int = None,
        area: str = None
    ) -> Dict:
        """
        Get RF position distribution for population analysis.

        Parameters
        ----------
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.

        Returns
        -------
        dict with:
            rf_positions : np.ndarray
                RF position for each cell.
            n_positions : int
                Total number of angle positions.
            coverage : np.ndarray
                Number of cells with RF at each position.
            cell_indices : List[int]
                DataFrame indices of cells.
        """
        rf_positions, cell_indices = self.compute_rf_positions(
            monocular=monocular, area=area
        )

        # Count cells per RF position
        coverage = np.bincount(rf_positions, minlength=self.n_positions)

        return {
            'rf_positions': rf_positions,
            'n_positions': self.n_positions,
            'coverage': coverage,
            'cell_indices': cell_indices,
        }

    def get_decoder_data_by_condition(
        self,
        monocular: int = None,
        area: str = None,
        session: str = None,
        distances: List[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get data pooled by stimulus condition (distance).

        Averages all trials of the same distance condition per cell FIRST,
        then pools across sessions. This ensures unambiguous stimulus matching
        across sessions regardless of trial order.

        Parameters
        ----------
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.
        session : str, optional
            Session name to filter by. None for all sessions.
        distances : List[int], optional
            Distance values to include. Default [0, 1, 2, 3].

        Returns
        -------
        X : np.ndarray, shape (n_neurons, n_angles, n_distances)
            Mean response per neuron per angle per distance condition.
        distance_values : np.ndarray, shape (n_distances,)
            Distance values corresponding to last axis of X.
        """
        if distances is None:
            distances = [0, 1, 2, 3]
        distances = np.array(distances)

        cell_indices = self._get_cell_indices(area)

        # Filter by session if specified
        if session is not None:
            if session not in self._trial_info:
                available = list(self._trial_info.keys())
                raise ValueError(f"Session '{session}' not found. Available: {available}")
            cell_indices = [idx for idx in cell_indices
                           if self._df.loc[idx, 'session_name'] == session]

        # Filter valid cells
        valid_cells = []
        for idx in cell_indices:
            session_name = self._df.loc[idx, 'session_name']
            # Check that this cell has data for the requested monocular condition
            mask = self._get_trial_mask(session_name, None, monocular, include_control=False)
            if mask.sum() > 0:
                valid_cells.append((idx, session_name))

        if len(valid_cells) == 0:
            raise ValueError("No valid cells found for the given conditions")

        n_neurons = len(valid_cells)
        n_angles = self._target_length
        n_distances = len(distances)

        # Collect condition-averaged data
        X = np.zeros((n_neurons, n_angles, n_distances))

        for neuron_idx, (df_idx, session_name) in enumerate(valid_cells):
            cell_traces = self._traces[df_idx]  # (n_trials, n_angles)
            info = self._trial_info[session_name]

            for dist_idx, dist_val in enumerate(distances):
                # Get trials matching this distance (and monocular condition)
                dist_mask = info['distances'] == dist_val
                if monocular is not None:
                    dist_mask &= info['monocular'] == monocular

                if dist_mask.sum() > 0:
                    # Average all trials with this distance
                    X[neuron_idx, :, dist_idx] = np.mean(cell_traces[dist_mask], axis=0)
                else:
                    # No trials for this condition - use NaN or zero
                    X[neuron_idx, :, dist_idx] = np.nan

        return X, distances

    def get_split_decoder_data(
        self,
        monocular: int = None,
        area: str = None,
        session: str = None,
        distances: List[int] = None,
        test_fraction: float = 0.5,
        random_state: int = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get train/test data split at the trial level, then pooled by condition.

        Both train and test sets are constructed the same way:
        1. Split trials randomly into train/test halves (stratified by distance)
        2. Average within condition for each half
        3. Return condition-pooled train and test arrays

        Parameters
        ----------
        monocular : int, optional
            0 for binocular, 1 for monocular, None for all.
        area : str, optional
            Visual area ('V1', 'LM', etc.). None for all areas.
        session : str, optional
            Session name to filter by. None for all sessions.
        distances : List[int], optional
            Distance values to include. Default [0, 1, 2, 3].
        test_fraction : float
            Fraction of trials for test set. Default 0.5.
        random_state : int, optional
            Random seed for reproducibility.

        Returns
        -------
        X_train : np.ndarray, shape (n_neurons, n_angles, n_distances)
            Condition-averaged training data.
        X_test : np.ndarray, shape (n_neurons, n_angles, n_distances)
            Condition-averaged test data.
        distances : np.ndarray, shape (n_distances,)
            Distance values.
        """
        if distances is None:
            distances = [0, 1, 2, 3]
        distances = np.array(distances)

        rng = np.random.RandomState(random_state)

        cell_indices = self._get_cell_indices(area)

        # Filter by session if specified
        if session is not None:
            if session not in self._trial_info:
                available = list(self._trial_info.keys())
                raise ValueError(f"Session '{session}' not found. Available: {available}")
            cell_indices = [idx for idx in cell_indices
                           if self._df.loc[idx, 'session_name'] == session]

        # Filter valid cells
        valid_cells = []
        for idx in cell_indices:
            session_name = self._df.loc[idx, 'session_name']
            mask = self._get_trial_mask(session_name, None, monocular, include_control=False)
            if mask.sum() > 0:
                valid_cells.append((idx, session_name))

        if len(valid_cells) == 0:
            raise ValueError("No valid cells found for the given conditions")

        n_neurons = len(valid_cells)
        n_angles = self._target_length
        n_distances = len(distances)

        X_train = np.zeros((n_neurons, n_angles, n_distances))
        X_test = np.zeros((n_neurons, n_angles, n_distances))

        for neuron_idx, (df_idx, session_name) in enumerate(valid_cells):
            cell_traces = self._traces[df_idx]  # (n_trials, n_angles)
            info = self._trial_info[session_name]

            for dist_idx, dist_val in enumerate(distances):
                # Get trials matching this distance (and monocular condition)
                dist_mask = info['distances'] == dist_val
                if monocular is not None:
                    dist_mask &= info['monocular'] == monocular

                matching_trials = np.where(dist_mask)[0]

                if len(matching_trials) >= 2:
                    # Split trials for this condition
                    n_test = max(1, int(len(matching_trials) * test_fraction))
                    shuffled = rng.permutation(matching_trials)
                    test_trials = shuffled[:n_test]
                    train_trials = shuffled[n_test:]

                    # Average within each split
                    X_train[neuron_idx, :, dist_idx] = np.mean(cell_traces[train_trials], axis=0)
                    X_test[neuron_idx, :, dist_idx] = np.mean(cell_traces[test_trials], axis=0)
                elif len(matching_trials) == 1:
                    # Only one trial - use for both (not ideal but avoids NaN)
                    X_train[neuron_idx, :, dist_idx] = cell_traces[matching_trials[0]]
                    X_test[neuron_idx, :, dist_idx] = cell_traces[matching_trials[0]]
                else:
                    # No trials for this condition
                    X_train[neuron_idx, :, dist_idx] = np.nan
                    X_test[neuron_idx, :, dist_idx] = np.nan

        return X_train, X_test, distances

