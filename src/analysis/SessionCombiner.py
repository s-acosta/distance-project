import sys
import os
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

import numpy as np
import pandas as pd
import pickle
from src.data.RadialDevice import RadialDevice
from src.analysis.CellAnalysis import SessionAnalyzer

class SessionCombiner:
    """
    Combines multiple session DataFrames into a unified multi-session DataFrame.

    Reads session lists from txt files, automatically computes missing DataFrames,
    and combines them with session metadata for cross-session analysis.

    Parameters
    ----------
    session_list_names : str or list of str
        Name(s) of txt file(s) containing session lists (without .txt extension).
        Files are expected to be in the 'run-scripts' directory.
        Format per line: mouse_name,record_date,experiment_name
        Lines starting with # or // are treated as comments.
    base_data_dir : str, optional
        Base directory for processed data.
        Default: r'D:\\data-distance-coding-project\\processed-data'
    run_scripts_dir : str, optional
        Directory where session list txt files are stored.
        Default: r'C:\\Users\\acostamendoza\\projects\\distance-project\\run-scripts'
    widefield_mapper : WidefieldTransformer, optional
        If provided, will append widefield coordinates to each session DataFrame.

    Examples
    --------
    >>> from src.analysis.SessionCombiner import SessionCombiner
    >>>
    >>> # Combine sessions from one mouse
    >>> combiner = SessionCombiner('SAM041_tiling120um_sessions')
    >>> combined_df = combiner.combine_sessions()
    >>>
    >>> # Combine sessions from multiple mice
    >>> combiner = SessionCombiner(['SAM040_tiling120um_sessions',
    ...                              'SAM041_tiling120um_sessions'])
    >>> combined_df = combiner.combine_sessions()
    >>>
    >>> # With widefield coordinates
    >>> from src.data.WidefieldTransformer import WidefieldMapper
    >>> mapper = WidefieldMapper(config_path='run-scripts/SAM041_tiling120um_sessions.txt')
    >>> combiner = SessionCombiner('SAM041_tiling120um_sessions',
    ...                            widefield_mapper=mapper)
    >>> combined_df = combiner.combine_sessions()

    """

    def __init__(self, mouse_name, experiment_name,
                 base_data_dir=r'D:\data-distance-coding-project\processed-data',
                 base_multi_session_dir=r'D:\data-distance-coding-project\multi-session-data',
                 widefield_mapper=None):
        """
        Initialize the SessionCombiner.

        Parameters
        ----------
        session_list_names : str or list of str
            Name(s) of txt file(s) containing session information (e.g., 'SAM041_tiling120um_sessions')
            File extension (.txt) is optional and will be added automatically if missing.
        base_data_dir : str, optional
            Base directory where processed data is stored
        run_scripts_dir : str, optional
            Directory where session list txt files are located
        widefield_mapper : WidefieldTransformer, optional
            Mapper instance for adding widefield coordinates
        """

        self.mouse_name = mouse_name
        self.experiment_name = experiment_name
        self.session_list_name = f'{mouse_name}_{experiment_name}_sessions.txt'
        self.base_multi_session_dir = base_multi_session_dir
        self.multi_session_dir = os.path.join(self.base_multi_session_dir, f'{mouse_name}_{experiment_name}')

        # Build full path
        full_path = os.path.join(self.multi_session_dir, self.session_list_name)

        # Verify file exists
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Session list file not found: {full_path}")

        self.session_list_name = full_path

        self.base_data_dir = base_data_dir
        self.widefield_mapper = widefield_mapper

        # Parse all session files
        self.sessions = self._parse_session_files()

        print(f"[SessionCombiner] Found {len(self.sessions)} sessions across {self.session_list_name}")

    def _parse_session_files(self):
        """
        Parse session list files and extract session information.

        Returns
        -------
        list of dict
            List of session dictionaries with keys: mouse_name, record_date,
            experiment_name, session_name
        """
        sessions = []

        with open(self.session_list_name, 'r') as f:

            for line in f:
                if line.startswith('#') or line.startswith('//') or line.strip() == '':
                    continue
                line = line.strip()
                mouse_name, record_date, experiment_name = [p.strip() for p in line.split(',')]
                session_name = f"{record_date}_{mouse_name}_{experiment_name}"

                sessions.append({
                    'mouse_name': mouse_name,
                    'record_date': record_date,
                    'experiment_name': experiment_name,
                    'session_name': session_name
                })

        return sessions

    def _check_dataframe_exists(self, session_name):
        """
        Check if a dataframe exists for a session.

        By default, checks for registered dataframe first (has widefield coords),
        then falls back to standard dataframe.

        Parameters
        ----------
        session_name : str
            Session identifier (e.g., '251128_SAM041_tiling_3_120um')
        use_registered : bool, optional
            If True, prefer registered dataframe over standard (default: True).
            If False, only check for standard dataframe.

        Returns
        -------
        tuple of (str or None, bool)
            (path_to_dataframe, is_registered) or (None, False) if not found
        """
        session_dir = os.path.join(self.base_data_dir, session_name)

        # Check for registered dataframe first (has widefield coords)
        
        df_path = os.path.join(session_dir, f"{session_name}_dataframe.pkl")
        if os.path.exists(df_path):
            return df_path
        else:
            return None

    def _compute_session_dataframe(self, mouse_name, record_date, experiment_name,
                                   session_name, n_iterations=200):
        """
        Compute a session DataFrame using RadialDevice and SessionAnalyzer.

        Parameters
        ----------
        mouse_name : str
            Mouse identifier
        record_date : str
            Recording date
        experiment_name : str
            Experiment name
        session_name : str
            Session identifier
        n_iterations : int, optional
            Number of bootstrap iterations for analysis

        Returns
        -------
        pd.DataFrame
            Session DataFrame with cell statistics
        """
        print(f"  Computing DataFrame for {session_name}...")

        # Create RadialDevice
        device = RadialDevice(
            mouse_name=mouse_name,
            record_date=record_date,
            experiment_name=experiment_name,
            base_data_dir=self.base_data_dir
        )

        # Run analysis
        analyzer = SessionAnalyzer(device)
        session_df = analyzer.analyze_session_stats(n_iterations=n_iterations)

        # Save the dataframe
        session_dir = os.path.join(self.base_data_dir, session_name)
        df_path = os.path.join(session_dir, f"{session_name}_dataframe.pkl")

        with open(df_path, 'wb') as f:
            pickle.dump(session_df, f)
        print(f"  Saved DataFrame to {df_path}")

        return session_df

    def _load_session_dataframe(self, df_path):
        """
        Load a session DataFrame from disk.

        Parameters
        ----------
        df_path : str
            Path to the pickle file

        Returns
        -------
        pd.DataFrame
            Loaded session DataFrame
        """
        with open(df_path, 'rb') as f:
            return pickle.load(f)

    def _add_widefield_coords(self, session_name, session_df):
        """
        Add widefield coordinates to a session DataFrame.

        Parameters
        ----------
        session_name : str
            Session identifier
        session_df : pd.DataFrame
            Session DataFrame to augment

        Returns
        -------
        pd.DataFrame
            DataFrame with widefield coordinates added
        """
        if self.widefield_mapper is None:
            print(f"  Warning: No widefield mapper provided. Skipping widefield coordinates.")
            return session_df

        print(f"  Adding widefield coordinates for {session_name}...")
        session_df = self.widefield_mapper.append_widefield_to_session_df(session_name)
        return session_df

    def combine_sessions(self, n_iterations=200, add_widefield=True):
        """
        Combine all sessions into a single DataFrame.

        For each session:
        1. Check if registered DataFrame exists (has widefield coords), prefer that
        2. If not registered, check for standard DataFrame
        3. If neither exists, compute it using SessionAnalyzer
        4. If widefield coords are missing and mapper provided, add them and save as registered
        5. Add session_name column
        6. Concatenate into unified DataFrame

        Parameters
        ----------
        n_iterations : int, optional
            Number of bootstrap iterations when computing new DataFrames (default: 200)
        force_recompute : bool, optional
            If True, recompute all DataFrames even if they exist (default: False)
        add_widefield : bool, optional
            If True and widefield_mapper is provided, add widefield coordinates to
            DataFrames that don't have them yet (default: True)

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with all sessions, including:
            - session_name : str (e.g., '251128_SAM041_tiling_3_120um')
            Plus all original cell analysis columns and widefield coords if available
        """
        all_dfs = []

        for i, session_info in enumerate(self.sessions):
            mouse_name = session_info['mouse_name']
            record_date = session_info['record_date']
            experiment_name = session_info['experiment_name']
            session_name = session_info['session_name']

            print(f"\n[{i+1}/{len(self.sessions)}] Processing session: {session_name}")

            # Check if dataframe exists (prefer registered unless force_recompute)
            df_path = self._check_dataframe_exists(session_name)

            if df_path is not None:
                print(f"  Loading DataFrame from {os.path.basename(df_path)}")
                session_df = self._load_session_dataframe(df_path)
            else:
                session_df = self._compute_session_dataframe(mouse_name, record_date, experiment_name, session_name, n_iterations)
                df_path = os.path.join(self.base_data_dir, session_name, f"{session_name}_dataframe.pkl")
                with open(df_path, 'wb') as f:
                    pickle.dump(session_df, f)
                print(f"  Saved DataFrame to {df_path} as standard dataframe")

            # Add widefield coordinates if requested and not already present
            # Note: _add_widefield_coords will save the result as _registered.pkl
            if add_widefield and 'wf_coords' not in session_df.columns:
                session_df = self._add_widefield_coords(session_name, session_df)
            elif 'wf_coords' in session_df.columns:
                print(f"  Widefield coordinates already present")

            # Add session metadata column
            session_df = session_df.copy()  # Avoid modifying original
            session_df['session_name'] = session_name

            all_dfs.append(session_df)
            print(f"  Added {len(session_df)} cells from this session")

        # Combine all dataframes
        combined_df = pd.concat(all_dfs, ignore_index=True)

        print(f"\n[SessionCombiner] Combined {len(combined_df)} cells from {len(self.sessions)} sessions")

        return combined_df

    def get_session_summary(self, combined_df=None):
        """
        Get summary statistics for each session.

        Parameters
        ----------
        combined_df : pd.DataFrame, optional
            Combined DataFrame. If None, will call combine_sessions()

        Returns
        -------
        pd.DataFrame
            Summary DataFrame with one row per session showing n_cells
        """
        if combined_df is None:
            combined_df = self.combine_sessions()

        summary = combined_df.groupby('session_name').agg({
            'cell_id': 'count',
        }).rename(columns={'cell_id': 'n_cells'})

        return summary.reset_index()

    def save_combined_df(self, combined_df):

        save_path = os.path.join(self.multi_session_dir, f'{self.mouse_name}_{self.experiment_name}_combined_sessions.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(combined_df, f)
        print(f"Combined DataFrame saved to {save_path}")

        return save_path