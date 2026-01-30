"""
High-level workflow functions for distance coding project analysis.

This module provides convenience functions that orchestrate the existing
analysis classes for common tasks like single-session analysis, cell visualization,
batch processing, and widefield alignment.

Example usage:
    from src.workflows import analyze_session, visualize_cell

    # Quick session analysis
    df, device = analyze_session('SAM041', '251128', 'tiling_3_120um')

    # Dive into specific cell
    visualize_cell(device, cell_id=42)
"""
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from src.data.RadialDevice import RadialDevice
from src.analysis.CellAnalysis import CellAnalyzer, SessionAnalyzer
from src.analysis.SessionCombiner import SessionCombiner
from src.viz.CellVisualizer import CellVisualizer
from src.viz.SessionVisualizer import SessionVisualizer
from src.data.WidefieldTransformer import WidefieldAligner, WidefieldMapper

def analyze_session(
    mouse_name: str,
    record_date: str,
    experiment_name: str,
    base_data_dir: str = r'D:/data-distance-coding-project/processed-data',
    show_plots: bool = True,
    save_df: bool = True,
    n_iterations: int = 200,
    force_recompute: bool = False
) -> tuple:
    """
    Complete single-session analysis workflow.

    Loads a session, computes the analysis DataFrame with cell statistics,
    optionally shows summary plots and saves the DataFrame.

    Parameters
    ----------
    mouse_name : str
        Mouse identifier (e.g., 'SAM041')
    record_date : str
        Recording date (e.g., '251128')
    experiment_name : str
        Experiment name (e.g., 'tiling_3_120um')
    base_data_dir : str, optional
        Base directory for processed data
    show_plots : bool, optional
        Whether to show summary plots (default: True)
    save_df : bool, optional
        Whether to save the DataFrame to disk (default: True)
    n_iterations : int, optional
        Number of bootstrap iterations for analysis (default: 200)
    force_recompute : bool, optional
        If True, recompute DataFrame even if it exists on disk (default: False)

    Returns
    -------
    tuple of (pd.DataFrame, RadialDevice)
        df : Session DataFrame with cell statistics
        device : RadialDevice instance for further analysis

    Examples
    --------
    >>> df, device = analyze_session('SAM041', '251128', 'tiling_3_120um')
    >>> print(f"Analyzed {len(df)} cells")
    """
    # Load session
    device = RadialDevice(
        mouse_name=mouse_name,
        record_date=record_date,
        experiment_name=experiment_name,
        base_data_dir=base_data_dir
    )

    # Check if DataFrame already exists
    session_name = device.session_name
    df_path = os.path.join(device.data_dir, f"{session_name}_dataframe.pkl")

    if os.path.exists(df_path) and not force_recompute:
        print(f"[ok] Loading existing DataFrame from {df_path}")
        with open(df_path, 'rb') as f:
            session_df = pickle.load(f)
    else:
        # Run analysis
        print(f"[...] Computing analysis for {session_name}")
        analyzer = SessionAnalyzer(device)
        session_df = analyzer.analyze_session_stats(n_iterations=n_iterations)

        # Save if requested
        if save_df:
            with open(df_path, 'wb') as f:
                pickle.dump(session_df, f)
            print(f"[ok] Saved DataFrame to {df_path}")

    # Show plots if requested
    if show_plots:
        visualizer = SessionVisualizer(device, session_df)
        visualizer.plot_modulation_bars()
        visualizer.plot_distance_bars()
        plt.show()

    return session_df, device

def combine_animal_sessions(
    animal_name: str,
    experiment_type: str,
    base_data_dir: str = r'D:/data-distance-coding-project/processed-data',
    base_multi_session_dir: str = r'D:/data-distance-coding-project/multi-session-data',
    show_summary: bool = True,
    save_combined: bool = True
) -> pd.DataFrame:
    """
    Combine all sessions for an animal into one DataFrame.

    Reads session list from txt file, loads or computes DataFrames for each,
    and combines them into a single DataFrame.

    Parameters
    ----------
    animal_name : str
        Animal identifier (e.g., 'SAM041')
    experiment_type : str
        Experiment type (e.g., 'tiling120um')
    base_data_dir : str, optional
        Base directory for processed data
    run_scripts_dir : str, optional
        Directory where session list files are located
    show_summary : bool, optional
        Whether to print summary statistics (default: True)
    save_combined : bool, optional
        Whether to save the combined DataFrame (default: True)

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all sessions

    Examples
    --------
    >>> combined_df = combine_animal_sessions('SAM041', 'tiling120um')
    >>> print(f"Combined {len(combined_df)} cells from all sessions")
    """

    aligner = WidefieldAligner(animal_name, experiment_type)
    mapper = WidefieldMapper(aligner.aligner_params)
    
    combiner = SessionCombiner(
        mouse_name=animal_name,
        experiment_name=experiment_type,
        base_data_dir=base_data_dir,
        base_multi_session_dir=base_multi_session_dir,
        widefield_mapper=mapper
    )

    combined_df = combiner.combine_sessions()

    if show_summary:
        summary = combiner.get_session_summary(combined_df)
        print("\nSession Summary:")
        print(summary.to_string(index=False))
        print(f"\nTotal cells: {len(combined_df)}")

    if save_combined:
        combiner.save_combined_df(combined_df)

    return combined_df

def align_to_widefield(
    animal_name: str,
    experiment_type: str
) -> dict:
    """
    Launch interactive widefield alignment for an animal's sessions.

    Opens interactive alignment tool to align 2P sessions to widefield
    reference image and draw visual area boundaries.

    Parameters
    ----------
    animal_name : str
        Animal identifier (e.g., 'SAM041')
    experiment_type : str
        Experiment type (e.g., 'tiling120um')

    Returns
    -------
    dict
        Dictionary containing alignment parameters:
        - wf_maps : Widefield maps (img, VFS, azi, alt)
        - areas_verts : Visual area boundary vertices
        - wf_transformations : Per-session transformation parameters

    Examples
    --------
    >>> params = align_to_widefield('SAM041', 'tiling120um')
    >>> # Interactive alignment is performed
    >>> print(f"Aligned {len(params['wf_transformations'])} sessions")
    """
    from src.data.WidefieldTransformer import WidefieldAligner

    aligner = WidefieldAligner(animal_name, experiment_type)

    return aligner.aligner_params

def batch_analyze_sessions(
    session_list_path: str,
    base_data_dir: str = r'D:/data-distance-coding-project/processed-data',
    n_iterations: int = 200,
    save_each: bool = True,
    force_recompute: bool = False
) -> list:
    """
    Analyze multiple sessions from a session list file.

    Reads session information from a txt file and analyzes each session,
    computing DataFrames for sessions that don't already have them.

    Parameters
    ----------
    session_list_path : str
        Path to session list file (format: mouse_name,record_date,experiment_name per line)
    base_data_dir : str, optional
        Base directory for processed data
    n_iterations : int, optional
        Number of bootstrap iterations for analysis (default: 200)
    save_each : bool, optional
        Whether to save each DataFrame after computing (default: True)

    Returns
    -------
    list of pd.DataFrame
        List of session DataFrames

    Examples
    --------
    >>> dfs = batch_analyze_sessions('run-scripts/SAM041_tiling120um_sessions.txt')
    >>> print(f"Analyzed {len(dfs)} sessions")
    """
    # Parse session list file
    sessions = []
    with open(session_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line.startswith('//') or not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 3:
                sessions.append({
                    'mouse_name': parts[0],
                    'record_date': parts[1],
                    'experiment_name': parts[2]
                })

    print(f"[batch] Found {len(sessions)} sessions to process")

    all_dfs = []
    failed_sessions = []
    for i, session in enumerate(sessions):
        session_id = f"{session['record_date']}_{session['mouse_name']}_{session['experiment_name']}"
        print(f"\n[{i+1}/{len(sessions)}] Processing: {session_id}")

        try:
            df, _ = analyze_session(
                mouse_name=session['mouse_name'],
                record_date=session['record_date'],
                experiment_name=session['experiment_name'],
                base_data_dir=base_data_dir,
                show_plots=False,
                save_df=save_each,
                n_iterations=n_iterations,
                force_recompute=force_recompute
            )
            all_dfs.append(df)
            print(f"    Analyzed {len(df)} cells")
        except Exception as e:
            print(f"[ERROR] Session {session_id} failed: {e}")
            print(f"    Continuing with next session...")
            failed_sessions.append(session_id)

    print(f"\n[batch] Completed {len(all_dfs)}/{len(sessions)} sessions")
    if failed_sessions:
        print(f"[batch] Failed sessions ({len(failed_sessions)}):")
        for s in failed_sessions:
            print(f"    - {s}")
    return all_dfs

def batch_combine_and_export(
    animal_name: str,
    experiment_type: str,
    output_path: str = None,
    export_csv: bool = True,
) -> pd.DataFrame:
    """
    Batch process sessions, combine, and export results.

    Combines all sessions for an animal and exports to CSV and/or pickle.

    Parameters
    ----------
    animal_name : str
        Animal identifier (e.g., 'SAM041')
    experiment_type : str
        Experiment type (e.g., 'tiling120um')
    output_path : str, optional
        Custom output path. If None, uses default location.
    export_csv : bool, optional
        Whether to also export as CSV (default: True)

    Returns
    -------
    pd.DataFrame
        Combined DataFrame

    Examples
    --------
    >>> df = batch_combine_and_export('SAM041', 'tiling120um')
    >>> # Exports to pickle and CSV
    """
    combined_df = combine_animal_sessions(
        animal_name=animal_name,
        experiment_type=experiment_type,
        show_summary=True,
        save_combined=True
    )

    if export_csv and output_path:
        csv_path = output_path.replace('.pkl', '.csv') if output_path.endswith('.pkl') else output_path + '.csv'

        # Flatten nested columns for CSV export
        flat_df = combined_df.copy()
        for col in ['monocular', 'binocular']:
            if col in flat_df.columns:
                nested_cols = flat_df[col].apply(pd.Series)
                nested_cols.columns = [f"{col}_{c}" for c in nested_cols.columns]
                flat_df = flat_df.drop(columns=[col]).join(nested_cols)

        flat_df.to_csv(csv_path, index=False)
        print(f"[ok] Exported CSV to {csv_path}")

    return combined_df
