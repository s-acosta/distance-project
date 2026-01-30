"""
Batch Session Analysis Script

Run analysis on multiple sessions from a session list file.
If a session fails, the script continues with the next session.

Usage:
    python run-scripts/run_batch_analysis.py sessions.txt
    python run-scripts/run_batch_analysis.py sessions.txt --force-recompute
    python run-scripts/run_batch_analysis.py sessions.txt --n-iterations 500
"""

import sys
import argparse
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

from src.workflows import batch_analyze_sessions


def main():
    parser = argparse.ArgumentParser(description='Batch analyze sessions from a session list file')
    parser.add_argument('session_list', type=str, help='Path to session list file (CSV format: mouse_name,record_date,experiment_name)')
    parser.add_argument('--force-recompute', action='store_true',
                        help='Recompute DataFrames even if they already exist')
    parser.add_argument('--n-iterations', type=int, default=200,
                        help='Number of bootstrap iterations (default: 200)')
    parser.add_argument('--base-data-dir', type=str,
                        default=r'D:/data-distance-coding-project/processed-data',
                        help='Base directory for processed data')

    args = parser.parse_args()

    print(f"Batch Analysis Script")
    print(f"Session list: {args.session_list}")
    print(f"Force recompute: {args.force_recompute}")
    print(f"Bootstrap iterations: {args.n_iterations}")
    print("-" * 60)

    dfs = batch_analyze_sessions(
        session_list_path=args.session_list,
        base_data_dir=args.base_data_dir,
        n_iterations=args.n_iterations,
        save_each=True,
        force_recompute=args.force_recompute
    )

    print("\nDone!")
    print(f"Successfully analyzed {len(dfs)} sessions")


if __name__ == "__main__":
    main()
