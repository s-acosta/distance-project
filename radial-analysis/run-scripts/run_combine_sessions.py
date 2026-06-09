"""
Combine Sessions Script

Combine all individual session DataFrames for an animal into a single combined DataFrame.

Usage:
    python run-scripts/run_combine_sessions.py SAM041 tiling_120um
    python run-scripts/run_combine_sessions.py SAM041 tiling_120um --no-save
    python run-scripts/run_combine_sessions.py SAM041 tiling_120um --no-summary
"""

import sys
import argparse
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project\radial-analysis")
from path_setup import setup_project_path
setup_project_path()

from src.workflows import combine_animal_sessions


def main():
    parser = argparse.ArgumentParser(description='Combine all session DataFrames for an animal')
    parser.add_argument('mouse_name', type=str, help='Mouse ID (e.g., SAM041)')
    parser.add_argument('experiment_name', type=str, help='Experiment name (e.g., tiling_120um)')
    parser.add_argument('--base-data-dir', type=str,
                        default=r'D:/data-distance-coding-project/processed-data',
                        help='Base directory for processed data')
    parser.add_argument('--base-multi-session-dir', type=str,
                        default=r'D:/data-distance-coding-project/multi-session-data',
                        help='Base directory for multi-session combined data')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the combined DataFrame')
    parser.add_argument('--no-summary', action='store_true',
                        help='Do not print summary statistics')

    args = parser.parse_args()

    print(f"Combine Sessions Script")
    print(f"Mouse: {args.mouse_name}")
    print(f"Experiment: {args.experiment_name}")
    print("-" * 60)

    combined_df = combine_animal_sessions(
        animal_name=args.mouse_name,
        experiment_type=args.experiment_name,
        base_data_dir=args.base_data_dir,
        base_multi_session_dir=args.base_multi_session_dir,
        show_summary=not args.no_summary,
        save_combined=not args.no_save
    )

    print("\nDone!")
    print(f"Combined DataFrame has {len(combined_df)} cells")


if __name__ == "__main__":
    main()
