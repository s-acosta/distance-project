"""
Command-line interface for distance coding project analysis.

Usage:
    # Single session analysis
    python -m src.cli analyze --mouse SAM041 --date 251128 --experiment tiling_3_120um

    # Batch analysis from session list
    python -m src.cli batch --session-list run-scripts/SAM041_tiling120um_sessions.txt

    # Combine sessions for an animal
    python -m src.cli combine --animal SAM041 --experiment tiling120um

    # Widefield alignment
    python -m src.cli align --animal SAM041 --experiment tiling120um
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Distance coding project analysis CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s analyze --mouse SAM041 --date 251128 --experiment tiling_3_120um
  %(prog)s batch --session-list run-scripts/SAM041_tiling120um_sessions.txt
  %(prog)s combine --animal SAM041 --experiment tiling120um
  %(prog)s align --animal SAM041 --experiment tiling120um
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze a single session'
    )
    analyze_parser.add_argument(
        '--mouse', '-m',
        required=True,
        help='Mouse name (e.g., SAM041)'
    )
    analyze_parser.add_argument(
        '--date', '-d',
        required=True,
        help='Recording date (e.g., 251128)'
    )
    analyze_parser.add_argument(
        '--experiment', '-e',
        required=True,
        help='Experiment name (e.g., tiling_3_120um)'
    )
    analyze_parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip showing plots'
    )
    analyze_parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving DataFrame to disk'
    )
    analyze_parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=200,
        help='Number of bootstrap iterations (default: 200)'
    )
    analyze_parser.add_argument(
        '--data-dir',
        default=r'D:/data-distance-coding-project/processed-data',
        help='Base directory for processed data'
    )

    # batch command
    batch_parser = subparsers.add_parser(
        'batch',
        help='Batch analyze multiple sessions from a list file'
    )
    batch_parser.add_argument(
        '--session-list', '-s',
        required=True,
        help='Path to session list file'
    )
    batch_parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=200,
        help='Number of bootstrap iterations (default: 200)'
    )
    batch_parser.add_argument(
        '--data-dir',
        default=r'D:/data-distance-coding-project/processed-data',
        help='Base directory for processed data'
    )

    # combine command
    combine_parser = subparsers.add_parser(
        'combine',
        help='Combine all sessions for an animal'
    )
    combine_parser.add_argument(
        '--animal', '-a',
        required=True,
        help='Animal name (e.g., SAM041)'
    )
    combine_parser.add_argument(
        '--experiment', '-e',
        required=True,
        help='Experiment type (e.g., tiling120um)'
    )
    combine_parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Also export as CSV'
    )

    # align command
    align_parser = subparsers.add_parser(
        'align',
        help='Interactive widefield alignment'
    )
    align_parser.add_argument(
        '--animal', '-a',
        required=True,
        help='Animal name (e.g., SAM041)'
    )
    align_parser.add_argument(
        '--experiment', '-e',
        required=True,
        help='Experiment type (e.g., tiling120um)'
    )

    # visualize command
    viz_parser = subparsers.add_parser(
        'visualize',
        help='Visualize a single cell'
    )
    viz_parser.add_argument(
        '--mouse', '-m',
        required=True,
        help='Mouse name (e.g., SAM041)'
    )
    viz_parser.add_argument(
        '--date', '-d',
        required=True,
        help='Recording date (e.g., 251128)'
    )
    viz_parser.add_argument(
        '--experiment', '-e',
        required=True,
        help='Experiment name (e.g., tiling_3_120um)'
    )
    viz_parser.add_argument(
        '--cell', '-c',
        type=int,
        required=True,
        help='Cell ID to visualize'
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Import workflows here to avoid slow imports when just showing help
    from src.workflows import (
        analyze_session,
        batch_analyze_sessions,
        combine_animal_sessions,
        batch_combine_and_export,
        align_to_widefield,
        visualize_cell
    )

    if args.command == 'analyze':
        print(f"Analyzing session: {args.date}_{args.mouse}_{args.experiment}")
        df, device = analyze_session(
            mouse_name=args.mouse,
            record_date=args.date,
            experiment_name=args.experiment,
            base_data_dir=args.data_dir,
            show_plots=not args.no_plots,
            save_df=not args.no_save,
            n_iterations=args.iterations
        )
        print(f"\nAnalysis complete: {len(df)} cells")

    elif args.command == 'batch':
        print(f"Batch analyzing sessions from: {args.session_list}")
        dfs = batch_analyze_sessions(
            session_list_path=args.session_list,
            base_data_dir=args.data_dir,
            n_iterations=args.iterations
        )
        total_cells = sum(len(df) for df in dfs)
        print(f"\nBatch complete: {len(dfs)} sessions, {total_cells} total cells")

    elif args.command == 'combine':
        print(f"Combining sessions for: {args.animal}_{args.experiment}")
        if args.export_csv:
            df = batch_combine_and_export(
                animal_name=args.animal,
                experiment_type=args.experiment,
                export_csv=True
            )
        else:
            df = combine_animal_sessions(
                animal_name=args.animal,
                experiment_type=args.experiment
            )
        print(f"\nCombined: {len(df)} total cells")

    elif args.command == 'align':
        print(f"Starting widefield alignment for: {args.animal}_{args.experiment}")
        params = align_to_widefield(
            animal_name=args.animal,
            experiment_type=args.experiment
        )
        print(f"\nAlignment complete: {len(params['wf_transformations'])} sessions aligned")

    elif args.command == 'visualize':
        print(f"Visualizing cell {args.cell} from {args.date}_{args.mouse}_{args.experiment}")
        df, device = analyze_session(
            mouse_name=args.mouse,
            record_date=args.date,
            experiment_name=args.experiment,
            show_plots=False,
            save_df=False
        )
        visualize_cell(device, cell_id=args.cell)


if __name__ == '__main__':
    main()
