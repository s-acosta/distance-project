"""
Widefield Alignment Script

Run interactive alignment of two-photon sessions to widefield retinotopy maps.

Usage:
    python run-scripts/run_widefield_alignment.py SAM041 tiling_120um
    python run-scripts/run_widefield_alignment.py SAM041 tiling_120um --redraw-areas
    python run-scripts/run_widefield_alignment.py SAM041 tiling_120um --redraw-area V1
    python run-scripts/run_widefield_alignment.py SAM041 tiling_120um --realign-session 2024-01-15_SAM041_tiling_120um
"""

import sys
import argparse
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project\radial-analysis")
from path_setup import setup_project_path
setup_project_path()

import matplotlib
matplotlib.use('Qt5Agg')

from src.data.WidefieldAligner import WidefieldAligner
from src.viz.WidefieldVisualizer import WidefieldVisualizer

def main():
    parser = argparse.ArgumentParser(description='Align two-photon sessions to widefield maps')
    parser.add_argument('mouse_name', type=str, help='Mouse ID (e.g., SAM041)')
    parser.add_argument('experiment_name', type=str, help='Experiment name (e.g., tiling_120um)')
    parser.add_argument('--redraw-areas', action='store_true',
                        help='Redo all area drawings (keeps session alignments)')
    parser.add_argument('--redraw-area', type=str, metavar='AREA',
                        help='Redo one area (e.g., V1, LM, AL, RL, AM, PM)')
    parser.add_argument('--realign-session', type=str, metavar='SESSION',
                        help='Redo alignment for one session')

    args = parser.parse_args()

    print(f"Loading aligner for {args.mouse_name} - {args.experiment_name}")
    print("-" * 60)

    aligner = WidefieldAligner(args.mouse_name, args.experiment_name)

    # Handle redo operations
    if args.redraw_areas:
        print("\nRedrawing all areas...")
        aligner.redraw_areas()

    elif args.redraw_area:
        print(f"\nRedrawing area: {args.redraw_area}")
        aligner.redraw_area(args.redraw_area)

    elif args.realign_session:
        print(f"\nRealigning session: {args.realign_session}")
        aligner.realign_session(args.realign_session)

    # Plot the result using WidefieldVisualizer
    print("\nPlotting results...")
    visualizer = WidefieldVisualizer(args.mouse_name, args.experiment_name)
    visualizer.plot_widefield_with_areas_and_sessions()

    print("\nDone!")


if __name__ == "__main__":
    main()
