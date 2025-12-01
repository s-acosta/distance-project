#!/usr/bin/env python3
"""
Terminal script for running radial session analysis
Based on analysis-radial-session notebook

Usage:
    python run_radial_analysis.py --mouse SAM041 --date 251120 --experiment tiling_4_120um
    python run_radial_analysis.py --mouse SAM041 --date 251120 --experiment tiling_4_120um --plot-single-cells
    python run_radial_analysis.py --mouse SAM041 --date 251120 --experiment tiling_4_120um --plot-reliable-map
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for terminal use
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure matplotlib for PDF output
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

# Add project path
sys.path.append(r"C:\Users\acostamendoza\projects\distance-project")
from path_setup import setup_project_path
setup_project_path()

from src.RadialExperiment import RadialDevice, RadialPlots, RadialSummaries

def main():
    parser = argparse.ArgumentParser(
        description='Run radial session analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python run_radial_analysis.py --mouse SAM041 --date 251120 --experiment tiling_4_120um
  
  # With single cell plots
  python run_radial_analysis.py --mouse SAM041 --date 251120 --experiment tiling_4_120um --plot-single-cells
  
  # With reliable cells map
  python run_radial_analysis.py --mouse SAM041 --date 251120 --experiment tiling_4_120um --plot-reliable-map
  
  # Custom number of bootstrap iterations
  python run_radial_analysis.py --mouse SAM041 --date 251120 --experiment tiling_4_120um --n-iterations 500
        """
    )
    parser.add_argument('--mouse', type=str, required=True, help='Mouse name (e.g., SAM041)')
    parser.add_argument('--date', type=str, required=True, help='Recording date YYMMDD (e.g., 251120)')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment name (e.g., tiling_4_120um)')
    parser.add_argument('--response-mode', type=str, default='spikes', 
                       choices=['spikes', 'dff'],
                       help='Response mode: spikes or dff (default: spikes)')
    parser.add_argument('--n-iterations', type=int, default=100,
                       help='Number of iterations for bootstrap analysis (default: 100)')
    parser.add_argument('--plot-single-cells', action='store_true',
                       help='Generate plots for individual cells')
    parser.add_argument('--plot-reliable-map', action='store_true',
                       help='Generate reliable cells map')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RADIAL SESSION ANALYSIS")
    print("="*60)
    print(f"Mouse: {args.mouse}")
    print(f"Date: {args.date}")
    print(f"Experiment: {args.experiment}")
    print(f"Response mode: {args.response_mode}")
    print(f"Bootstrap iterations: {args.n_iterations}")
    print("="*60)
    
    # Initialize RadialDevice (matches notebook usage)
    print("\nInitializing RadialDevice...")
    D = RadialDevice(
        mouse_name=args.mouse, 
        record_date=args.date, 
        experiment_name=args.experiment,
        response_mode=args.response_mode
    )
    
    print(f"Loaded {len(D.suite2p_id)} cells")
    print(f"Data directory: {D.data_dir}")
    
    # Run session analysis (matches notebook: D.analyze_session_stats())
    print("\nRunning session statistics analysis...")
    stats_df = D.analyze_session_stats(n_iterations=args.n_iterations)
    
    # Save statistics
    results_dir = os.path.join(D.data_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    stats_file = os.path.join(results_dir, 'session_stats.csv')
    stats_df.to_csv(stats_file, index=False)
    print(f"Statistics saved to: {stats_file}")
    
    # Initialize plots
    plots = RadialPlots(D)
    
    # Initialize summaries (matches notebook usage)
    print("\nInitializing RadialSummaries...")
    summ = RadialSummaries(radial_class=D, radial_stats=stats_df, radial_plots=plots)
    
    print(f"Found {len(summ.responsive_cells)} responsive cells")
    print(f"Found {sum(len(cells) for cells in summ.modulated_cells.values())} total modulated cells")
    
    # Save summ object as pickle
    summ_file = os.path.join(D.data_dir, 'radial_summaries.pkl')
    import pickle
    with open(summ_file, 'wb') as f:
        pickle.dump(summ, f)
    print(f"RadialSummaries object saved to: {summ_file}")
    
    # Plot single cells if requested (matches notebook: summ.plot_single_cells())
    if args.plot_single_cells:
        print("\nGenerating single cell plots...")
        summ.plot_single_cells()
        print("Single cell plots saved")
    
    # Plot reliable cells map if requested (matches notebook: summ.plot_reliable_cells_map())
    if args.plot_reliable_map:
        print("\nGenerating reliable cells map...")
        fig = summ.plot_reliable_cells_map()
        map_file = os.path.join(results_dir, 'reliable_cells_map.png')
        fig.savefig(map_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Reliable cells map saved to: {map_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Results directory: {results_dir}")

if __name__ == '__main__':
    main()
