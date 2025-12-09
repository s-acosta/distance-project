#!/usr/bin/env python3
"""
Batch processing script for multiple radial sessions

Usage:
    python run_batch_analysis.py --config sessions.txt
    python run_batch_analysis.py --config sessions.txt --plot-single-cells
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

def parse_session_file(filepath):
    """
    Parse a session configuration file.
    
    Expected format (one session per line):
    mouse_name,record_date,experiment_name
    
    Example:
    SAM041,251120,tiling_4_120um
    SAM042,251121,tiling_5_120um
    SAM043,251122,tiling_6_120um
    
    Lines starting with # are ignored (comments)
    """
    sessions = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                print(f"Warning: Skipping invalid line: {line}")
                continue
            
            mouse, date, experiment = parts
            sessions.append({
                'mouse': mouse,
                'date': date,
                'experiment': experiment
            })
    
    return sessions

def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple radial sessions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all sessions in config file
  python run_batch_analysis.py --config sessions.txt
  
  # With additional options
  python run_batch_analysis.py --config sessions.txt --plot-single-cells --plot-summary-maps --n-iterations 200
  
Config file format (sessions.txt):
  # One session per line: mouse,date,experiment
  SAM041,251120,tiling_4_120um
  SAM042,251121,tiling_5_120um
  SAM043,251122,tiling_6_120um
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to session configuration file')
    parser.add_argument('--response-mode', type=str, default='spikes',
                       choices=['spikes', 'dff'],
                       help='Response mode: spikes or dff (default: spikes)')
    parser.add_argument('--n-iterations', type=int, default=100,
                       help='Number of iterations for bootstrap analysis (default: 100)')
    parser.add_argument('--plot-single-cells', action='store_true',
                       help='Generate plots for individual cells')
    parser.add_argument('--plot-summary-maps', action='store_true',
                       help='Generate summary maps')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue processing remaining sessions if one fails')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Parse sessions
    print("="*60)
    print("BATCH RADIAL SESSION ANALYSIS")
    print("="*60)
    sessions = parse_session_file(args.config)
    print(f"Found {len(sessions)} sessions to process")
    print("="*60)
    
    # Get path to run_radial_analysis.py (should be in same directory)
    script_dir = Path(__file__).parent
    analysis_script = script_dir / 'run_radial_analysis.py'
    
    if not analysis_script.exists():
        print(f"Error: run_radial_analysis.py not found in {script_dir}")
        sys.exit(1)
    
    # Process each session
    results = []
    for i, session in enumerate(sessions, 1):
        print(f"\n{'='*60}")
        print(f"Processing session {i}/{len(sessions)}")
        print(f"Mouse: {session['mouse']}, Date: {session['date']}, Experiment: {session['experiment']}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            'python',
            str(analysis_script),
            '--mouse', session['mouse'],
            '--date', session['date'],
            '--experiment', session['experiment'],
            '--response-mode', args.response_mode,
            '--n-iterations', str(args.n_iterations)
        ]
        
        if args.plot_single_cells:
            cmd.append('--plot-single-cells')
        if args.plot_summary_maps:
            cmd.append('--plot-summary-maps')
        
        # Run the analysis
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(result.stdout)
            results.append({
                'session': session,
                'status': 'SUCCESS'
            })
            print(f"[ok] Session completed successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Session failed with error:")
            print(e.stderr)
            results.append({
                'session': session,
                'status': 'FAILED',
                'error': str(e)
            })
            
            if not args.continue_on_error:
                print("\nStopping batch processing due to error.")
                print("Use --continue-on-error to process remaining sessions.")
                break
            else:
                print("Continuing to next session...")
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PROCESSING SUMMARY")
    print("="*60)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    failed_count = sum(1 for r in results if r['status'] == 'FAILED')
    
    print(f"Total sessions: {len(sessions)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failed_count}")
    
    if failed_count > 0:
        print("\nFailed sessions:")
        for r in results:
            if r['status'] == 'FAILED':
                s = r['session']
                print(f"  - {s['mouse']}, {s['date']}, {s['experiment']}")
    
    print("="*60)

if __name__ == '__main__':
    main()