r"""
Run OMSI spike inference on a processed session and write the two sibling
session folders the comparison notebook expects.

OMSI needs ray, which the notebook env does not have, so run this with the
`spikeinf` interpreter:

    "C:\Users\acostamendoza\AppData\Local\anaconda3\envs\spikeinf\python.exe" ^
        radial-analysis\run-scripts\run_omsi_spikes.py ^
        --session 260903_SAM044_distance-temporal

That writes, under the processed-data root:

    260903_SAM044_distance-temporal_newspikes        spks.npy = spike counts
    260903_SAM044_distance-temporal_newspikes_prob   spks.npy = spike probability

Both parse under the pipeline's {record_date}_{mouse}_{experiment} convention,
so analyze_session picks them up with no further work.
"""

import argparse
import os
import sys

# Must precede numpy/OMSI: see the note in utils/omsi.py.
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

sys.path.append(r'C:\Users\acostamendoza\projects\distance-project')

DEFAULT_DATA_DIR = r'D:/data-distance-coding-project/processed-data'
SPIKEINF_PYTHON = (r'C:\Users\acostamendoza\AppData\Local\anaconda3'
                   r'\envs\spikeinf\python.exe')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run OMSI spike inference on a suite2p session.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--session', required=True,
                        help='Session folder name, e.g. 260903_SAM044_distance-temporal')
    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                        help='Processed data root')
    parser.add_argument('--counts-suffix', default='_newspikes',
                        help='Suffix for the spike-count session folder')
    parser.add_argument('--prob-suffix', default='_newspikes_prob',
                        help='Suffix for the spike-probability session folder')
    parser.add_argument('--hz', type=float, default=None,
                        help='Frame rate in Hz. Read from ops.npy when omitted')
    parser.add_argument('--f-corr', type=float, default=0.7,
                        help='Neuropil correction coefficient')
    parser.add_argument('--tau', type=float, default=None,
                        help="Indicator decay in seconds (suite2p ops['tau']). "
                             'Pins the kernel instead of estimating per cell')
    parser.add_argument('--spike-method', default='map',
                        choices=['map', 'last', 'prob'],
                        help='OMSI spike extraction method')
    parser.add_argument('--ray-dir', default=None,
                        help='Ray scratch directory (default C:\\ray_tmp)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rebuild output folders that already exist')
    parser.add_argument('--max-cells', type=int, default=None,
                        help='Debug: run only the first N cells. Output is not '
                             'usable for analysis')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import OMSI  # noqa: F401
    except ImportError:
        sys.exit(
            'OMSI is not importable from this interpreter:\n'
            f'  {sys.executable}\n\n'
            'It is installed in the spikeinf env. Rerun with:\n'
            f'  "{SPIKEINF_PYTHON}" {" ".join(sys.argv)}'
        )

    from utils.omsi import build_omsi_sessions

    src = os.path.join(args.data_dir, args.session)
    if not os.path.isdir(src):
        sys.exit(f'Session directory not found: {src}')

    build_omsi_sessions(
        src,
        counts_dir=src + args.counts_suffix,
        prob_dir=src + args.prob_suffix,
        hz=args.hz,
        f_corr=args.f_corr,
        tau=args.tau,
        spike_method=args.spike_method,
        ray_dir=args.ray_dir,
        overwrite=args.overwrite,
        max_cells=args.max_cells,
    )


if __name__ == '__main__':
    main()
