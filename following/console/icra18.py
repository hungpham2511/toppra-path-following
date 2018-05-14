import argparse
from .compare_robust_controllable_sets import main as robust_main
from .compare_tracking_expm import main as tracking_main


def main():
    parser = argparse.ArgumentParser(description="A collection of programs in ICRA18 paper.")
    subparsers = parser.add_subparsers()
    # A subparser for the program that computes the robust
    # controllable sets
    parser_robust = subparsers.add_parser(
        'robust_sets',
        description='A simple example showing the controllable sets in different setting.')
    parser_robust.set_defaults(which='robust_sets')
    parser_robust.add_argument('-v', action='store_true', dest='verbose', help='On for verbose')
    parser_robust.add_argument('-s', "--savefig", action='store_true', help='If true save the figure.', default=False)
    # A subparser for the program that computes the robust
    # controllable sets
    parser_tracking = subparsers.add_parser('tracking_exps')
    parser_tracking.set_defaults(which="tracking_exps")
    parser_sim = subparsers.add_parser(
        'simulate',
        description="Run simulation of the controllers on some different scenarios.")
    parser_sim.set_defaults(which="simulate")
    args = parser.parse_args()

    if args.which == "tracking_exps":
        tracking_main()
    if args.which == "robust_sets":
        robust_main(verbose=args.verbose, savefig=args.savefig)
    
