import os
import argparse


def passed_args():
    parser = argparse.ArgumentParser(description='Parse accuracy')
    parser.add_argument('--exp_dir', type=str, help='Path to experiment .out files.')
    parser.add_argument('-s', '--start', type=int, help='Starting job ID')
    parser.add_argument('-e', '--end', type=int, help='Ending job ID')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = passed_args()

    out = os.listdir(args.exp_dir)
    start = args.start
    while start <= args.end:
        fp = os.path.join(args.exp_dir, f'std_{start}.out')
        if not os.path.isfile(fp):
            start += 1
            continue

        with open(fp, 'r') as f:
            lines = f.readlines()
        exp_str = lines[0]
        for i, l in enumerate(lines):
            if 'score' in l:
                break
        print(exp_str.replace('\n', ''))
        for l in lines[i+1:i+4]:
            print(l.replace('\n', ''))
        print()
        start += 1
