import sys
sys.path.append("..")
import config

import argparse

parser = argparse.ArgumentParser()

# Common
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default="resnet")
parser.add_argument('--augment', type=str, default="none")
parser.add_argument('--dataset', type=str,
                    choices=["cpsc", "pnet"], default="cpsc")
parser.add_argument('--model_id', type=str, default=None)
parser.add_argument('--label', type=str, choices=["3class", "9class"],
                    default="3class")
parser.add_argument('--lead', type=str, choices=["all", "I"], default="I")

# NN
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default="adam")
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--save-every', type=int, default=5)
parser.add_argument('--cw', type=str, default="auto")
parser.add_argument('--clip-value', type=float, default=10.)

parser.add_argument('--kernel', type=int, default=8)
parser.add_argument('--nfilter', type=int, default=64)

args = parser.parse_args()

args.freq = 50

if args.dataset == "cpsc":
    max_duration = config.cpsc_max_duration
elif args.dataset == "pnet":
    max_duration = config.pnet_max_duration
else:
    raise NotImplementedError
args.max_duration = max_duration
args.i_dim = int(args.freq * max_duration)

if args.label == "3class":
    args.o_dim = 3
elif args.label == "9class":
    args.o_dim = 9


# Conv params (fixed for this research)
args.pool = 2
#args.kernel = 8
#args.nfilter = 64

# o_dim
#args.o_dim = 9

print(args)
if __name__ == "__main__":
    print("-"*80)
    print(args)
