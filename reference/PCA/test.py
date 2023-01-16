import argparse
import numpy as np

# input param file
parser = argparse.ArgumentParser(description="A tutorial of argparse!")
parser.add_argument("--p", default=None, type=str, help="Your name")
args = parser.parse_args()

p_args = args.p
p = np.genfromtxt("{}".format(p_args), dtype=None, delimiter=",", encoding="ascii")
p_float = np.genfromtxt(
    "{}".format(p_args), dtype=float, delimiter=",", encoding="ascii"
)
p_int = np.genfromtxt("{}".format(p_args), dtype=int, delimiter=",", encoding="ascii")
p_bool = np.genfromtxt("{}".format(p_args), dtype=bool, delimiter=",", encoding="ascii")

(
    flux_folder,
    output_folder,
    filter_folder,
    template_folder,
    template_list_folder,
    filter_list_folder,
) = p[:6]

flux_name, flux_error_name, filter_list, template_list = p[6:10]
zlist = np.loadtxt("{}".format(p_args), skiprows=32, max_rows=1)

print(template_names)
