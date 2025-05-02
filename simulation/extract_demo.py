import argparse
import h5py
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key


def extract_demo(hdf5_path, output_key, interval):
    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    num_demos = len(demos)
    
    mask = []
    for ind in range(num_demos):
        if ind % interval != 0:
            continue
        
        print("Processing episode {}".format(demos[ind]))

        ep = demos[ind]
        mask.append(ep)

    f.close()

    print(mask)

    lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=mask, key_name=output_key)
    
    print("Total number of samples: {}".format(np.sum(lengths)))
    print("Average number of samples {}".format(np.mean(lengths)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="path to hdf5 dataset",
    )
    parser.add_argument(
        "--output_key",
        type=str,
        default="core_20",
        help="key to store the useful data mask",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="interval to extract demos",
    )

    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    extract_demo(args.dataset, args.output_key, args.interval)
