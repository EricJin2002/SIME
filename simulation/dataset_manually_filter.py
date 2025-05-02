import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

def dataset_manually_filter(args):
    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    
    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(args.filter_key)])]
    else:
        demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    print("output_path:", output_path)
    f_out = h5py.File(output_path, "w")
    if "data" in f:
        f.copy("data", f_out)
    if "mask" in f:
        f.copy("mask", f_out)

    for ind in tqdm(range(len(demos))):
        # print("Processing episode {}".format(demos[ind]))
        ep = demos[ind]
        image_dir = os.path.join(args.filtered_dir, ep)
        assert os.path.exists(image_dir)
        image_ids = []
        for im in os.listdir(image_dir):
            if im.endswith(".png") and im.startswith("frame_"):
                image_ids.append(int(im.split("_")[1].split(".")[0]))
        
        states = f["data/{}/states".format(ep)][()]
        stepwise_mask = np.ones(len(states)) * -1
        for i in image_ids:
            assert i % 5 == 0
            stepwise_mask[i:i+5] = 1
        f_out["data/{}".format(ep)].create_dataset("manual_filter", data=stepwise_mask)

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--filtered_dir",
        type=str,
        required=True,
        help="path to the filtered image directory",
    )



    args = parser.parse_args()
    dataset_manually_filter(args)
