import argparse
import h5py
import numpy as np

from robomimic.utils.file_utils import create_hdf5_filter_key


def extract_useful_data(hdf5_path, output_key, same_init_state_repeated_times, threshold=0.9):
    # retrieve number of demos
    f = h5py.File(hdf5_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    num_demos = len(demos)
    
    mask = []
    for ind in range(num_demos):
        if ind % same_init_state_repeated_times != 0:
            continue
        
        print("Processing episode {}".format(demos[ind]))

        _ep_list = []
        for offset in range(same_init_state_repeated_times):
            ep = demos[ind+offset]
            dones = f["data/{}/dones".format(ep)][()]
            is_success = np.any(dones) or len(dones) < 400
            _ep_list.append((ep, is_success))
        
        success_rate = np.mean([v[1] for v in _ep_list])
        if success_rate >= threshold:
            continue
        else:
            for ep, is_success in _ep_list:
                if is_success:
                    mask.append(ep)
                    # if success_rate > 0.5:
                    #     break

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
        default="sr_lss_0.9",
        help="key to store the useful data mask",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="threshold for success rate",
    )
    parser.add_argument(
        "--same_init_state_repeated_times",
        type=int,
        default=5,
        help="number of times the same initial state is repeated",
    )
    args = parser.parse_args()

    # seed to make sure results are consistent
    np.random.seed(0)

    extract_useful_data(args.dataset, args.output_key, args.same_init_state_repeated_times, threshold=args.threshold)
