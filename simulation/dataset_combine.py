import os
import h5py
import json
import argparse
from easydict import EasyDict

def combine_dataset(cfg):
    """
    Combine multiple datasets into a single dataset.
    """
    # make directory if it doesn't exist
    if not os.path.exists(os.path.dirname(cfg.output)):
        os.makedirs(os.path.dirname(cfg.output))
    # open output file
    with h5py.File(cfg.output, "w") as out_file:
        out_file.create_group("data")
        out_file.create_group("mask")

        last_ind = 0
        for i, input_i in enumerate(cfg.input):
            with h5py.File(input_i.path, "r") as in_file:
                if i==0:
                    out_file["data"].attrs["env_args"] = in_file["data"].attrs["env_args"]
                for demo_key in in_file["data"].keys():
                    ind = int(demo_key[5:])
                    out_file.copy(in_file["data"][demo_key], f"data/demo_{last_ind+ind}")
                for in_out_key_pair in input_i.map:
                    assert len(in_out_key_pair) == 2
                    in_key = in_out_key_pair[0]
                    out_key = in_out_key_pair[1]
                    if in_key is None:
                        new_data = [
                            "demo_{}".format(last_ind+int(demo_key[5:])).encode() 
                            for demo_key in in_file["data"]
                        ]
                    else:
                        new_data = [
                            "demo_{}".format(last_ind+int(demo_key.decode()[5:])).encode() 
                            for demo_key in in_file["mask"][in_key]
                        ]
                    if len(new_data) == 0:
                        continue
                    if out_key in out_file["mask"]:
                        out_file["mask"][out_key].resize(out_file["mask"][out_key].shape[0] + len(new_data), axis=0)
                        out_file["mask"][out_key][-len(new_data):] = new_data
                    else:
                        out_file.create_dataset(
                            f"mask/{out_key}", 
                            data=new_data,
                            maxshape=(None,),
                        )
                last_ind += len(in_file["data"].keys())
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file.",
    )
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg = EasyDict(cfg)

    combine_dataset(cfg)