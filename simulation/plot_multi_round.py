import argparse
import numpy as np
import matplotlib.pyplot as plt

def decode_log(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    baseline_mean = []
    baseline_upper = []
    baseline_lower = []
    baseline_std = []
    ours_mean = []
    ours_upper = []
    ours_lower = []
    ours_std = []
    for line in lines:
        if "eval" not in line or "Success rate" not in line:
            continue
        
        print(line.strip())
        round_num = int(line.split("_")[1].split(" ")[0])
        # print("round_num", round_num)
        update_baseline = "sime" not in line
        update_ours = "sime" in line or round_num == 0
        val = float(line.strip().split(" ")[-1])
        print(f"update_baseline: {update_baseline}, update_ours: {update_ours}, val: {val}")
        if "mean" in line:
            if update_baseline:
                if len(baseline_mean) <= round_num:
                    baseline_mean.append(val)
                else:
                    baseline_mean[round_num] = val
            if update_ours:
                if len(ours_mean) <= round_num:
                    ours_mean.append(val)
                else:
                    ours_mean[round_num] = val
        elif "std" in line:
            if update_baseline:
                if len(baseline_std) <= round_num:
                    baseline_std.append(val)
                else:
                    baseline_std[round_num] = val
            if update_ours:
                if len(ours_std) <= round_num:
                    ours_std.append(val)
                else:
                    ours_std[round_num] = val
        else:
            if update_baseline:
                if len(baseline_upper) <= round_num:
                    baseline_upper.append(val)
                else:
                    baseline_upper[round_num] = max(val, baseline_upper[round_num])
            if update_ours:
                if len(ours_upper) <= round_num:
                    ours_upper.append(val)
                else:
                    ours_upper[round_num] = max(val, ours_upper[round_num])
            if update_baseline:
                if len(baseline_lower) <= round_num:
                    baseline_lower.append(val)
                else:
                    baseline_lower[round_num] = min(val, baseline_lower[round_num])
            if update_ours:
                if len(ours_lower) <= round_num:
                    ours_lower.append(val)
                else:
                    ours_lower[round_num] = min(val, ours_lower[round_num])

    print("baseline_mean: ", baseline_mean)
    print("baseline_upper: ", baseline_upper)
    print("baseline_lower: ", baseline_lower)
    print("baseline_std: ", baseline_std)
    print("ours_mean: ", ours_mean)
    print("ours_upper: ", ours_upper)
    print("ours_lower: ", ours_lower)
    print("ours_std: ", ours_std)

    return baseline_mean, baseline_upper, baseline_lower, baseline_std, ours_mean, ours_upper, ours_lower, ours_std

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default="out/square_multi_round/full_run_multi_round.log")
    parser.add_argument("--task_name", type=str, default="Square-20")
    parser.add_argument("--ylim_low", type=float, default=0.35)
    parser.add_argument("--ylim_high", type=float, default=0.85)
    args = parser.parse_args()
    log_path = args.log_path
    task_name = args.task_name

    baseline_mean, baseline_upper, baseline_lower, baseline_std, ours_mean, ours_upper, ours_lower, ours_std = decode_log(log_path)
    x = np.arange(len(baseline_mean))
    plt.plot(x, np.array(baseline_mean), linestyle='--', color='red', label=f"{task_name} (Baseline)")
    plt.plot(x, np.array(ours_mean), linestyle='-', color='red', label=f"{task_name} (Ours)")
    plt.fill_between(x, np.array(baseline_upper), np.array(baseline_lower), color='#fde3e3', alpha=0.3)
    plt.fill_between(x, np.array(ours_upper), np.array(ours_lower), color='#fde3e3', alpha=0.7)

    plt.xlabel("Round", fontsize=20)
    plt.ylabel("Success Rate", fontsize=20)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc="lower right")
    plt.grid(axis='y')

    plt.xlim(0, len(baseline_mean)-1)
    plt.ylim(args.ylim_low, args.ylim_high)
    plt.xticks(np.arange(0, len(baseline_mean), 1))

    plt.savefig(f"Multiround-{task_name}.pdf", format="pdf")
    plt.show()