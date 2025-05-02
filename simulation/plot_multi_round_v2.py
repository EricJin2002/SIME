import argparse
import numpy as np
import matplotlib.pyplot as plt

def decode_log(log_path, baselines, ours):
    with open(log_path, "r") as f:
        lines = f.readlines()

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

        if "mean" in line or "std" in line:
            pass
        else:
            if update_baseline:
                if len(baselines) <= round_num:
                    baselines.append([val])
                else:
                    baselines[round_num].append(val)
                print(f"baselines: {baselines}")
            if update_ours:
                if len(ours) <= round_num:
                    ours.append([val])
                else:
                    ours[round_num].append(val)
                print(f"ours: {ours}")

    return baselines, ours

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, nargs="+", required=True)
    parser.add_argument("--task_name", type=str, default="Square-20")
    parser.add_argument("--ylim_low", type=float, default=0.3)
    parser.add_argument("--ylim_high", type=float, default=0.85)
    args = parser.parse_args()
    log_path = args.log_path
    task_name = args.task_name

    baseline, ours = [], []
    for log in log_path:
        baseline, ours = decode_log(log, baseline, ours)
    baseline = np.array(baseline)
    ours = np.array(ours)
    baseline_mean = np.mean(baseline, axis=1)
    baseline_std = np.std(baseline, axis=1)
    ours_mean = np.mean(ours, axis=1)
    ours_std = np.std(ours, axis=1)
    baseline_upper = np.max(baseline, axis=1)
    baseline_lower = np.min(baseline, axis=1)
    ours_upper = np.max(ours, axis=1)
    ours_lower = np.min(ours, axis=1)

    print("baseline_mean: ", baseline_mean)
    print("baseline_upper: ", baseline_upper)
    print("baseline_lower: ", baseline_lower)
    print("baseline_std: ", baseline_std)
    print("ours_mean: ", ours_mean)
    print("ours_upper: ", ours_upper)
    print("ours_lower: ", ours_lower)
    print("ours_std: ", ours_std)
    
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