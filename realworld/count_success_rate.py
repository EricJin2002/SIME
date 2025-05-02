import glob
import os

def count_success_rate(dataset):
    demos = glob.glob(f'{dataset}/*')
    print(demos)
    rm_cnt = 0
    not_rm_cnt = 0
    for demo in demos:
        # print(demo)
        assert os.path.exists(os.path.join(demo, 'preference.txt'))
        with open(os.path.join(demo, 'preference.txt'), 'r') as f:
            preference = f.read()
        if preference == 'fail':
            # print('Remove', demo)
            # os.system(f'rm -rf {demo}')
            rm_cnt += 1
        elif preference == 'success':
            not_rm_cnt += 1
        else:
            raise ValueError('Unknown preference')
    # print(rm_cnt, not_rm_cnt)
    # print("Success rate: ", not_rm_cnt / (rm_cnt + not_rm_cnt))
    # print('Done!')
    return not_rm_cnt, rm_cnt

if __name__ == '__main__':
    success = 0
    fail = 0
    for dataset in glob.glob('/data/jinyang/realworld/0222_two_cups/baseline/*'):
        tmp_success, tmp_fail = count_success_rate(dataset)
        success += tmp_success
        fail += tmp_fail
    print(success, fail)
    print("Success rate: ", success / (success + fail))
    success = 0
    fail = 0
    for dataset in glob.glob('/data/jinyang/realworld/0222_two_cups/sime/*'):
        tmp_success, tmp_fail = count_success_rate(dataset)
        success += tmp_success
        fail += tmp_fail
    print(success, fail)
    print("Success rate: ", success / (success + fail))
    