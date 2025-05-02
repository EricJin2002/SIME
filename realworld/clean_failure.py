import glob
import os

def clean_failure(dataset):
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
            print('Remove', demo)
            os.system(f'rm -rf {demo}')
            rm_cnt += 1
        elif preference == 'success':
            not_rm_cnt += 1
        else:
            raise ValueError('Unknown preference')
    print(rm_cnt, not_rm_cnt)
    print('Done!')

if __name__ == '__main__':
    for dataset in glob.glob('/data/jinyang/realworld/0222_two_cups/normal/baseline/*'):
        clean_failure(dataset)
    for dataset in glob.glob('/data/jinyang/realworld/0222_two_cups/normal/sime/*'):
        clean_failure(dataset)
    