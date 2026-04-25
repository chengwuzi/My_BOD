import os

from SELFRec import SELFRec
from core_runtime import ModelConf

if __name__ == '__main__':
    dataset_configs = {
        '1': ('iFashion_UB', './conf/LightGCN_iFashion.conf'),
        '2': ('Youshu', './conf/LightGCN_Youshu.conf'),
        '3': ('NetEase', './conf/LightGCN_NetEase.conf'),
    }

    print('Model: LightGCN')
    print('Available Datasets:')
    print('1. iFashion_UB')
    print('2. Youshu')
    print('3. NetEase')
    print('=' * 80)
    dataset_choice = input('Please enter the dataset number [1]:').strip() or '1'

    import time

    s = time.time()
    if dataset_choice in dataset_configs:
        dataset_name, config_path = dataset_configs[dataset_choice]
        print('Selected Dataset:', dataset_name)
        if not os.path.exists(config_path):
            print('Config file is not found:', config_path)
            exit(-1)
        conf = ModelConf(config_path)
        missing_paths = [
            path for path in (conf['training.set'], conf['test.set'])
            if not os.path.exists(path)
        ]
        if missing_paths:
            print('Dataset files are not ready for', dataset_name)
            for path in missing_paths:
                print('Missing:', path)
            exit(-1)
    else:
        print('Wrong dataset number!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
