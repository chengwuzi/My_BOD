from SELFRec import SELFRec
from core_runtime import ModelConf

if __name__ == '__main__':
    available_models = ['LightGCN']

    print('Available Models:')
    print('   '.join(available_models))
    print('=' * 80)
    model = input('Please enter the model you want to run [LightGCN]:').strip() or 'LightGCN'
    import time

    s = time.time()
    if model in available_models:
        conf = ModelConf('./conf/' + model + '.conf')
    else:
        print('Wrong model name!')
        exit(-1)
    rec = SELFRec(conf)
    rec.execute()
    e = time.time()
    print("Running time: %f s" % (e - s))
