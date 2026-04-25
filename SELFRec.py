import importlib

from core_runtime import FileIO


class SELFRec(object):
    def __init__(self, config):
        self.social_data = []
        self.feature_data = []
        self.config = config
        if config['model.type'] == 'sequential':
            self.training_data, self.test_data = FileIO.load_data_set(config['sequence.data'], config['model.type'])
        else:
            self.training_data = FileIO.load_data_set(config['training.set'], config['model.type'])
            self.test_data = FileIO.load_data_set(config['test.set'], config['model.type'])

        self.kwargs = {}
        if config.contain('social.data'):
            social_data = FileIO.load_social_data(self.config['social.data'])
            self.kwargs['social.data'] = social_data
        # if config.contains('feature.data'):
        #     self.social_data = FileIO.loadFeature(config,self.config['feature.data'])
        print('Reading data and preprocessing...')

    def execute(self):
        module_name = 'model.' + self.config['model.type'] + '.' + self.config['model.name']
        module = importlib.import_module(module_name)
        model_cls = getattr(module, self.config['model.name'])
        recommender = model_cls(self.config, self.training_data, self.test_data, **self.kwargs)
        recommender.execute()
