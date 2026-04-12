from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from time import strftime, localtime, time
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation, format_ranking_evaluation
import sys


class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, test_set)
        self.bestPerformance = None
        top = self.ranking['-topN'].split(',')
        self.topN = [int(num) for num in top]
        self.max_N = max(self.topN)

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # # print dataset statistics
        print('Training Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.training_size()))
        print('Test Set Size: (user number: %d, item number %d, interaction number: %d)' % (self.data.test_size()))
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        def process_bar(num, total):
            rate = float(num) / total
            ratenum = int(50 * rate)
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum*2)
            sys.stdout.write(r)
            sys.stdout.flush()

        # predict
        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
            # predictedItems = denormalize(predictedItems, self.data.rScale[-1], self.data.rScale[0])
            rated_list, li = self.data.user_rated(user)
            for item in rated_list:
                candidates[self.data.item[item]] = -10e8
            ids, scores = find_k_largest(self.max_N, candidates)
            item_names = [self.data.id2item[iid] for iid in ids]
            rec_list[user] = list(zip(item_names, scores))
            if i % 1000 == 0:
                process_bar(i, user_count)
        process_bar(user_count, user_count)
        print('')
        return rec_list

    def evaluate(self, rec_list):
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        out_dir = self.output['-dir']
        final_performance = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        if self.bestPerformance is None:
            self.bestPerformance = {
                'epoch': self.maxEpoch,
                'metrics': final_performance,
            }
        self.result = format_ranking_evaluation(final_performance)
        self.model_log.add('###Evaluation Results###')
        self.model_log.add(''.join(self.result))
        summary_file = 'experiment_results.txt'
        FileIO.write_file(out_dir, summary_file, self._build_experiment_record(current_time), op='a')
        print('The experiment summary has been appended to ', abspath(out_dir), '.')
        print('The result of %s:\n%s' % (self.model_name, ''.join(self.result)))

    def _build_experiment_record(self, current_time):
        record = []
        record.append('=' * 120 + '\n')
        record.append('Time: ' + current_time + '\n')
        record.append('Model: ' + self.model_name + '\n')
        record.append('Dataset: ' + self.config['dataset.name'] + '\n')
        record.append('Training Set: ' + abspath(self.config['training.set']) + '\n')
        record.append('Test Set: ' + abspath(self.config['test.set']) + '\n')
        record.append('Best Epoch: ' + str(self.bestPerformance['epoch']) + '\n')
        record.append('Best Metrics: ' + self._format_performance_summary(self.bestPerformance['metrics']) + '\n')
        record.append('Configuration:\n')
        for key in sorted(self.config.config.keys()):
            record.append(key + '=' + self.config[key] + '\n')
        record.append('\n')
        return record

    def _metric_sort_key(self, performance):
        metric_key = []
        for top_n in sorted(performance.keys(), reverse=True):
            metric_key.append(performance[top_n]['NDCG'])
            metric_key.append(performance[top_n]['Recall'])
        return tuple(metric_key)

    def _format_performance_summary(self, performance):
        summary = []
        for top_n in sorted(performance.keys()):
            summary.append('Recall@' + str(top_n) + ':' + str(performance[top_n]['Recall']))
            summary.append('NDCG@' + str(top_n) + ':' + str(performance[top_n]['NDCG']))
        return ' | '.join(summary)

    def fast_evaluation(self, epoch):
        print('evaluating the model...')
        rec_list = self.test()
        performance = ranking_evaluation(self.data.test_set, rec_list, self.topN)
        current_key = self._metric_sort_key(performance)
        is_new_best = self.bestPerformance is None or current_key > self._metric_sort_key(self.bestPerformance['metrics'])
        if is_new_best:
            self.bestPerformance = {
                'epoch': epoch + 1,
                'metrics': performance,
            }
            self.save()
        print('-' * 120)
        print('Quick Ranking Performance')
        print('*Current Performance*')
        print('Epoch:', str(epoch + 1) + ',', self._format_performance_summary(performance))
        if is_new_best:
            print('Best Epoch Updated:', str(epoch + 1))
        print('*Best Performance* ')
        print('Epoch:', str(self.bestPerformance['epoch']) + ',', self._format_performance_summary(self.bestPerformance['metrics']))
        print('-' * 120)
        return performance
