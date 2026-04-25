import logging
import math
import os
import os.path
import random
import sys
from collections import defaultdict
from os import remove
from os.path import abspath
from re import split
from time import localtime, strftime, time

import numpy as np
import scipy.sparse as sp
import torch


class ModelConf(object):
    def __init__(self, file):
        self.config = {}
        self.read_configuration(file)

    def __getitem__(self, item):
        if not self.contain(item):
            print('parameter ' + item + ' is not found in the configuration file!')
            exit(-1)
        return self.config[item]

    def contain(self, key):
        return key in self.config

    def read_configuration(self, file):
        if not os.path.exists(file):
            print('config file is not found!')
            raise IOError
        with open(file, encoding='utf-8-sig') as f:
            for ind, line in enumerate(f):
                stripped = line.strip()
                if stripped == '' or stripped.startswith('#'):
                    continue
                if stripped != '':
                    try:
                        key, value = stripped.split('=', 1)
                        self.config[key] = value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d' % ind)


class OptionConf(object):
    def __init__(self, content):
        self.line = content.strip().split(' ')
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':
            self.mainOption = True
        elif self.line[0] == 'off':
            self.mainOption = False
        for i, item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and not item[1:].isdigit():
                ind = i + 1
                for j, sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:]) - 1:
                        ind = j + 1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i + 1:i + 1 + ind])
                except IndexError:
                    self.options[item] = 1

    def __getitem__(self, item):
        if not self.contain(item):
            print('parameter ' + item + ' is invalid!')
            exit(-1)
        return self.options[item]

    def keys(self):
        return self.options.keys()

    def is_main_on(self):
        return self.mainOption

    def contain(self, key):
        return key in self.options


class Log(object):
    def __init__(self, module, filename):
        self.logger = logging.getLogger(f'{module}.{filename}')
        self.logger.setLevel(level=logging.INFO)
        self.logger.propagate = False
        os.makedirs('./log/', exist_ok=True)
        handler = logging.FileHandler('./log/' + filename + '.log')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def add(self, text):
        self.logger.info(text)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def next_batch_pairwise(data, batch_size):
    training_data = data.training_data[:]
    random.shuffle(training_data)
    batch_id = 0
    data_size = len(training_data)
    while batch_id < data_size:
        if batch_id + batch_size <= data_size:
            users = [training_data[idx][0] for idx in range(batch_id, batch_size + batch_id)]
            items = [training_data[idx][1] for idx in range(batch_id, batch_size + batch_id)]
            batch_id += batch_size
        else:
            users = [training_data[idx][0] for idx in range(batch_id, data_size)]
            items = [training_data[idx][1] for idx in range(batch_id, data_size)]
            batch_id = data_size
        u_idx, i_idx, j_idx = [], [], []
        item_list = list(data.item.keys())
        for i, user in enumerate(users):
            i_idx.append(data.item[items[i]])
            u_idx.append(data.user[user])
            neg_item = random.choice(item_list)
            while neg_item in data.training_set_u[user]:
                neg_item = random.choice(item_list)
            j_idx.append(data.item[neg_item])
        yield u_idx, i_idx, j_idx


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-8 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)


def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)
    return emb_loss * reg


def find_k_largest(K, candidates):
    candidates = np.asarray(candidates)
    if K <= 0 or candidates.size == 0:
        return [], []
    K = min(K, candidates.shape[0])
    top_ids = np.argpartition(candidates, -K)[-K:]
    top_ids = top_ids[np.argsort(candidates[top_ids])[::-1]]
    return top_ids.tolist(), candidates[top_ids].tolist()


class Metric(object):
    def __init__(self):
        pass

    @staticmethod
    def hits(origin, res):
        hit_count = {}
        for user in origin:
            items = list(origin[user].keys())
            predicted = [item[0] for item in res[user]]
            hit_count[user] = len(set(items).intersection(set(predicted)))
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        total_num = 0
        for user in origin:
            items = list(origin[user].keys())
            total_num += len(items)
        hit_num = 0
        for user in hits:
            hit_num += hits[user]
        return hit_num / total_num

    @staticmethod
    def precision(hits, N):
        prec = sum([hits[user] for user in hits])
        return prec / (len(hits) * N)

    @staticmethod
    def recall(hits, origin):
        recall_list = [hits[user] / len(origin[user]) for user in hits]
        recall = sum(recall_list) / len(recall_list)
        return recall

    @staticmethod
    def F1(prec, recall):
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)
        else:
            return 0

    @staticmethod
    def MAE(res):
        error = 0
        count = 0
        for entry in res:
            error += abs(entry[2] - entry[3])
            count += 1
        if count == 0:
            return error
        return error / count

    @staticmethod
    def RMSE(res):
        error = 0
        count = 0
        for entry in res:
            error += (entry[2] - entry[3]) ** 2
            count += 1
        if count == 0:
            return error
        return math.sqrt(error / count)

    @staticmethod
    def NDCG(origin, res, N):
        sum_NDCG = 0
        for user in res:
            DCG = 0
            IDCG = 0
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:
                    DCG += 1.0 / math.log(n + 2)
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2)
            sum_NDCG += DCG / IDCG
        return sum_NDCG / len(res)


def ranking_evaluation(origin, res, N):
    measure = {}
    if len(origin) != len(res):
        print('The Lengths of test set and predicted set do not match!')
        exit(-1)
    for n in N:
        predicted = {}
        for user in res:
            predicted[user] = res[user][:n]
        hits = Metric.hits(origin, predicted)
        recall = Metric.recall(hits, origin)
        ndcg = Metric.NDCG(origin, predicted, n)
        measure[n] = {
            'Recall': recall,
            'NDCG': ndcg,
        }
    return measure


def format_ranking_evaluation(measure):
    lines = []
    for n in sorted(measure):
        lines.append('Top ' + str(n) + '\n')
        lines.append('Recall:' + str(measure[n]['Recall']) + '\n')
        lines.append('NDCG:' + str(measure[n]['NDCG']) + '\n')
    return lines


class FileIO(object):
    def __init__(self):
        pass

    @staticmethod
    def write_file(dir, file, content, op='w'):
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_path = os.path.join(dir, file)
        with open(file_path, op) as f:
            f.writelines(content)

    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            remove(file_path)

    @staticmethod
    def load_data_set(file, dtype):
        data = []
        if dtype == 'graph':
            with open(file) as f:
                for line in f:
                    items = split(' ', line.strip())
                    user_id = items[0]
                    item_id = items[1]
                    weight = items[2]
                    data.append([user_id, item_id, float(weight)])

        if dtype == 'sequential':
            training_data, test_data = [], []
            with open(file) as f:
                for line in f:
                    items = split(':', line.strip())
                    user_id = items[0]
                    seq = items[1].strip().split()
                    training_data.append(seq[:-1])
                    test_data.append(seq[-1])
                data = (training_data, test_data)
        return data

    @staticmethod
    def load_user_list(file):
        user_list = []
        print('loading user List...')
        with open(file) as f:
            for line in f:
                user_list.append(line.strip().split()[0])
        return user_list

    @staticmethod
    def load_social_data(file):
        social_data = []
        print('loading social data...')
        with open(file) as f:
            for line in f:
                items = split(' ', line.strip())
                user1 = items[0]
                user2 = items[1]
                if len(items) < 3:
                    weight = 1
                else:
                    weight = float(items[2])
                social_data.append([user1, user2, weight])
        return social_data


class Data(object):
    def __init__(self, conf, training, test):
        self.config = conf
        self.training_data = training[:]
        self.test_data = test[:]


class Graph(object):
    def __init__(self):
        pass

    @staticmethod
    def normalize_graph_mat(adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        pass


class Interaction(Data, Graph):
    def __init__(self, conf, training, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, test)

        self.user = {}
        self.item = {}
        self.id2user = {}
        self.id2item = {}
        self.training_set_u = defaultdict(dict)
        self.training_set_i = defaultdict(dict)
        self.test_set = defaultdict(dict)
        self.test_set_item = set()
        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()
        self.norm_adj = self.normalize_graph_mat(self.ui_adj)
        self.interaction_mat = self.__create_sparse_interaction_matrix()

    def __generate_set(self):
        for entry in self.training_data:
            user, item, rating = entry
            if user not in self.user:
                self.user[user] = len(self.user)
                self.id2user[self.user[user]] = user
            if item not in self.item:
                self.item[item] = len(self.item)
                self.id2item[self.item[item]] = item
            self.training_set_u[user][item] = rating
            self.training_set_i[item][user] = rating
        for entry in self.test_data:
            user, item, rating = entry
            if user not in self.user:
                continue
            self.test_set[user][item] = rating
            self.test_set_item.add(item)

    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num
        row_idx = [self.user[pair[0]] for pair in self.training_data]
        col_idx = [self.item[pair[1]] for pair in self.training_data]
        user_np = np.array(row_idx)
        item_np = np.array(col_idx)
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np + self.user_num)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        adj_mat = tmp_adj + tmp_adj.T
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix(
            (ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def __create_sparse_interaction_matrix(self):
        row, col, entries = [], [], []
        for pair in self.training_data:
            row += [self.user[pair[0]]]
            col += [self.item[pair[1]]]
            entries += [1.0]
        interaction_mat = sp.csr_matrix(
            (entries, (row, col)),
            shape=(self.user_num, self.item_num),
            dtype=np.float32,
        )
        return interaction_mat

    def get_user_id(self, u):
        if u in self.user:
            return self.user[u]

    def get_item_id(self, i):
        if i in self.item:
            return self.item[i]

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        if u in self.user and i in self.training_set_u[u]:
            return True
        else:
            return False

    def contain_user(self, u):
        if u in self.user:
            return True
        else:
            return False

    def contain_item(self, i):
        if i in self.item:
            return True
        else:
            return False

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        u = self.id2user[u]
        k, v = self.user_rated(u)
        vec = np.zeros(len(self.item))
        for pair in zip(k, v):
            iid = self.item[pair[0]]
            vec[iid] = pair[1]
        return vec

    def col(self, i):
        i = self.id2item[i]
        k, v = self.item_rated(i)
        vec = np.zeros(len(self.user))
        for pair in zip(k, v):
            uid = self.user[pair[0]]
            vec[uid] = pair[1]
        return vec

    def matrix(self):
        m = np.zeros((len(self.user), len(self.item)))
        for u in self.user:
            k, v = self.user_rated(u)
            vec = np.zeros(len(self.item))
            for pair in zip(k, v):
                iid = self.item[pair[0]]
                vec[iid] = pair[1]
            m[self.user[u]] = vec
        return m


class TorchGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor(X, device=None):
        coo = X.tocoo().astype(np.float32)
        indices = np.vstack((coo.row, coo.col)).astype(np.int64, copy=False)
        values = coo.data.astype(np.float32, copy=False)

        sparse_tensor = torch.sparse_coo_tensor(
            torch.from_numpy(indices).contiguous(),
            torch.from_numpy(values).contiguous(),
            coo.shape,
            dtype=torch.float32,
        ).coalesce()

        if device is not None:
            sparse_tensor = sparse_tensor.to(device)

        return sparse_tensor


class Recommender(object):
    def __init__(self, conf, training_set, test_set, **kwargs):
        self.config = conf
        self.data = Data(self.config, training_set, test_set)
        self.model_name = self.config['model.name']
        self.ranking = OptionConf(self.config['item.ranking'])
        self.emb_size = int(self.config['embbedding.size'])
        self.maxEpoch = int(self.config['num.max.epoch'])
        self.batch_size = int(self.config['batch_size'])
        self.lRate = float(self.config['learnRate'])
        self.reg = float(self.config['reg.lambda'])
        self.seed = int(self.config['seed']) if self.config.contain('seed') else None
        if self.seed is not None:
            seed_everything(self.seed)
        self.output = OptionConf(self.config['output.setup'])
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        self.model_log = Log(self.model_name, self.model_name + ' ' + current_time)
        self.result = []
        self.recOutput = []

    def initializing_log(self):
        self.model_log.add('### model configuration ###')
        for k in self.config.config:
            self.model_log.add(k + '=' + self.config[k])

    def print_model_info(self):
        print('Model:', self.config['model.name'])
        print('Training Set:', abspath(self.config['training.set']))
        print('Test Set:', abspath(self.config['test.set']))
        print('Embedding Dimension:', self.emb_size)
        print('Maximum Epoch:', self.maxEpoch)
        print('Learning Rate:', self.lRate)
        print('Batch Size:', self.batch_size)
        print('Regularization Parameter: reg %.4f' % self.reg)
        if self.seed is not None:
            print('Random Seed:', self.seed)
        parStr = ''
        if self.config.contain(self.config['model.name']):
            args = OptionConf(self.config[self.config['model.name']])
            for key in args.keys():
                parStr += key[1:] + ':' + args[key] + '  '
            print('Specific parameters:', parStr)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def evaluate(self, rec_list):
        pass

    def execute(self):
        self.initializing_log()
        self.print_model_info()
        print('Initializing and building model...')
        self.build()
        print('Training Model...')
        self.train()
        print('Testing...')
        rec_list = self.test()
        print('Evaluating...')
        self.evaluate(rec_list)


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
            r = '\rProgress: [{}{}]{}%'.format('+' * ratenum, ' ' * (50 - ratenum), ratenum * 2)
            sys.stdout.write(r)
            sys.stdout.flush()

        rec_list = {}
        user_count = len(self.data.test_set)
        for i, user in enumerate(self.data.test_set):
            candidates = self.predict(user)
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
