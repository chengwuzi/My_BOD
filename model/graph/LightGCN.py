import torch
import torch.nn as nn
import core_runtime as rt
import time
# paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR'20


class LightGCN(rt.GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, test_set)
        args = rt.OptionConf(self.config['LightGCN'])
        self.n_layers = int(args['-n_layer'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers)

    def train(self):
        model = self.model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            start_time = time.time()
            for n, batch in enumerate(rt.next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                user_idx = torch.as_tensor(user_idx, device=self.device, dtype=torch.long)
                pos_idx = torch.as_tensor(pos_idx, device=self.device, dtype=torch.long)
                neg_idx = torch.as_tensor(neg_idx, device=self.device, dtype=torch.long)
                model.train()
                rec_user_emb, rec_item_emb = model()
                user_emb = rec_user_emb.index_select(0, user_idx)
                pos_item_emb = rec_item_emb.index_select(0, pos_idx)
                neg_item_emb = rec_item_emb.index_select(0, neg_idx)
                batch_loss = rt.bpr_loss(user_emb, pos_item_emb, neg_item_emb) + rt.l2_reg_loss(self.reg, user_emb, pos_item_emb)
                # Backward and optimize
                optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                optimizer.step()
                if n % 100 == 0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                del user_idx, pos_idx, neg_idx, user_emb, pos_item_emb, neg_item_emb, rec_user_emb, rec_item_emb, batch_loss
                if self.device.type == 'cuda' and (n + 1) % 1000 == 0:
                    torch.cuda.empty_cache()
            model.eval()
            end_time = time.time()
            print("One epoch Running time: %f s" % (end_time - start_time))
            with torch.no_grad():
                self.user_emb, self.item_emb = (emb.detach() for emb in model())
            self.fast_evaluation(epoch)
            if self.device.type == 'cuda':
                self.user_emb = self.user_emb.cpu()
                self.item_emb = self.item_emb.cpu()
                torch.cuda.empty_cache()
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb



    def save(self):
        with torch.no_grad():
            best_user_emb, best_item_emb = self.model.forward()
            self.best_user_emb = best_user_emb.detach().cpu()
            self.best_item_emb = best_item_emb.detach().cpu()

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.detach().cpu().numpy()


class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj
        self.embedding_dict = self._init_model()
        self.register_buffer('sparse_norm_adj', rt.TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj))

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


