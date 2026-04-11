import numpy as np
import torch

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
