import numpy as np

from scipy.special import expit as sigmoid
import igraph as ig
import random




def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

#判断一个邻接矩阵是否表示一个有向无环图（DAG）
def is_dag(W):  #W：邻接矩阵（二维 NumPy 数组）。
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

#生成一个随机的有向无环图（DAG）。
def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    #对矩阵 M 进行随机排列。
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        #生成一个随机排列矩阵 P（单位矩阵的行随机排列）。
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P  #P.T 是 P 的转置。
        # P.T @ M @ P 表示矩阵乘法，即先计算 M @ P，然后计算 P.T @ (M @ P)。

    #将无向图的邻接矩阵转换为有向无环图的邻接矩阵。
    def _random_acyclic_orientation(B_und):
        #np.tril 将矩阵转换为下三角矩阵（k=-1 表示对角线下方的元素保留，上方的元素置为零），从而确保图是有向且无环的。
        return np.tril(_random_permutation(B_und), k=-1)

    #将 igraph 图对象转换为邻接矩阵。
    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)  #获取图的邻接矩阵。将结果转换为 NumPy 数组。

    #生成 Erdos-Renyi 图。
    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)  #生成随机无向图，节点数量d,预期边数30
        B_und = _graph_to_adjmat(G_und)  #将无向图 G_und 转换为邻形状为 [d, d] 的二值邻接矩阵
        B = _random_acyclic_orientation(B_und)  #将无向图的邻接矩阵转换为有向无环图（DAG）的邻接矩阵。
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')

    #对生成的有向无环图进行随机排列，并验证其是否为 DAG。
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm  #二值邻接矩阵 B_perm，表示生成的 DAG 的结构。


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges 权重范围

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)  #初始化一个与 B 同形状的零矩阵 W，用于存储最终的加权邻接矩阵。
    S = np.random.randint(len(w_ranges), size=B.shape)  # 生成一个与 B 同形状的矩阵 S，其中每个元素是一个随机整数，表示选择的权重范围索引。
    #为每条边分配随机权重。遍历每个权重范围 (low, high)
    for i, (low, high) in enumerate(w_ranges):
        # 生成一个与 B 同形状的矩阵 U，其中每个元素是从范围 [low, high) 中均匀随机选择的。
        U = np.random.uniform(low=low, high=high, size=B.shape)
        # B * (S == i) 创建一个掩码矩阵，仅保留选择当前权重范围的边。
        W += B * (S == i) * U  #U 与掩码矩阵相乘，并加到 W 中，从而为选择当前权重范围的边分配权重。
    return W  #W 是一个加权邻接矩阵

#从线性结构方程模型（SEM）中生成数据。
def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones
        噪声的标准差，可以是标量或长度为 d 的数组，默认为全 1。

    Returns:返回一个形状为 [n, d] 的样本矩阵 X，表示生成的数据。
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    #根据指定的噪声类型生成单个变量的数据。
    def _simulate_single_equation(X, w, scale):
        """X: 父节点的数据[n, num of parents], w:父节点的权重 [num of parents], scale:噪声的标准差，x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]  #获取节点数量 d
    #处理 noise_scale 参数。
    #如果 noise_scale 是 None，则默认为全 1。
    if noise_scale is None:
        scale_vec = np.ones(d)
    #如果 noise_scale 是标量，则将其扩展为长度为 d 的数组。
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    #如果 noise_scale 是数组，则检查其长度是否为 d
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale

    #检查输入的加权邻接矩阵 W 是否表示一个有向无环图（DAG）。
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    #如果 n=inf，生成理论上的协方差矩阵（population risk）。
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)  #生成协方差矩阵
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    #根据加权邻接矩阵 W 创建图对象，并进行拓扑排序。
    G = ig.Graph.Weighted_Adjacency(W.tolist())  #将 NumPy 数组 W 转换为 Python 列表格式，因为 igraph 的 Graph.Weighted_Adjacency 方法需要一个列表作为输入。
    ordered_vertices = G.topological_sorting()  #返回图的拓扑排序结果，即一个节点的顺序列表。
    assert len(ordered_vertices) == d  #验证拓扑排序的结果是否正确。
    #生成数据样本矩阵 X
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)  #获取节点 j 的所有父节点。
        #X[:, j]：将生成的值存储到样本矩阵 X 的第 j 列中。
        #_simulate_single_equation：根据指定的噪声类型和父节点的值生成节点 j 的值。
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        #X[:, parents]：提取所有样本中父节点的值，形状为 [n, len(parents)]。
        #W[parents, j]：提取父节点到节点 j 的权重，形状为 [len(parents)]。
        #scale_vec[j]：节点 j 的噪声标准差。
    return X

#从非线性结构方程模型（SEM）中生成数据样本。返回一个形状为 [n, d] 的样本矩阵 X，表示生成的数据。
def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str)非线性模型类型: mlp, mim, gp, gp-add
        noise_scale (np.ndarray): 噪声的标准差，可以是标量或长度为 d 的数组，默认为全 1。scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    #根据指定的非线性模型类型生成单个变量的数据。
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)  #生成一个形状为 [n] 的数组，每个元素是从均值为 0、标准差为 scale 的高斯分布中随机抽取的。
        pa_size = X.shape[1]  #父节点的数据矩阵 X 的列数，表示父节点的数量。
        #如果当前节点没有父节点，直接返回噪声向量 z。
        if pa_size == 0:
            return z
        #使用多层感知机（MLP）模型生成数据。
        if sem_type == 'mlp':
            hidden = 100  #隐藏层的神经元数量。
            #输入层到隐藏层的权重矩阵，形状为 [pa_size, hidden]。权重从均匀分布 [0.5, 2.0] 中随机抽取。
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1  #一半的权重被随机取反，以引入负权重。
            #W2：隐藏层到输出层的权重向量，形状为 [hidden]。
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            ##父节点的值通过输入层到隐藏层的权重矩阵 W1，并应用 Sigmoid 激活函数。将隐藏层的输出通过权重向量 W2，得到最终的输出。加上噪声向量 z
            x = sigmoid(X @ W1) @ W2 + z
        #使用混合非线性模型（MIM）生成数据。
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            #x：通过混合非线性函数生成的当前节点的值。
            #父节点的值通过权重向量 w1，并应用双曲正切函数。通过 w2，应用余弦函数。通过w3，应用正弦函数。加上噪声向量 z。
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        #使用高斯过程回归（GP）模型生成数据。
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            #.flatten()：将生成的样本展平为一维数组
            x = gp.sample_y(X, random_state=None).flatten() + z
        #使用加性高斯过程回归（GP-add）模型生成数据。
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            #对每个父节点单独生成高斯过程的样本，并将它们相加。
            #X[:, i, None]：提取第 i 个父节点的数据，形状为 [n, 1]
            #gp.sample_y(X[:, i, None], random_state=None).flatten()：生成第 i 个父节点的高斯过程样本,将所有父节点的样本相加。
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X

#计算估计图 B_est与真实图B_true之间的各种准确性指标
def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, 元素{0, 1}
        B_est (np.ndarray): [d, d] estimate, 元素{0, 1, -1}, -1 is undirected edge in CPDAG中的无向边

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    #验证输入矩阵的有效性。
    #如果B_est包含-1，则认为它是CPDAG
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')

    #如果B_est不包含-1，则认为它是DAG
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')

    #获取节点数量 d。
    d = B_true.shape[0]
    # linear index of nonzeros获取预测和真实图中非零元素的索引
    pred_und = np.flatnonzero(B_est == -1)  #估计图中无向边的索引。
    pred = np.flatnonzero(B_est == 1)  #估计图中有向边的索引。
    cond = np.flatnonzero(B_true)  #真实图中有向边的索引。
    cond_reversed = np.flatnonzero(B_true.T) #真实图中反向边的索引。
    cond_skeleton = np.concatenate([cond, cond_reversed])  #真实图的骨架（包括有向边和反向边）的索引。

    # true pos  计算真阳性（预测正确的边）。
    true_pos = np.intersect1d(pred, cond, assume_unique=True)  #预测正确的有向边。
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)  #预测正确的无向边（视为骨架中的边）。
    true_pos = np.concatenate([true_pos, true_pos_und])  #合并为最终的真阳性集合。

    # false pos  计算假阳性（预测错误的边）。
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)  #预测错误的有向边。
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)  #预测错误的无向边。
    false_pos = np.concatenate([false_pos, false_pos_und])  #并为最终的假阳性集合。

    # reverse  计算反向边（预测的边方向与真实方向相反）。
    extra = np.setdiff1d(pred, cond, assume_unique=True)  #预测的边中不在真实图中的边。
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True) #预测的边中与真实图反向的边。

    # compute ratio 计算假发现率（FDR）、真正率（TPR）和假正率（FPR）。
    pred_size = len(pred) + len(pred_und)  #预测的边总数（包括有向边和无向边）。
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)  #真实图中不存在的边的数量。
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)  #假发现率 = （反向边 + 假阳性）/ 预测的边总数。
    tpr = float(len(true_pos)) / max(len(cond), 1)   #真正率 = 真阳性 / 真实的边总数。
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)  #假正率 = （反向边 + 假阳性）/ 真实图中不存在的边的数量。

    # structural hamming distance  计算结构汉明距离（SHD）
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))  #预测图的下三角矩阵的非零元素索引。
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))  #真实图的下三角矩阵的非零元素索引。
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)  #预测图中多余的边。
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)  #真实图中缺失的边。
    shd = len(extra_lower) + len(missing_lower) + len(reverse)  #结构汉明距离 = 多余的边 + 缺失的边 + 反向边。

    #返回一个字典，包含所有计算的准确性指标。
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

