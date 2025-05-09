import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from npeet import entropy_estimators as ee
from notears import utils
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import warnings  # 添加警告过滤


# ================== 数据加载与预处理 ==================
def load_sachs_data(file_path):
    """加载Sachs数据集并进行标准化"""
    data = pd.read_csv(file_path, sep='\t')
    # 标准化处理
    data = (data - data.mean()) / data.std()
    return data.values.astype(float)

def load_sachs_true_dag():
    """定义Sachs真实DAG邻接矩阵（已知结构）"""
    nodes = [
        'Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk',
        'Akt', 'PKA', 'PKC', 'P38', 'Jnk'
    ]
    edges = [
        ('Raf', 'Mek'), ('Mek', 'Erk'), ('Plcg', 'PIP2'),
        ('Plcg', 'PIP3'), ('PIP2', 'PKC'), ('PIP3', 'PIP2'),
        ('PIP3', 'Akt'), ('Erk', 'Akt'), ('Akt', 'PKA'),
        ('PKA', 'Raf'), ('PKA', 'Mek'), ('PKA', 'Erk'),
        ('PKC', 'Raf'), ('PKC', 'Mek'), ('PKC', 'PKA'),
        ('PKC', 'P38'), ('PKC', 'Jnk')
    ]
    n = len(nodes)
    adj = np.zeros((n, n))
    node_index = {node: i for i, node in enumerate(nodes)}
    for src, dst in edges:
        adj[node_index[src], node_index[dst]] = 1
    return adj

# ================== 原始NOTEARS算法 ==================
def notears_linear_original(X, lambda1, loss_type='l2', max_iter=300, h_tol=1e-10, rho_max=1e+20, w_threshold=0.1):
    """原始算法实现"""

    def _loss(W):
        M = X @ W
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        G_loss = -1.0 / X.shape[0] * X.T @ R
        return loss, G_loss

    def _h(W):
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + lambda1, -G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new, h_new = sol.x, _h(_adj(sol.x))[0]
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# ================== MI改进版NOTEARS算法 ==================
def compute_mi_matrix(X, k=3, n_jobs=4):
    """使用k近邻估计连续变量的互信息"""
    n_features = X.shape[1]
    MI = np.zeros((n_features, n_features))

    def _compute_mi(i, j):
        return ee.mi(X[:, [i]], X[:, [j]], k=k) if i < j else 0.0

    triu_indices = np.triu_indices(n_features, k=1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_mi)(i, j) for i, j in zip(*triu_indices)
    )
    MI[triu_indices] = results
    MI += MI.T
    MI = (MI - MI.min()) / (MI.max() - MI.min() + 1e-8)
    return MI




def _get_mi_weights(mi_values):
    """动态正则化权重函数（无需修改）"""
    q_low, q_high = np.quantile(mi_values, [0.3, 0.7])
    weights = np.ones_like(mi_values)
    high_mask = mi_values > q_high
    weights[high_mask] = 0.1
    mid_mask = (mi_values >= q_low) & (mi_values <= q_high)
    mid_point = 0.5
    low_mask = mi_values < q_low
    weights[low_mask] = 1.0
    return weights

def notears_linear_mi(X, MI, lambda1, lambda2=0.01,loss_type='l2', max_iter=500, h_tol=1e-10, rho_max=1e+20, w_threshold=0.1):
    """优化数值稳定性和正则化项"""
    np.seterr(all='warn', over='raise')
    n, d = X.shape
    MI_flat = MI.flatten()
    X = X - X.mean(axis=0)  # 确保中心化

    def _loss(W):
        M = X @ W
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        G_loss = -1.0 / X.shape[0] * X.T @ R
        return loss, G_loss

    def _h(W):
        """E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h"""
        # 添加权重裁剪防止溢出
        W_clipped = np.clip(W, -5, 5)  # 限制权重范围
        E = slin.expm(W_clipped * W_clipped)
        h = np.trace(E) - d
        G_h = E.T * W_clipped * 2  # 使用裁剪后的W计算梯度
        return h, G_h

    def _adj(w):
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        w_pos, w_neg = w[:d * d], w[d * d:]
        mi_weights = _get_mi_weights(MI_flat)

        l1_dynamic = lambda1 * np.sum((w_pos + w_neg) * mi_weights)
        l2_dynamic = 0.5 * lambda2 * np.sum(w_pos ** 2 + w_neg ** 2)  # 新增L2项
        obj = loss + 0.5 * rho * h * h + alpha * h + l1_dynamic+l2_dynamic
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_l1 = lambda1 * mi_weights
        g_obj = np.concatenate((G_smooth.flatten() + g_l1, -G_smooth.flatten() + g_l1))
        return obj, g_obj


    """n, d = X.shape
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)"""

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        for __ in range(10):  # 内层循环加速收敛
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True,
                                bounds=bnds, options={'maxiter': 1000})
            W_new = (sol.x[:d * d] - sol.x[d * d:]).reshape((d, d))
            h_new = _h(W_new)[0]
            if h_new < h_tol or rho >= rho_max:
                break
            if h_new > 0.5 * h:
                rho *= 5
            else:
                break
        w_est, h = sol.x, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = W_new
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

def analyze_errors(W_pred, B_true, node_names):
        """分析错误识别的边"""
        FP = np.where((W_pred != 0) & (B_true == 0))  # 假阳性
        FN = np.where((W_pred == 0) & (B_true != 0))  # 假阴性

        print("\n[误报边]")
        for i, j in zip(*FP):
            print(f"{node_names[i]} -> {node_names[j]}")

        print("\n[漏报边]")
        for i, j in zip(*FN):
            print(f"{node_names[i]} -> {node_names[j]}")


# ================== 主程序 ==================
if __name__ == '__main__':
    np.random.seed(123)
    warnings.filterwarnings("ignore", category=UserWarning)  # 过滤特定警告

    # 加载Sachs数据
    X = load_sachs_data("D:/notears-mi/sachs.data.txt/sachs.data.txt")  # 确保文件路径正确
    B_true = load_sachs_true_dag()
    print(f"数据维度: {X.shape}, 真实边数: {int(B_true.sum())}")

    # 预计算MI矩阵（使用k=5近邻）
    MI = compute_mi_matrix(X, k=5, n_jobs=4)

    # 参数搜索
    """best_shd, best_W = float('inf'), None
    for lambda1 in [0.01, 0.02,0.025,0.03,0.035,0.04, 0.045,0.05, 0.055,0.06]:
        print(f"\n=== lambda1={lambda1} ===")

        # 原始算法
        W_original = notears_linear_original(X, lambda1=lambda1)
        acc_original = utils.count_accuracy(B_true, W_original != 0)
        print(f"[原始] SHD: {acc_original['shd']}, TPR: {acc_original['tpr']:.2f}")

        # 改进算法
        lambda2=0.01
        W_mi = notears_linear_mi(X, MI, lambda1=lambda1,lambda2=lambda2)
        acc_mi = utils.count_accuracy(B_true, W_mi != 0)
        print(f"[改进] SHD: {acc_mi['shd']}, TPR: {acc_mi['tpr']:.2f}")

        # 记录最佳结果
        if acc_mi['shd'] < best_shd:
            best_shd, best_W = acc_mi['shd'], W_mi"""
    # 参数搜索范围（扩展参数范围）
    best_shd, best_W, best_params = float('inf'), None, {}
    lambda1_list = [0.005, 0.01, 0.015, 0.02, 0.025,0.03]
    lambda2_list = [ 0.01, 0.015, 0.02, 0.025,0.03,0.04,0.05]

    for lambda1 in lambda1_list:
        for lambda2 in lambda2_list:
            print(f"\n=== lambda1={lambda1}, lambda2={lambda2} ===")

            # 原始算法
            W_original = notears_linear_original(X, lambda1=lambda1)
            W_original_bin = (W_original != 0).astype(int)  # 确保二值化
            acc_original = utils.count_accuracy(B_true, W_original_bin != 0)
            print(f"[原始] SHD: {acc_original['shd']}, TPR: {acc_original['tpr']:.2f},FDR:{acc_original['fdr']:2f},NNZ:{acc_original['nnz']:2f}")

            # 改进算法
            # 改进算法
            try:
                W_mi = notears_linear_mi(X, MI, lambda1=lambda1, lambda2=lambda2,
                                         max_iter=1000, w_threshold=0.05)
                acc_mi = utils.count_accuracy(B_true, W_mi != 0)
                print(f"[改进] SHD: {acc_mi['shd']}, TPR: {acc_mi['tpr']:.2f},FDR:{acc_mi['fdr']:2f},NNZ:{acc_mi['nnz']:2f}")

                # 记录最佳结果
                if acc_mi['shd'] < best_shd:
                    best_shd = acc_mi['shd']
                    best_W = W_mi
                    best_params = {'lambda1': lambda1, 'lambda2': lambda2}
            except Exception as e:
                print(f"参数组合失败: {str(e)}")

    # 在主程序评估后调用
    node_names = ['Raf', 'Mek', 'Plcg', 'PIP2', 'PIP3', 'Erk', 'Akt', 'PKA', 'PKC', 'P38', 'Jnk']
    analyze_errors((W_mi != 0).astype(int), B_true, node_names)
    # 可视化
    plt.figure(figsize=(28, 5))
    plt.subplot(141, title="True DAG").imshow(B_true, cmap='Greens')
    plt.subplot(142, title="MI Matrix").imshow(MI, cmap='viridis')
    plt.subplot(143,title="Original Notears").imshow(W_original,cmap='Purples')
    if best_W is not None:
        plt.subplot(144, title=f"Best (λ1={best_params['lambda1']}, λ2={best_params['lambda2']})").imshow(best_W != 0,
                                                                                                          cmap='Reds')
    plt.tight_layout()
    plt.show()
    """# 绘制参数性能热力图
    lambda1_grid, lambda2_grid = np.meshgrid(lambda1_list, lambda2_list)
    shd_matrix = ...  # 根据记录结果填充

    plt.figure(figsize=(10, 6))
    plt.contourf(lambda1_grid, lambda2_grid, shd_matrix, cmap='viridis')
    plt.colorbar(label='SHD')
    plt.xlabel('lambda1')
    plt.ylabel('lambda2')
    plt.title('Hyperparameter Performance')
    plt.show()"""

    """plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(B_true, cmap='Blues')
    plt.title("True DAG")
    plt.subplot(1, 3, 2)
    plt.imshow(W_original != 0, cmap='Blues')
    plt.title("Original NOTEARS")
    plt.subplot(1, 3, 3)
    plt.imshow(best_W != 0, cmap='Blues')
    plt.title("MI-Weighted NOTEARS")
    plt.show()"""