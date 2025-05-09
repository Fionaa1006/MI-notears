import numpy as np
import pandas as pd
import scipy.linalg as slin
import scipy.optimize as sopt
from npeet import entropy_estimators as ee
from notears import utils
from pgmpy.readwrite import BIFReader
from pgmpy.sampling import BayesianModelSampling
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import mutual_info_score
# 设置中文字体（以微软雅黑为例，根据系统安装的字体调整）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 ['SimHei'] 等其他中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# ================== 数据加载与预处理 ==================
def load_alarm_and_generate_data(file_path, n_samples=2000):
    """加载Alarm数据集并生成模拟数据"""
    reader = BIFReader(file_path)
    model = reader.get_model()

    # 生成模拟数据
    sampler = BayesianModelSampling(model)
    data = sampler.forward_sample(size=n_samples,seed=42)

    # 定义所有可能的离散值映射
    replace_dict = {
        'yes': 1, 'no': 0, 'TRUE': 1, 'FALSE': 0,
        'high': 1, 'low': 0, 'normal': 0, 'abnormal': 1,
        'NORMAL': 0, 'ABNORMAL': 1, 'LOW': 0, 'HIGH': 1,
    }

    # 替换并转换数值类型
    data = data.replace(replace_dict)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

    return data.values.astype(float), model


def load_true_dag(model):
    """生成真实DAG邻接矩阵"""
    nodes = model.nodes()
    node_index = {node: idx for idx, node in enumerate(nodes)}
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    for edge in model.edges():
        i, j = node_index[edge[0]], node_index[edge[1]]
        adj_matrix[i, j] = 1
    return adj_matrix


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
def compute_mi_matrix(X, k='auto', n_jobs=4):
    """并行计算互信息矩阵"""
    if k=='auto':
        k=int(np.sqrt(X.shape[0]))
    n_features = X.shape[1]
    MI = np.zeros((n_features, n_features))

    def _compute_mi(i, j):
        """return ee.mi(X[:, i], X[:, j], k=k) if i < j else 0.0"""
        return mutual_info_score(X[:, i], X[:, j])

    triu_indices = np.triu_indices(n_features, k=1)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_mi)(i, j) for i, j in zip(*triu_indices)
    )
    MI[triu_indices] = results
    MI += MI.T
    MI = (MI - MI.min()) / (MI.max() - MI.min() + 1e-8)
    return MI


def _get_mi_weights(mi_values):
    """动态正则化权重函数"""
    eps = 1e-8
    valid_mi = mi_values[mi_values > eps]
    if len(valid_mi) == 0:  # 全零情况
        return np.ones_like(mi_values)
    q_low = np.quantile(valid_mi, 0.3)
    q_high = np.quantile(valid_mi, 0.7)
    weights = np.ones_like(mi_values)
    # 高MI区域：动态惩罚
    high_mask = mi_values > q_high
    weights[high_mask] = 0.1
    # 中MI区域：Sigmoid过渡
    mid_mask = (mi_values >= q_low) & (mi_values <= q_high)
    # 将MI值归一化到[-5,5]区间实现Sigmoid过渡
    x_normalized = 10 * ((mi_values[mid_mask] - q_low) / (q_high - q_low)) - 5
    weights[mid_mask] = 1 / (1 + np.exp(x_normalized))  # 输出范围(0,1)
    # 将Sigmoid输出映射到[0.1,1.0]区间
    weights[mid_mask] = 0.1 + 0.9 * (1 - weights[mid_mask])
    # 低MI区域（弱关联，最高惩罚）
    low_mask = mi_values < q_low
    weights[low_mask] = 1.0  # 最高惩罚
    return weights
    """
    q_low, q_high = np.quantile(mi_values, [0.3, 0.7])
    weights = np.ones_like(mi_values)
    # 高MI区域：动态惩罚
    high_mask = mi_values > q_high
    weights[high_mask] = 0.1
    # 中MI区域：Sigmoid过渡
    mid_mask = (mi_values >= q_low) & (mi_values <= q_high)
    mid_point = 0.5
    # 低MI区域（弱关联，最高惩罚）
    low_mask = mi_values < q_low
    weights[low_mask] = 1.0  # 最高惩罚
    return weights"""


def notears_linear_mi(X, MI, lambda1, lambda2=0.01,loss_type='l2', max_iter=500, h_tol=1e-10, rho_max=1e+20, w_threshold=0.1):
    """MI加权改进版"""
    MI_flat = MI.flatten()
    # 在函数开头添加
    np.seterr(all='warn', over='raise')  # 捕获溢出错误

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

    n, d = X.shape
    if loss_type == 'l2':
        X = X - np.mean(X, axis=0, keepdims=True)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True,
                                bounds=bnds, options={'maxfun': 1e5, 'maxiter': 1e5})
            w_new, h_new = sol.x, _h(_adj(sol.x))[0]
            if h_new > 0.25 * h:
                rho *= 2
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break

    W_est = _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


# ================== 主程序 ==================
if __name__ == '__main__':
    np.random.seed(123)
    # 加载数据
    X, model = load_alarm_and_generate_data("D:/data/alarm/alarm.bif", n_samples=2000)  # 调整文件路径
    B_true = load_true_dag(model)
    print(f"数据维度: {X.shape}, 真实边数: {int(B_true.sum())}")

    # 预计算MI矩阵
    MI = compute_mi_matrix(X, n_jobs=4)
    # +++ 新增绘图部分：绘制MI权重分段函数 +++
    plt.figure(figsize=(10, 6))
    # 显式指定字体参数
    font_prop = fm.FontProperties(fname='simhei.ttf')  # 使用具体字体文件路径

    # 生成测试数据
    x = np.linspace(0, 1, 1000)
    weights = _get_mi_weights(x)

    # 计算分位数位置
    q_low = np.quantile(x, 0.3)
    q_high = np.quantile(x, 0.7)

    # 绘制主曲线
    plt.plot(x, weights, linewidth=3, color='darkorange', label='动态权重')

    # 添加区间标注
    plt.axvspan(0, q_low, alpha=0.2, color='red', label='低MI区间')
    plt.axvspan(q_low, q_high, alpha=0.2, color='gold', label='中MI区间')
    plt.axvspan(q_high, 1, alpha=0.2, color='limegreen', label='高MI区间')

    # 添加分界线
    plt.axvline(q_low, color='gray', linestyle='--', linewidth=1)
    plt.axvline(q_high, color='gray', linestyle='--', linewidth=1)

    # 标注典型值
    plt.scatter(q_low, 1.0, color='red', zorder=5)
    plt.scatter((q_low + q_high) / 2, 0.55, color='darkorange', zorder=5)
    plt.scatter(q_high, 0.1, color='green', zorder=5)

    plt.title("MI阈值与正则化权重的分段函数", fontsize=14)
    plt.xlabel("归一化后的互信息值",fontsize=12)
    plt.ylabel("L1惩罚系数权重", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

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
    # 参数搜索
    best_shd, best_W, best_params = float('inf'), None, {}
    lambda1_list = [0.02, 0.025, 0.03, 0.035, 0.04]  # 更细粒度的lambda1
    lambda2_list = [0.005, 0.01, 0.015, 0.02]  # 扩展lambda2范围
    # 初始化存储矩阵
    shd_matrix = np.zeros((len(lambda2_list), len(lambda1_list)))
    tpr_matrix = np.zeros_like(shd_matrix)

    for i, lambda1 in enumerate(lambda1_list):
        for j, lambda2 in enumerate(lambda2_list):
            print(f"\n=== lambda1={lambda1}, lambda2={lambda2} ===")

            # 原始算法
            W_original = notears_linear_original(X, lambda1=lambda1)
            acc_original = utils.count_accuracy(B_true, W_original != 0)
            print(f"[原始] SHD: {acc_original['shd']}, TPR: {acc_original['tpr']:.2f}")

            # 改进算法
            W_mi = notears_linear_mi(X, MI, lambda1=lambda1, lambda2=lambda2)
            acc_mi = utils.count_accuracy(B_true, W_mi != 0)
            shd_matrix[j, i] = acc_mi['shd']
            tpr_matrix[j, i] = acc_mi['tpr']
            print(f"[改进] SHD: {acc_mi['shd']}, TPR: {acc_mi['tpr']:.2f}")

            # 记录最佳结果
            if acc_mi['shd'] < best_shd:
                best_shd = acc_mi['shd']
                best_W = W_mi
                best_params = {'lambda1': lambda1, 'lambda2': lambda2}
        """# 生成网格
        lambda1_grid, lambda2_grid = np.meshgrid(lambda1_list, lambda2_list)

        # 绘制SHD和TPR热力图
        plt.figure(figsize=(12, 5))

        # SHD热力图
        plt.subplot(121)
        contour = plt.contourf(lambda1_grid, lambda2_grid, shd_matrix, cmap='viridis', levels=20)
        plt.colorbar(contour, label='SHD')
        plt.xlabel('λ₁')
        plt.ylabel('λ₂')
        plt.title('结构汉明距离 (SHD) 热力图')

        # TPR热力图
        plt.subplot(122)
        contour = plt.contourf(lambda1_grid, lambda2_grid, tpr_matrix, cmap='plasma', levels=20)
        plt.colorbar(contour, label='TPR')
        plt.xlabel('λ₁')
        plt.ylabel('λ₂')
        plt.title('真正例率 (TPR) 热力图')

        plt.tight_layout()
        plt.show()"""
    # ================== Pareto前沿分析 ==================
    # 生成参数网格
    lambda1_grid, lambda2_grid = np.meshgrid(lambda1_list, lambda2_list)

    # 收集所有参数组合的性能指标
    all_points = []
    for i in range(len(lambda1_list)):
        for j in range(len(lambda2_list)):
            all_points.append((lambda1_list[i], lambda2_list[j], shd_matrix[j, i], tpr_matrix[j, i]))

    # 计算Pareto前沿
    pareto_front = []
    for point in all_points:
        dominated = False
        for other in all_points:
            if (other[2] < point[2] and other[3] >= point[3]) or (other[2] <= point[2] and other[3] > point[3]):
                dominated = True
                break
        if not dominated:
            pareto_front.append(point)

    # ================== 可视化 ==================
    plt.figure(figsize=(18, 6))

    # 子图1: SHD热力图 + Pareto前沿
    plt.subplot(131)
    contour = plt.contourf(lambda1_grid, lambda2_grid, shd_matrix, cmap='viridis', levels=20)
    plt.colorbar(contour, label='SHD')
    plt.xlabel('λ1')
    plt.ylabel('λ2')
    plt.title('结构汉明距离 (SHD)')

    # 标记Pareto前沿
    plt.scatter([p[0] for p in pareto_front], [p[1] for p in pareto_front],
                c='red', s=60, edgecolors='white', label='Pareto Front')
    plt.legend()

    # 子图2: TPR热力图 + Pareto前沿
    plt.subplot(132)
    contour = plt.contourf(lambda1_grid, lambda2_grid, tpr_matrix, cmap='plasma', levels=20)
    plt.colorbar(contour, label='TPR')
    plt.xlabel('λ1')
    plt.ylabel('λ2')
    plt.title('真正例率 (TPR)')
    plt.scatter([p[0] for p in pareto_front], [p[1] for p in pareto_front],
                c='red', s=60, edgecolors='white', label='Pareto Front')
    plt.legend()

    # 子图3: SHD-TPR空间分布
    plt.subplot(133)
    plt.scatter([p[2] for p in all_points], [p[3] for p in all_points],
                c='blue', alpha=0.6, label='参数组合')
    plt.scatter([p[2] for p in pareto_front], [p[3] for p in pareto_front],
                c='red', s=80, edgecolors='black', label='Pareto Front')
    plt.xlabel('SHD')
    plt.ylabel('TPR')
    plt.title('SHD-TPR权衡关系')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()
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