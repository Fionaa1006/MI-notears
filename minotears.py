import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from scipy.ndimage import gaussian_filter  # 新增导入
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from notears import utils  # 确保utils模块可用
import time


# ================== 原始算法 ==================
def notears_linear_original(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """原始NOTEARS算法，与linear.py完全一致"""

    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = - 1.0 / X.shape[0] * X.T @ R
        else:
            raise ValueError('unknown loss type')
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
        g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
        return obj, g_obj

    n, d = X.shape
    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    if loss_type == 'l2':
        X = X - X.mean(axis=0, keepdims=True)

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
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


# ================== 改进算法（原demo.py的MI加权版本） ==================
def compute_mi_matrix(X, discretize=False):
    n_features = X.shape[1]
    MI = np.zeros((n_features, n_features))
    for i in range(n_features):
        X_temp = np.delete(X, i, axis=1)
        mi = mutual_info_regression(X_temp, X[:, i], n_neighbors=10)
        MI[i, np.arange(n_features) != i] = mi
    MI = (MI + MI.T) / 2
    np.fill_diagonal(MI, 0.0)
    # Modified: 增加矩阵平滑
    MI = gaussian_filter(MI, sigma=0.7)  # 高斯平滑消除噪声

    mi_median = np.median(MI[MI > 0])
    MI = MI / (mi_median + 1e-8)
    return np.log1p(MI)


def notears_linear_mi(X, lambda1, loss_type, max_iter=100, h_tol=1e-8, rho_max=1e+16, w_threshold=0.3):
    """改进版：使用MI动态调整L1正则化"""
    X_original = X.copy()
    MI = compute_mi_matrix(X_original)
    eps = 1e-8
    MI_flat = MI.flatten()

    def _get_mi_weights(mi_values):
        return 1.0 / (1.0 + np.exp(10.0 * (mi_values - 0.5)))

    def _loss(W):
        M = X @ W
        if loss_type == 'l2':
            R = X - M
            loss = 0.5 / X.shape[0] * (R ** 2).sum()
            G_loss = -1.0 / X.shape[0] * X.T @ R
        else:
            raise ValueError('仅支持l2损失')
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
        w_pos, w_neg = w[:d * d], w[d * d:]
        mi_weights = _get_mi_weights(MI_flat)
        l1_dynamic = lambda1 * np.sum((w_pos + w_neg) * mi_weights)
        obj = loss + 0.5 * rho * h * h + alpha * h + l1_dynamic
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_l1 = lambda1 * mi_weights
        g_pos = G_smooth.flatten() + g_l1
        g_neg = -G_smooth.flatten() + g_l1
        g_obj = np.concatenate((g_pos, g_neg))
        return obj, g_obj

    n, d = X.shape
    if loss_type == 'l2':
        X = X - X.mean(axis=0, keepdims=True)

    w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
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
    return W_est, MI


# 新增动态L1+L2算法
def notears_linear_dynamic_l1_l2(X, lambda1, lambda2, loss_type='l2', max_iter=100,
                                 h_tol=1e-8, rho_max=1e+16, w_threshold=0.3,  mi_slope=15.0, l2_slope=8.0):
    """动态L1+L2正则化版本"""

    # 带平滑的MI矩阵计算
    def compute_mi_matrix_smoothed(X):
        MI = compute_mi_matrix(X)
        # 应用高斯平滑
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(MI, sigma=0.7)
    X_original = X.copy()
    MI = compute_mi_matrix_smoothed(X_original)  # 使用平滑后的MI矩阵
    MI_flat = MI.flatten()

    def _get_mi_weights(mi_values):
        return 1.0 / (1.0 + np.exp(mi_slope * (mi_values - 0.6)))  # 调整中点位置

    def _get_l2_weights(mi_values):
        return 1.0 / (1.0 + np.exp(-l2_slope * (mi_values - 0.4)))  # 调整斜率和中点
        # 自适应正则化参数

    def _adapt_reg_params(iter):
        """随迭代次数动态调整正则化强度"""
        decay_factor = 0.95 ** (iter // 10)
        return lambda1 * decay_factor, lambda2 * (1 + 0.1 * (iter // 10))
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
        w_pos, w_neg = w[:d * d], w[d * d:]

        # Modified: 使用当前迭代参数
        curr_l1, curr_l2 = _adapt_reg_params(iter_count[0])

        mi_weights = _get_mi_weights(MI_flat)
        l2_weights = _get_l2_weights(MI_flat)

        l1_dynamic = curr_l1 * np.sum((w_pos + w_neg) * mi_weights)
        l2_dynamic = 0.5 * curr_l2 * np.sum((w_pos ** 2 + w_neg ** 2) * l2_weights)

        obj = loss + 0.5 * rho * h * h + alpha * h + l1_dynamic + l2_dynamic
        G_smooth = G_loss + (rho * h + alpha) * G_h

        g_l1 = curr_l1 * mi_weights
        g_l2 = curr_l2 * l2_weights * (w_pos + w_neg)

        g_pos = G_smooth.flatten() + g_l1 + g_l2
        g_neg = -G_smooth.flatten() + g_l1 + g_l2

        return obj, np.concatenate([g_pos, g_neg])


    n, d = X.shape
    if loss_type == 'l2':
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)  # 标准化数据
    w_est = np.zeros(2 * d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    # Added: 早停机制
    best_w, best_obj = None, np.inf
    patience = 3
    no_improve = 0
    iter_count = [0]  # 用于闭包捕获迭代次数

    for iter in range(max_iter):
        iter_count[0] = iter
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            current_obj = sol.fun

            # 记录最佳参数
            if current_obj < best_obj - 1e-5:
                best_obj = current_obj
                best_w = sol.x.copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break

        w_est, h = w_new, h_new
        alpha += rho * h

        if h <= h_tol or rho >= rho_max or no_improve >= patience:
            break

        # 恢复最佳参数
    W_est = _adj(best_w) if best_w is not None else _adj(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def run_ablation_experiment(config, n_runs=5):
    """改进点：
    1. 增加超参数搜索空间
    2. 动态参数采样
    3. 结果稳定性检查
    """
    # 定义各方法的超参数搜索空间
    param_spaces = {
        'L1': {
            'lambda1': {'type': 'log', 'low': 0.01, 'high': 1.0}
        },
        'DynamicL1': {
            'lambda1': {'type': 'log', 'low': 0.1, 'high': 0.5}
        },
        'DynamicL1L2': {
            'lambda1': {'type': 'log', 'low': 0.1, 'high': 0.5},
            'lambda2': {'type': 'log', 'low': 0.001, 'high': 0.1},
            'mi_slope': {'type': 'linear', 'low': 10, 'high': 20}
        }
    }

    def sample_params(method_name):
        """根据搜索空间采样参数"""
        space = param_spaces[method_name]
        params = {}
        for key, spec in space.items():
            if spec['type'] == 'log':
                # 对数空间均匀采样
                log_low = np.log(spec['low'])
                log_high = np.log(spec['high'])
                val = np.exp(np.random.uniform(log_low, log_high))
            elif spec['type'] == 'linear':
                # 线性空间均匀采样
                val = np.random.uniform(spec['low'], spec['high'])
            params[key] = val
        return params

    results = []

    for _ in range(n_runs):
        try:
            # 生成数据
            B_true = utils.simulate_dag(config['d'], config['s0'], 'ER')
            W_true = utils.simulate_parameter(B_true)
            X = utils.simulate_linear_sem(W_true, config['n'], sem_type='gauss')
            X += np.random.normal(0, 0.3, X.shape)  # 添加噪声

            # 动态生成各方法参数
            methods = {
                'L1': (notears_linear_original, sample_params('L1')),
                'DynamicL1': (notears_linear_mi, sample_params('DynamicL1')),
                'DynamicL1L2': (notears_linear_dynamic_l1_l2, sample_params('DynamicL1L2'))
            }

            run_results = {}
            for name, (func, params) in methods.items():
                for attempt in range(3):  # 重试机制
                    try:
                        start_time = time.time()
                        # 处理不同返回格式
                        if name == 'DynamicL1':
                            W_est, _ = func(X, **params, loss_type='l2', w_threshold=0.2)
                        else:
                            W_est = func(X, **params, loss_type='l2', w_threshold=0.2)
                        runtime = time.time() - start_time
                        break
                    except Exception as e:
                        if attempt == 2: raise
                        print(f"Retrying {name} due to {str(e)}")
                        time.sleep(1)

                # 计算结果指标
                pred_adj = (W_est != 0).astype(int)
                TP = np.sum((pred_adj == 1) & (B_true == 1))
                FP = np.sum((pred_adj == 1) & (B_true == 0))
                FDR = FP/(TP+FP) if (TP+FP) > 0 else 0.0
                acc = utils.count_accuracy(B_true, pred_adj)

                run_results[name] = {
                    'shd': acc['shd'],
                    'tpr': acc['tpr'],
                    'fdr': FDR,
                    'time': runtime,
                    'params': params  # 记录参数用于分析
                }

            results.append(run_results)
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            continue

    # 统计平均结果
    final_results = {}
    for name in param_spaces.keys():
        filtered = [r[name] for r in results if name in r]
        if not filtered: continue
        final_results[name] = {
            'shd': np.mean([x['shd'] for x in filtered]),
            'tpr': np.mean([x['tpr'] for x in filtered]),
            'fdr': np.mean([x['fdr'] for x in filtered]),
            'time': np.mean([x['time'] for x in filtered])
        }
    return final_results


# ================== 结果可视化 ==================
def visualize_results(results, config):
    """新增标准差显示"""
    metrics = ['shd', 'tpr', 'fdr', 'time']
    labels = ['SHD (Lower better)', 'TPR (Higher better)',
              'FDR (Lower better)', 'Runtime (Seconds)']

    # 计算统计量
    stats = {
        name: {
            'mean': np.mean([r[name][m] for r in results]),
            'std': np.std([r[name][m] for r in results])
        }
        for m in metrics for name in ['L1', 'DynamicL1', 'DynamicL1L2']
    }

    plt.figure(figsize=(15, 10))
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.subplot(2, 2, i + 1)
        x = ['L1', 'DynamicL1', 'DynamicL1L2']
        means = [stats[name]['mean'][metric] for name in x]
        stds = [stats[name]['std'][metric] for name in x]

        bars = plt.bar(x, means, yerr=stds, capsize=5,
                       color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}±{stds[bars.index(bar)]:.2f}',
                     ha='center', va='bottom')

        plt.title(f"{config['name']} Dataset: {label}")
    plt.tight_layout()
    plt.show()

# ================== 主程序 ==================
if __name__ == '__main__':
    # 实验配置
    configs = [
        {'name': 'Small', 'd': 20, 's0': 10, 'n': 100},
        {'name': 'Medium', 'd': 50, 's0': 20, 'n': 200},
        {'name': 'Large', 'd': 100, 's0': 40, 'n': 400}
    ]

    # 运行实验
    all_results = {}
    for cfg in configs:
        print(f"\n=== Running {cfg['name']} Dataset ===")
        results = run_ablation_experiment(cfg, n_runs=3)
        all_results[cfg['name']] = results
        visualize_results(results, cfg)

        # 打印数值结果
        print(f"\n{cfg['name']} Dataset Results:")
        for name in ['L1', 'DynamicL1', 'DynamicL1L2']:
            res = results[name]
            print(f"{name}: SHD={res['shd']:.1f}, TPR={res['tpr']:.2f}, FDR={res['fdr']:.2f}, Time={res['time']:.1f}s")