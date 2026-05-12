import numpy as np
from scipy.optimize import linprog
from typing import List, Tuple
from src.utils.payoff_table import PayoffTable


def uniform_sample(payoff_table,  **solver_config):
    n = payoff_table.table.shape[0]
    return np.ones(n)/n 

def naive_sp(payoff_table,  **solver_config):
    n = payoff_table.table.shape[0]
    w = np.zeros(n)
    w[-1]=1.0
    return w

# from payoff_table import PayoffTable
def is_zero_sum_game(payoff_matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    检查双人博弈是否为零和博弈
    
    参数:
        payoff_matrix: 玩家1的收益矩阵
        tolerance: 数值误差容许度
        
    返回:
        bool: 是否为零和博弈
    """

    return np.allclose(payoff_matrix + payoff_matrix.T, 0, atol=tolerance)

def LP_nash(payoff_table, **solver_config):
    """
    使用线性规划求解零和博弈的纳什均衡
    行玩家希望最大化他的最低收益，列玩家希望最小化行玩家的收益
    
    参数:
        payoff_table: 行玩家收益矩阵
        
    返回:
        nash_weights: 纳什均衡策略
    """
    if hasattr(payoff_table, 'table'):
        payoff_matrix = payoff_table.table
    else:
        payoff_matrix = np.array(payoff_table)
    assert is_zero_sum_game(payoff_matrix), "Payoff table is not zero-sum game"
    # Get dimensions
    num_rows, num_cols = payoff_matrix.shape
    
    # Solve for row player's strategy
    # The row player wants to maximize min_j (sum_i x_i * A_ij)
    
    # Variables: x_1, x_2, ..., x_n, v
    # where x_i is probability of row i, and v is the game value
    
    # Objective: maximize v
    c = np.zeros(num_rows + 1)
    c[-1] = -1  # maximize v ⟺ minimize -v
    
    # Constraints:
    # 1. For each column j: sum_i (x_i * A_ij) >= v
    #    This ensures v is the minimum payoff across all columns
    A_ub = np.zeros((num_cols, num_rows + 1))
    for j in range(num_cols):
        A_ub[j, :num_rows] = -payoff_matrix[:, j]  # Negative because we want >=
        A_ub[j, -1] = 1
    b_ub = np.zeros(num_cols)
    
    # 2. Sum of probabilities equals 1
    A_eq = np.zeros((1, num_rows + 1))
    A_eq[0, :num_rows] = 1
    A_eq[0, -1] = 0  # v is not part of the probability distribution
    b_eq = np.array([1.0])
    
    # 3. Probabilities are non-negative
    bounds = [(0, None) for _ in range(num_rows)] + [(None, None)]  # x_i >= 0, v free
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                  bounds=bounds, method='highs', **solver_config)
    
    if not res.success:
        raise ValueError(f"Linear programming failed: {res.message}")
    
    # Extract results
    row_strategy = res.x[:num_rows]
    value = res.x[-1]
    
    # Now solve for column player's strategy
    # The column player wants to minimize max_i (sum_j y_j * A_ij)
    
    # Variables: y_1, y_2, ..., y_m, u
    # where y_j is probability of column j, and u is the game value
    
    # Objective: minimize u
    c = np.zeros(num_cols + 1)
    c[-1] = 1  # minimize u
    
    # Constraints:
    # 1. For each row i: sum_j (y_j * A_ij) <= u
    #    This ensures u is the maximum loss across all rows
    A_ub = np.zeros((num_rows, num_cols + 1))
    for i in range(num_rows):
        A_ub[i, :num_cols] = payoff_matrix[i, :]
        A_ub[i, -1] = -1  # Negative because we want <=
    b_ub = np.zeros(num_rows)
    
    # 2. Sum of probabilities equals 1
    A_eq = np.zeros((1, num_cols + 1))
    A_eq[0, :num_cols] = 1
    A_eq[0, -1] = 0  # u is not part of the probability distribution
    b_eq = np.array([1.0])
    
    # 3. Probabilities are non-negative
    bounds = [(0, None) for _ in range(num_cols)] + [(None, None)]  # y_j >= 0, u free
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                  bounds=bounds, method='highs', **solver_config)
    
    if not res.success:
        raise ValueError(f"Linear programming failed: {res.message}")
    
    # Extract results
    col_strategy = res.x[:num_cols]
    col_value = res.x[-1]
    
    # Sanity check: the values from both LPs should be equal
    if not np.isclose(value, col_value):
        print(f"Warning: Game values don't match exactly: {value} vs {col_value}")
    
    return row_strategy, col_strategy, value

class SamplingFictitiousPlay:
    def __init__(self, base_iterations: int = 500):
        self.base_iterations = base_iterations
        
    def solve(self, payoff_table, init_weights = None) -> Tuple[np.ndarray, List[float]]:
        """
        基于采样的 Fictitious Play 求解器
        
        参数:
            agents: 策略列表（PPO agents）,不需要了
            payoff_table: 收益矩阵
            
        返回:
            策略分布, exploitability历史
        """
        n_strategies = payoff_table.table.shape[0]
        n_iterations = self.base_iterations * n_strategies
        
        # 初始化随机分布

        population_weights = init_weights if init_weights is not None else np.random.uniform(0, 1, (1, n_strategies)) 
        # init_weights_ = np.ones((1, n_strategies))
        # population_weights = init_weights_ / init_weights_.sum(axis=1)[:, None]
 
        # population_weights = [current_weights]
        averages = population_weights
        exploitability_history = []
        
        for _ in range(n_iterations):
            # 1. 计算当前平均策略（简单平均所有历史权重）
            average_weights = np.average(population_weights, axis=0)
            
            # 2. 计算最佳响应（找到收益最大的策略）
            br = self._get_best_response(average_weights, payoff_table)

            # 3. 计算exploitability
            exp = self._compute_exploitability(average_weights, br, payoff_table)
            exploitability_history.append(exp)
            
            # 4. 更新策略池
            averages = np.vstack((averages, average_weights))
            population_weights = np.vstack((population_weights, br))
            
            # 5. 收敛检查（可选）
            if exp < 1e-4:  # 设置收敛阈值
                break
            # # 5. 使用稳定的收敛检查
            # if len(exploitability_history) > 10:
            #     if np.mean(exploitability_history[-10:]) < 1e-4:
            #         print("Converged!")
            #         break

                
        # 返回最终的平均策略和exploitability历史
        final_weights = np.mean(population_weights, axis=0)
        # final_weights = averages[-1]
        return final_weights, exploitability_history
    
    def _get_best_response(self, weights: np.ndarray, payoff_table) -> int:
        """找到对当前平均策略的最佳响应"""
        expected_payoffs = weights @ payoff_table.table
        br = np.zeros_like(expected_payoffs)
        br[np.argmin(expected_payoffs)] = 1
        return br
    
    def _compute_exploitability(self, 
                              average_strategy: np.ndarray, 
                              br_strategy: np.ndarray, 
                              payoff_table) -> float:
        """计算当前解的exploitability"""
        value_br = br_strategy @ payoff_table.table @ average_strategy.T 
        value_avg = average_strategy @ payoff_table.table @ br_strategy.T
        return value_br - value_avg
    

# Fictituous play as a nash equilibrium solver
def fictitious_play(payoff_table, n_iterations=10, **solver_config):
    """
    PSRO中使用的meta solver接口
    """
    solver = SamplingFictitiousPlay(base_iterations=n_iterations)
    nash_weights, _ = solver.solve(payoff_table)
    return nash_weights

def regularized_fictitious_play(payoff_table, 
                                n_iterations: int = 10000, 
                                alpha: float = 2.0, 
                                tol: float = 1e-5):
    """
    双边熵正则化的 Fictitious Play
    
    Args:
        payoff_table: PayoffTable 对象或矩阵
        n_iterations: 最大迭代次数
        alpha: 逆温度系数 (Inverse Temperature). 
               Alpha 越大 -> 越接近纳什均衡 (Pure/Sharp). 
               Alpha 越小 -> 越趋近均匀分布 (Random).
        tol: 收敛阈值 (基于策略变化的 L1 范数)
        
    Returns:
        avg_col_strategy: 列玩家的 Robust Meta-Strategy
    """
    # 提取矩阵
    if hasattr(payoff_table, 'table'):
        A = payoff_table.table
    else:
        A = np.array(payoff_table)
        
    n_rows, n_cols = A.shape
    
    # 1. 初始化双方的平均策略 (历史信念)
    avg_row_strategy = np.zeros(n_rows)
    avg_col_strategy = np.zeros(n_cols)
    
    # 初始动作可以是均匀分布
    curr_row_strategy = np.ones(n_rows) / n_rows
    curr_col_strategy = np.ones(n_cols) / n_cols
    
    for t in range(1, n_iterations + 1):
        # --- 更新历史平均 ---
        # 这是一个在线更新平均值的技巧： new_avg = (1-1/t)*old + 1/t*new
        # 注意：FP 的标准做法是先基于旧平均计算 BR，再更新平均。
        
        # 记录旧策略用于检查收敛
        old_avg_col = avg_col_strategy.copy()

        # Step 1: 行玩家观察列玩家的历史平均，计算正则化最佳响应
        # 行玩家收益 = A @ avg_col
        ev_row = A @ avg_col_strategy 
        
        # 行玩家 Softmax (Maximize Payoff)
        # logits = alpha * ev_row
        row_logits = alpha * ev_row
        # 数值稳定性处理 (减去最大值防止溢出)
        row_probs = np.exp(row_logits - np.max(row_logits))
        curr_row_strategy = row_probs / np.sum(row_probs)
        
        # Step 2: 列玩家观察行玩家的历史平均，计算正则化最佳响应
        # 列玩家收益 (零和博弈) = - (row_avg @ A)
        # 或者直接理解为列玩家要最小化 row @ A
        ev_col = avg_row_strategy @ A
        
        # 列玩家 Softmin (Minimize Payoff -> Maximize Negative Payoff)
        # logits = alpha * (-ev_col)
        col_logits = -alpha * ev_col
        col_probs = np.exp(col_logits - np.max(col_logits))
        curr_col_strategy = col_probs / np.sum(col_probs)
        
        # Step 3: 更新平均策略
        # 初始阶段 avg 还是 0，所以 t=1 时直接赋值
        if t == 1:
            avg_row_strategy = curr_row_strategy
            avg_col_strategy = curr_col_strategy
        else:
            step_size = 1.0 / t # 经典的 FP 步长
            avg_row_strategy = (1 - step_size) * avg_row_strategy + step_size * curr_row_strategy
            avg_col_strategy = (1 - step_size) * avg_col_strategy + step_size * curr_col_strategy

        # --- 收敛性检查 ---
        # 检查平均策略的变化幅度
        diff = np.linalg.norm(avg_col_strategy - old_avg_col, ord=1)
        if t > 100 and diff < tol:
            print(f"Converged at iteration {t}, diff: {diff:.2e}")
            break
            
    return avg_col_strategy

def dpp_selection(payoff_matrix=None):
    np.set_printoptions(precision=3, suppress=True)
    
    # --- 1. 提取特征 & 2. 计算 Sigma (Median Heuristic) --
    n = len(payoff_matrix)

    max_abs_val = np.max(np.abs(payoff_matrix))

    if max_abs_val > 0:
        # 将矩阵缩放到 [-1, 1]
        norm_payoff_matrix = payoff_matrix / max_abs_val
        print(f"归一化处理完成。缩放因子: {max_abs_val:.4f}")
    else:
        norm_payoff_matrix = payoff_matrix
        print("矩阵全为0，跳过归一化")


    features = norm_payoff_matrix.copy()
    
    # 计算所有两两距离的平方
    dists = []
    for i in range(n):
        for j in range(i + 1, n): # 只算上三角，避免重复
            d = np.sum((features[i] - features[j]) ** 2)
            dists.append(d)
    
    # 中位数启发式
    median_dist = np.median(dists)
    # 设定 sigma，使得 exp(-dist / (2*sigma^2)) 在中位数处等于 exp(-1)
    # 即 2 * sigma^2 = median_dist
    two_sigma_sq = median_dist

    # --- 3. 构建核矩阵 L ---
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_sq = np.sum((features[i] - features[j]) ** 2)
            L[i, j] = np.exp(-dist_sq / two_sigma_sq)
    
    # 增加微小抖动保证数值稳定
    L += np.eye(n) * 1e-6

    # --- 4. 特征值分解 & 5. 计算有效秩 ---
    eigvals = np.linalg.eigvalsh(L)
    # eigvalsh 返回的是升序，我们要从大到小看，所以倒序一下
    eigvals = eigvals[::-1]
    
    # 计算每一项的贡献 lambda / (lambda + 1)
    contributions = eigvals / (eigvals + 1)
    k_eff = np.sum(contributions)
    k_final = int(np.round(k_eff))

    # --- 6. 最终选择 (贪心法) ---
    selected_indices = []

    for _ in range(k_final):
        best_idx = -1
        max_log_det = -np.inf
        
        for i in range(n):
            if i in selected_indices: continue
            
            # 试探性加入
            trial = selected_indices + [i]
            sub_L = L[np.ix_(trial, trial)]
            sign, logdet = np.linalg.slogdet(sub_L)
            
            if logdet > max_log_det:
                max_log_det = logdet
                best_idx = i
        
        if best_idx != -1:
            selected_indices.append(best_idx)

    return selected_indices

def range_strategy_by_meta_mass(sigma, payoff_matrix, mass_threshold=0.9):
    """
    sigma: meta-strategy (N,)
    """
    len_ = len(payoff_matrix)
    # 按概率从高到低排序（排除 core）
    order = np.argsort(-sigma)
    order_ = []
    total_mass = 0.0
    for idx in order:
        order_.append(idx)
        total_mass += sigma[idx]
        if total_mass >= mass_threshold:
            break
    return order_

def calculate_mixture_weights(payoff_matrix,  nash_weights, 
                              dpp_indices, gamma=0.1):
    """
    计算混合 Meta-Weights
    
    Args:
        payoff_matrix: 支付矩阵 (n x n)
        nash_weights: Soft FP 对并集算出的对应权重
        dpp_indices: DPP 选出的多样性策略索引
        gamma: 多样性底座的混合比例 (0.0 ~ 1.0)
    
    Returns:
        final_weights: 长度为 n 的全量权重数组
    """
    n = len(payoff_matrix)
    
    # 1. nash_weights 可能已经是归一化的，如果不是需归一化
    pi_nash = nash_weights / np.sum(nash_weights)

    # 2. 构建 DPP 均匀分布向量 (底座)
    pi_dpp = np.zeros(n)
    # 给所有 DPP 选中的策略平均分配权重
    for idx in dpp_indices:
        pi_dpp[idx] = 1.0 / len(dpp_indices)
        
    # 3. 线性混合
    final_weights = (1 - gamma) * pi_nash + gamma * pi_dpp
    
    return final_weights


def dpp_driven_nash(payoff_table,
                    n_iterations=10000,
                    alpha=3,
                    diversity_weight=0.1,
                    mass_threshold=0.99,
                    chunk_size = 10, # 暴力截断部分
                    **solver_config):
    """
    用dpp方法选出最具多样性的集合
    合并fp选出的最强的策略
    然后在子集上求nash
    最后在dpp子集上分配一个微小的多样性权重
    然后暴力截取前chunk_size个策略，并重新归一化权重
    """
    if hasattr(payoff_table, 'table'):
        A = payoff_table.table
    else:
        A = np.array(payoff_table) 
    if len(A) <= 10: # 策略池过小直接返回均匀分布
        return np.ones(len(A)) / len(A) 

    dpp_indices = dpp_selection(A)
    meta = regularized_fictitious_play(payoff_table, n_iterations, alpha)
    nash_indices = range_strategy_by_meta_mass(meta, A, mass_threshold)
    union_indices = list(set(dpp_indices + nash_indices))
    union_indices.sort()
    #  提取子矩阵
    sub_payoff = A[np.ix_(union_indices, union_indices)]
    sub_weights = regularized_fictitious_play(sub_payoff, n_iterations, alpha=alpha)

    final_meta = np.zeros(len(A))
    for idx, weight in zip(union_indices, sub_weights):
        final_meta[idx] = weight

    final_w = calculate_mixture_weights(A, final_meta, dpp_indices, gamma=diversity_weight)
    
    idx_chunk = np.argsort(-final_w)[:chunk_size] 
    chunk_w = np.zeros_like(final_w)
    for i in idx_chunk:
        chunk_w[i] = final_w[i]


    return chunk_w / chunk_w.sum()  #确保归一化

