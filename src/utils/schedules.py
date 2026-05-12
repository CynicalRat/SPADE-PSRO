# src/utils/schedulers.py

def linear_schedule(initial_value: float, min_value: float = 0.0):
    """
    返回一个纯线性衰减的调度器。
    用于PSRO的第一次迭代。
    """
    def _func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)
    return _func

def warmup_linear_schedule(initial_value: float, 
                           min_value: float, 
                           warmup_fraction: float = 0.02, 
                           warmup_start_lr: float = 1e-7):
    """
    返回一个带有Warm-up阶段的线性衰减调度器。
    用于PSRO的后续微调迭代。
    """
    def _func(progress_remaining: float) -> float:
        warmup_end_progress = 1.0 - warmup_fraction
        if progress_remaining > warmup_end_progress:
            # Warm-up 阶段
            warmup_progress = (1.0 - progress_remaining) / warmup_fraction
            current_lr = warmup_start_lr + warmup_progress * (initial_value - warmup_start_lr)
        else:
            # Decay 阶段
            decay_progress = progress_remaining / warmup_end_progress
            current_lr = min_value + decay_progress * (initial_value - min_value)
        return current_lr
    return _func

def create_br_lr_scheduler(psro_iteration: int,
                           initial_lr: float,
                           min_lr: float,
                           warmup_fraction: float = 0.1):
    """
    学习率调度器工厂函数。
    根据PSRO的迭代次数，生成合适的学习率调度器。

    参数:
        psro_iteration (int): 当前PSRO的迭代次数 (从0开始)。
        initial_lr (float): 学习率的峰值/初始值。
        min_lr (float): 学习率的最终值。
        warmup_fraction (float): Warm-up阶段所占的比例。

    返回:
        一个学习率调度函数。
    """
    if psro_iteration == 0:
        # 第一次迭代：从头训练，使用简单的线性衰减，无需Warm-up。
        print(f"PSRO Iteration {psro_iteration}: Creating simple linear decay scheduler.")
        return linear_schedule(initial_lr, min_lr)
    else:
        # 后续迭代：微调现有模型，使用带Warm-up的调度器以稳定训练。
        print(f"PSRO Iteration {psro_iteration}: Creating scheduler with {warmup_fraction*100}% warm-up.")
        return warmup_linear_schedule(initial_lr, min_lr, warmup_fraction)
