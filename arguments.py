import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Rocket Recycling RL")

    # --- [Group 1] 기본 실행 설정 (Base Settings) ---
    base_group = parser.add_argument_group('Base Settings')
    base_group.add_argument('--mode', type=str, default='SAC', choices=['A2C', 'DQN', 'SAC'], help='Algorithm to use (A2C, DQN, SAC)')
    base_group.add_argument('--task', type=str, default='landing', choices=['hover', 'landing'], help='Task type')
    base_group.add_argument('--max_m_episode', type=int, default=800000, help='Total training episodes')
    base_group.add_argument('--max_steps', type=int, default=800, help='Max steps per episode')
    base_group.add_argument('--save_interval', type=int, default=1000, help='Interval for saving checkpoint/video (Train only)')
    base_group.add_argument('--episode_id', type=int, default=None, help='Specific episode ID to load (Inference only)')

    # --- [Group 2] 공통 하이퍼파라미터 (Common Hyperparams) ---
    common_group = parser.add_argument_group('Common Hyperparameters')
    common_group.add_argument('--lr', type=float, default=None, help='Learning Rate (Default varies by mode)')
    common_group.add_argument('--gamma', type=float, default=None, help='Discount Factor (Default varies by mode)')

    # --- [Group 3] DQN/SAC 전용 파라미터 (DQN/SAC Specific) ---
    sac_dqn_group = parser.add_argument_group('DQN / SAC Specific Parameters')
    sac_dqn_group.add_argument('--tau', type=float, default=None, help='Soft Update Rate (Used in DQN & SAC)')
    
    # --- [Group 4] SAC 전용 파라미터 (SAC Specific) ---
    sac_group = parser.add_argument_group('SAC Specific Parameters')
    sac_group.add_argument('--alpha', type=float, default=None, help='Entropy Coefficient (Used in SAC)')

    args = parser.parse_args()

    # --- Default Values Logic (기본값 자동 설정) ---
    defaults = {
        'A2C': {'lr': 5e-5, 'gamma': 0.99, 'tau': 0.0,   'alpha': 0.0},
        'DQN': {'lr': 1e-3, 'gamma': 0.99, 'tau': 0.005, 'alpha': 0.0},
        'SAC': {'lr': 3e-4, 'gamma': 0.99, 'tau': 0.005, 'alpha': 0.2}
    }

    # 사용자가 값을 입력하지 않았다면(None), 모드별 기본값으로 채움
    mode_defaults = defaults.get(args.mode)
    
    if args.lr is None: args.lr = mode_defaults['lr']
    if args.gamma is None: args.gamma = mode_defaults['gamma']
    if args.tau is None: args.tau = mode_defaults['tau']
    if args.alpha is None: args.alpha = mode_defaults['alpha']

    return args

def get_folder_name(args):
    """
    하이퍼파라미터를 포함한 폴더 이름 생성 함수
    - 각 모드에서 실제로 사용하는 파라미터만 폴더명에 포함시킴
    """
    def fmt(val): 
        return str(val).replace('.', 'p')

    base = f"{args.task}_{args.mode}_ckpt"
    
    # 1. 공통 파라미터
    suffix = f"_lr_{fmt(args.lr)}_gamma_{fmt(args.gamma)}"
    
    # 2. 모드별 추가 파라미터
    if args.mode == 'SAC':
        # SAC는 tau와 alpha 모두 사용
        suffix += f"_tau_{fmt(args.tau)}_alpha_{fmt(args.alpha)}"
    elif args.mode == 'DQN':
        # DQN은 tau만 사용 (alpha는 무시)
        suffix += f"_tau_{fmt(args.tau)}"
    # A2C는 lr, gamma 외에 추가 파라미터 없음
    
    return base + suffix