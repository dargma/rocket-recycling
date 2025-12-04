import os
import argparse
import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic, QNetwork, DiscreteSAC
import arguments 
import glob
import cv2
import imageio
from IPython.display import HTML, display
import base64

# Headless 환경 설정
os.environ["QT_QPA_PLATFORM"] = "offscreen"
cv2.imshow = lambda *args: None

def show_video(file_path):
    if not os.path.exists(file_path): return
    with open(file_path, 'rb') as f: data = f.read()
    encoded = base64.b64encode(data).decode()
    display(HTML(f'<img src="data:image/gif;base64,{encoded}" width="640" />'))

def save_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps, loop=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 1. Arguments 모듈 사용
    args = arguments.get_args()
    folder_name = arguments.get_folder_name(args)
    
    ckpt_folder = os.path.join('./', folder_name)
    if not os.path.exists(ckpt_folder):
        print(f"❌ 폴더를 찾을 수 없습니다: {ckpt_folder}")
        print("   학습할 때 사용했던 하이퍼파라미터(lr, gamma 등)와 동일하게 입력했는지 확인하세요.")
        exit()

    print(f"📂 Checkpoint Folder: {ckpt_folder}")

    # 2. 체크포인트 파일 찾기
    if args.episode_id is not None:
        ckpt_name = f"ckpt_{args.episode_id:08d}.pt"
        ckpt_path = os.path.join(ckpt_folder, ckpt_name)
    else:
        ckpt_list = glob.glob(os.path.join(ckpt_folder, '*.pt'))
        if not ckpt_list:
            print("❌ 폴더 내에 체크포인트 파일(*.pt)이 없습니다.")
            exit()
        ckpt_list.sort()
        ckpt_path = ckpt_list[-1]

    if not os.path.exists(ckpt_path):
        print(f"❌ 체크포인트 파일을 찾을 수 없습니다: {ckpt_path}")
        exit()

    print(f"➡️ Loading Checkpoint: {ckpt_path}")

    # 3. 모델 로드 및 초기화
    env = Rocket(task=args.task, max_steps=args.max_steps)
    
    # [수정] weights_only=False로 에러 방지
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if args.mode == 'A2C':
        net = ActorCritic(env.state_dims, env.action_dims).to(device)
        net.load_state_dict(checkpoint['model_G_state_dict'])
        
    elif args.mode == 'DQN':
        net = QNetwork(env.state_dims, env.action_dims).to(device)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    elif args.mode == 'SAC':
        net = DiscreteSAC(env.state_dims, env.action_dims).to(device)
        net.actor.load_state_dict(checkpoint['actor_state_dict'])

    # 4. 실행 및 저장
    state = env.reset()
    frames = []
    
    print(f"🚀 Start Inference... (Mode: {args.mode})")

    for step_id in range(args.max_steps):
        # Action 선택
        if args.mode == 'DQN':
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = net(state_t).argmax().item()
        elif args.mode == 'A2C':
            action, _, _ = net.get_action(state)
        elif args.mode == 'SAC':
            # 추론 시에는 deterministic=True로 설정하여 최적의 행동만 선택
            action, _, _ = net.get_action(state, deterministic=True)

        state, reward, done, _ = env.step(action)
        
        # [수정] 화염이 그려진 프레임(Index 1) 선택
        res = env.render()
        if isinstance(res, (tuple, list)):
            img = res[1] 
        else:
            img = res
            
        if isinstance(img, np.ndarray):
            # Float -> Uint8 변환
            if img.dtype != np.uint8: 
                img = (img*255).astype(np.uint8) if img.max() <= 1.5 else img.astype(np.uint8)
            # Channel 확장
            if len(img.shape) == 2: 
                img = np.stack((img,)*3, axis=-1)
            frames.append(img)
            
        if done: break
        
    if frames:
        # [수정] 파일명 8자리 포맷팅 적용 (inference_00001000.gif)
        ep_num = args.episode_id if args.episode_id is not None else 999999
        save_name = f"inference_{ep_num:08d}.gif" 
        save_path = os.path.join(ckpt_folder, save_name)
        
        save_video(frames, save_path, fps=30)
        print(f"✅ Saved GIF: {save_path}")
        show_video(save_path)
    else:
        print("❌ No frames captured.")