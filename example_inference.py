import os
import argparse # 인자 처리를 위해 추가
import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import glob
import cv2
import imageio 
from IPython.display import HTML, display
import base64
import io

# [Headless 설정] Colab 등 모니터가 없는 환경에서 Qt 플러그인 에러 방지
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# [GUI 무력화] cv2.imshow 호출 시 에러가 나지 않도록 빈 함수로 대체
cv2.imshow = lambda *args: None

# --- 1. GIF 저장 및 재생 헬퍼 함수 ---
def show_video(file_path):
    """저장된 GIF 파일을 읽어 Colab/Jupyter 화면에 출력"""
    if not os.path.exists(file_path):
        print("파일을 찾을 수 없습니다.")
        return

    with open(file_path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    display(HTML(f'<img src="data:image/gif;base64,{encoded}" width="640" />'))

def save_video(frames, path, fps=30):
    """프레임 리스트를 GIF 파일로 저장 (무한 반복 loop=0)"""
    imageio.mimsave(path, frames, fps=fps, loop=0)

# GPU 사용 가능 시 CUDA, 아니면 CPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    # --- 2. 인자(Argument) 설정 ---
    parser = argparse.ArgumentParser(description="Rocket Recycling Inference & GIF Generation")
    
    parser.add_argument('--task', type=str, default='landing', choices=['hover', 'landing'],
                        help="인퍼런스 목표 설정: 'hover'(호버링) 또는 'landing'(착륙). (기본값: landing)")
    
    parser.add_argument('--max_steps', type=int, default=800,
                        help="최대 인퍼런스 스텝 수. (기본값: 800)")

    args = parser.parse_args()

    task = args.task
    max_steps = args.max_steps
    
    print(f"🚀 Inference Start! Task: {task}, Device: {device}")

    # --- 3. 체크포인트 및 모델 로드 ---
    
    # 체크포인트 폴더 경로
    ckpt_folder = os.path.join('./', task + '_ckpt')
    
    # 가장 최신 체크포인트 파일(*.pt) 자동 검색
    ckpt_list = glob.glob(os.path.join(ckpt_folder, '*.pt'))
    if not ckpt_list:
        print(f"❌ 오류: '{ckpt_folder}' 폴더에 훈련된 체크포인트 파일이 없습니다.")
        print(f"   먼저 학습(train)을 진행하거나 task 설정을 확인하세요.")
        exit()
        
    ckpt_list.sort()
    ckpt_dir = ckpt_list[-1] # 가장 마지막 파일 선택
    
    print(f"➡️ Loading checkpoint: {ckpt_dir}")

    # 환경 및 모델 초기화
    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    
    # 가중치 로드 (weights_only=False는 구버전 호환성용)
    checkpoint = torch.load(ckpt_dir, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_G_state_dict'])

    # --- 4. 인퍼런스 루프 (GIF 프레임 수집) ---
    
    state = env.reset()
    frames = [] 
    step_count = 0
    
    print("--- 인퍼런스 진행 중... ---")

    for step_id in range(max_steps):
        
        # 행동 결정 (Action)
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        
        # 렌더링 및 이미지 캡처
        render_result = env.render()
        
        # 튜플/리스트 처리 (이미지만 추출)
        if isinstance(render_result, (tuple, list)):
            img = render_result[0]
        else:
            img = render_result
        
        # 이미지 유효성 검사 및 전처리
        if isinstance(img, np.ndarray):
            # Float -> Uint8 변환
            if img.dtype != np.uint8:
                if img.max() <= 1.5:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 흑백 -> RGB 변환
            if len(img.shape) == 2:
                 img = np.stack((img,)*3, axis=-1)
                 
            frames.append(img)
            
        step_count += 1
        
        # 종료 조건 (환경 완료, 추락, 또는 최대 스텝)
        if done or env.already_crash or step_id == max_steps - 1:
            break
            
    # --- 5. 결과 저장 및 출력 ---
    if frames:
        gif_filename = os.path.join(ckpt_folder, "inference_result.gif")
        
        print(f"✅ 인퍼런스 완료. 총 {step_count} 스텝.")
        print(f"💾 GIF 저장 중: {gif_filename}")
        save_video(frames, gif_filename, fps=30)
        
        print("--- 인퍼런스 결과 GIF ---")
        show_video(gif_filename)
    else:
        print("❌ 녹화된 프레임이 없어 GIF를 저장하지 못했습니다.")