import os
import argparse  # 인자 처리를 위해 추가
import numpy as np
import torch
from rocket import Rocket
from policy import ActorCritic
import matplotlib.pyplot as plt
import utils
import glob
import cv2
import imageio 
import matplotlib
from IPython.display import HTML, display
import base64
import io

# [Headless 설정] Colab 등 모니터가 없는 환경에서 Qt 플러그인 에러 방지
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# [GUI 무력화] cv2.imshow 호출 시 에러가 나지 않도록 빈 함수로 대체
cv2.imshow = lambda *args: None

# [Matplotlib 설정] GUI 백엔드 대신 Agg(이미지 생성용) 백엔드 사용
matplotlib.use('Agg')   

# --- 1. GIF 재생 및 저장 헬퍼 함수 ---
def show_video(file_path):
    """저장된 GIF 파일을 읽어 Colab/Jupyter 화면에 출력"""
    if not os.path.exists(file_path):
        print("파일을 찾을 수 없습니다.")
        return

    with open(file_path, 'rb') as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    # Colab에서 바로 보이도록 HTML img 태그 사용
    display(HTML(f'<img src="data:image/gif;base64,{encoded}" width="640" />'))

def save_video(frames, path, fps=30):
    """프레임 리스트를 GIF 파일로 저장 (무한 반복 loop=0)"""
    imageio.mimsave(path, frames, fps=fps, loop=0)

# GPU 사용 가능 시 CUDA, 아니면 CPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':

    # --- 2. 인자(Argument) 설정 ---
    parser = argparse.ArgumentParser(description="Rocket Recycling RL Training (PPO/A2C)")
    
    parser.add_argument('--task', type=str, default='landing', choices=['hover', 'landing'],
                        help="학습 목표 설정: 'hover'(호버링) 또는 'landing'(착륙). (기본값: landing)")
    
    parser.add_argument('--max_m_episode', type=int, default=800000,
                        help="총 학습 에피소드 수. (기본값: 800000)")
    
    parser.add_argument('--max_steps', type=int, default=800,
                        help="한 에피소드 당 최대 스텝 수. (기본값: 800)")
    
    parser.add_argument('--video_interval', type=int, default=8000,
                        help="GIF 저장 및 시각화 주기(에피소드 단위). (기본값: 50)")

    args = parser.parse_args()

    # 인자 값 변수 할당
    task = args.task
    max_m_episode = args.max_m_episode
    max_steps = args.max_steps
    video_interval = args.video_interval

    print(f"🚀 Training Start! Task: {task}, Device: {device}")
    print(f"⚙️  Settings: Episodes={max_m_episode}, MaxSteps={max_steps}, VideoInterval={video_interval}")

    # --- 3. 환경 및 모델 초기화 ---
    # 로켓 환경 생성
    env = Rocket(task=task, max_steps=max_steps)

    # 체크포인트 저장 폴더 생성
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    last_episode_id = 0
    REWARDS = []

    # Actor-Critic 네트워크 초기화
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)

    # --- 4. 체크포인트 로드 (이어서 학습하기) ---
    ckpt_list = glob.glob(os.path.join(ckpt_folder, '*.pt'))
    if len(ckpt_list) > 0:
        ckpt_list.sort()
        ckpt_path = ckpt_list[-1]
        print(f"🔄 Loading checkpoint: {ckpt_path}")

        # weights_only=False는 구버전 PyTorch 파일 로드 호환성을 위함
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_episode_id = int(checkpoint['episode_id'])
        REWARDS = list(map(float, checkpoint['REWARDS']))

    # --- 5. 메인 학습 루프 ---
    for episode_id in range(last_episode_id, max_m_episode):

        # 에피소드 시작: 상태 초기화
        state = env.reset()

        rewards, log_probs, values, masks = [], [], [], []
        
        # 시각화(GIF) 저장 여부 확인
        is_video_episode = (episode_id % video_interval == 0)
        frames = [] 

        for step_id in range(max_steps):

            # 행동 결정 (Action)
            action, log_prob, value = net.get_action(state)
            
            # 환경에 행동 적용 (Step)
            state, reward, done, _ = env.step(action)

            # 데이터 수집
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            masks.append(1 - done)

            # --- 이미지 캡처 (GIF 생성용) ---
            if is_video_episode:
                render_result = env.render()

                # 튜플 형태로 반환될 경우 이미지(첫번째 요소)만 추출
                if isinstance(render_result, (tuple, list)):
                    img = render_result[0]
                else:
                    img = render_result

                # 유효한 이미지 데이터인지 확인 및 전처리
                if isinstance(img, np.ndarray):
                    # Float(0~1) 타입을 Uint8(0~255)로 변환
                    if img.dtype != np.uint8:
                        if img.max() <= 1.5:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    
                    # 흑백(2D)일 경우 RGB(3D)로 차원 확장
                    if len(img.shape) == 2:
                         img = np.stack((img,)*3, axis=-1)
                         
                    frames.append(img)

            # 에피소드 종료 조건 (추락, 착륙, 또는 최대 스텝 도달)
            if done or step_id == max_steps - 1:
                _, _, Qval = net.get_action(state)
                
                # 정책 업데이트 (Policy Update)
                net.update_ac(
                    net, rewards, log_probs, values, masks, Qval, gamma=0.999
                )
                break

        # 에피소드별 총 보상 기록
        episode_reward = float(np.sum(rewards))
        REWARDS.append(episode_reward)

        # 진행 상황 로그 (10 에피소드 마다)
        if episode_id % 10 == 0:
            print(f"episode id: {episode_id}, episode reward: {episode_reward:.3f}")

        # --- 6. GIF 저장 및 출력 ---
        if is_video_episode and len(frames) > 0:
            gif_filename = os.path.join(ckpt_folder, f"video_{episode_id:08d}.gif")
            print(f"🎥 Saving GIF to {gif_filename}...")
            
            save_video(frames, gif_filename, fps=30)
            
            print("--- Current Training GIF ---")
            show_video(gif_filename)


        # --- 7. 결과 그래프 및 모델 저장 ---
        if episode_id % args.video_interval == 0: 

            # 보상 그래프 저장
            plt.figure()
            plt.plot(REWARDS)
            plt.plot(utils.moving_avg(REWARDS, N=50))
            plt.legend(['episode reward', 'moving avg'], loc=2)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, f"rewards_{episode_id:08d}.jpg"))
            plt.close()

            # 모델 가중치 저장
            save_path = os.path.join(ckpt_folder, f"ckpt_{episode_id:08d}.pt")
            torch.save(
                {
                    'episode_id': int(episode_id),
                    'REWARDS': [float(r) for r in REWARDS],
                    'model_G_state_dict': net.state_dict(),
                },
                save_path
            )
            print(f"💾 Saved checkpoint: {save_path}")