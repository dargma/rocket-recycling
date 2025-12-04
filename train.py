import os
import numpy as np
import torch
import torch.nn.functional as F
from rocket import Rocket
from policy import ActorCritic, QNetwork, ReplayBuffer, DiscreteSAC
import arguments 
import matplotlib.pyplot as plt
import utils
import glob
import cv2
import imageio
import random
import matplotlib

os.environ["QT_QPA_PLATFORM"] = "offscreen"
cv2.imshow = lambda *args: None
matplotlib.use('Agg')

def save_video(frames, path, fps=30):
    imageio.mimsave(path, frames, fps=fps, loop=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # 1. Arguments 모듈 사용
    args = arguments.get_args()
    folder_name = arguments.get_folder_name(args)
    
    ckpt_folder = os.path.join('./', folder_name)
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    print(f"🚀 Training Start! [{args.mode}] Task: {args.task}")
    print(f"📂 Save Folder: {ckpt_folder}")
    print(f"⚙️  Params: lr={args.lr}, gamma={args.gamma}, tau={args.tau}, alpha={args.alpha}")

    env = Rocket(task=args.task, max_steps=args.max_steps)
    REWARDS = []
    last_episode_id = 0

    # ==========================================================================
    # [MODE 1] A2C
    # ==========================================================================
    if args.mode == 'A2C':
        # Policy.py 수정본 사용 시 __init__에서 lr 전달 가능
        net = ActorCritic(env.state_dims, env.action_dims, lr=args.lr).to(device)

        ckpt_list = glob.glob(os.path.join(ckpt_folder, '*.pt'))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            # [수정] weights_only=False 추가 (에러 방지)
            checkpoint = torch.load(ckpt_list[-1], map_location=device, weights_only=False)
            net.load_state_dict(checkpoint['model_G_state_dict'])
            last_episode_id = int(checkpoint['episode_id']) + 1
            REWARDS = checkpoint['REWARDS']
            print(f"🔄 A2C Loaded: {ckpt_list[-1]} (Next episode: {last_episode_id})")

        # [수정] max_m_episode 번호까지 실행되도록 +1 (예: 1400이면 1400번 포함)
        target_episode = args.max_m_episode
        print(f"▶️  Running episodes: {last_episode_id} ~ {target_episode}")

        for episode_id in range(last_episode_id, target_episode + 1):
            state = env.reset()
            rewards, log_probs, values, masks = [], [], [], []
            frames = []
            
            # [수정] 저장 조건: 주기적 저장 OR 마지막 목표 에피소드일 때
            do_save = (episode_id % args.save_interval == 0) or (episode_id == target_episode)

            for step_id in range(args.max_steps):
                action, log_prob, value = net.get_action(state)
                state, reward, done, _ = env.step(action)
                rewards.append(reward); log_probs.append(log_prob); values.append(value); masks.append(1 - done)

                if do_save:
                    res = env.render()
                    img = res[1] if isinstance(res, (tuple, list)) else res
                    if isinstance(img, np.ndarray): frames.append(img)

                if done or step_id == args.max_steps - 1:
                    _, _, Qval = net.get_action(state)
                    net.update_ac(net, rewards, log_probs, values, masks, Qval, gamma=args.gamma)
                    break

            ep_reward = np.sum(rewards)
            REWARDS.append(ep_reward)
            if episode_id % 10 == 0: print(f"[{args.mode}] Ep: {episode_id}, Reward: {ep_reward:.3f}")

            if do_save:
                if frames: save_video(frames, os.path.join(ckpt_folder, f"video_{episode_id:08d}.gif"))
                plt.figure(); plt.plot(REWARDS); plt.plot(utils.moving_avg(REWARDS))
                plt.savefig(os.path.join(ckpt_folder, f"rewards_{episode_id:08d}.jpg")); plt.close()
                torch.save({'episode_id': episode_id, 'REWARDS': REWARDS, 'model_G_state_dict': net.state_dict()},
                           os.path.join(ckpt_folder, f"ckpt_{episode_id:08d}.pt"))
                print(f"💾 Saved at episode {episode_id}")

    # ==========================================================================
    # [MODE 2] DQN
    # ==========================================================================
    elif args.mode == 'DQN':
        batch_size = 64; buffer_capacity = 10000
        epsilon = 1.0; epsilon_end = 0.01; epsilon_decay = 0.995
        
        q_net = QNetwork(env.state_dims, env.action_dims, lr=args.lr).to(device)
        target_net = QNetwork(env.state_dims, env.action_dims, lr=args.lr).to(device)
        target_net.load_state_dict(q_net.state_dict()); target_net.eval()
        buffer = ReplayBuffer(buffer_capacity)

        ckpt_list = glob.glob(os.path.join(ckpt_folder, '*.pt'))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            checkpoint = torch.load(ckpt_list[-1], map_location=device, weights_only=False)
            q_net.load_state_dict(checkpoint['model_G_state_dict'])
            target_net.load_state_dict(q_net.state_dict())
            last_episode_id = int(checkpoint['episode_id']) + 1
            REWARDS = checkpoint['REWARDS']
            epsilon = max(epsilon_end, 1.0 * (epsilon_decay ** last_episode_id))
            print(f"🔄 DQN Loaded: {ckpt_list[-1]} (Next episode: {last_episode_id})")

        # [수정] 범위 확장
        target_episode = args.max_m_episode
        print(f"▶️  Running episodes: {last_episode_id} ~ {target_episode}")

        for episode_id in range(last_episode_id, target_episode + 1):
            state = env.reset(); ep_reward = 0; frames = []
            
            # [수정] 저장 조건 변경
            do_save = (episode_id % args.save_interval == 0) or (episode_id == target_episode)

            for step_id in range(args.max_steps):
                if random.random() < epsilon: action = env.get_random_action()
                else:
                    with torch.no_grad(): action = q_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax().item()
                
                next_state, reward, done, _ = env.step(action)
                buffer.push(state, action, reward, next_state, done)
                state = next_state; ep_reward += reward

                if do_save:
                    res = env.render(); img = res[1] if isinstance(res, (tuple, list)) else res
                    if isinstance(img, np.ndarray): frames.append(img)

                if len(buffer) > batch_size:
                    states, actions, rewards_b, next_states, dones = buffer.sample(batch_size)
                    states = torch.FloatTensor(states).to(device)
                    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards_b = torch.FloatTensor(rewards_b).to(device)
                    next_states = torch.FloatTensor(next_states).to(device)
                    dones = torch.FloatTensor(dones).to(device)

                    q_values = q_net(states).gather(1, actions).squeeze()
                    with torch.no_grad():
                        next_q = target_net(next_states).max(1)[0]
                        target_q = rewards_b + (1 - dones) * args.gamma * next_q
                    
                    loss = F.mse_loss(q_values, target_q)
                    q_net.optimizer.zero_grad(); loss.backward(); q_net.optimizer.step()
                    
                    for target_param, param in zip(target_net.parameters(), q_net.parameters()):
                        target_param.data.copy_(target_param.data * (1.0 - args.tau) + param.data * args.tau)
                if done: break
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay); REWARDS.append(ep_reward)
            if episode_id % 10 == 0: print(f"[{args.mode}] Ep: {episode_id}, Reward: {ep_reward:.3f}, Eps: {epsilon:.2f}")

            if do_save:
                if frames: save_video(frames, os.path.join(ckpt_folder, f"video_{episode_id:08d}.gif"))
                plt.figure(); plt.plot(REWARDS); plt.plot(utils.moving_avg(REWARDS))
                plt.savefig(os.path.join(ckpt_folder, f"rewards_{episode_id:08d}.jpg")); plt.close()
                torch.save({'episode_id': episode_id, 'REWARDS': REWARDS, 'model_G_state_dict': q_net.state_dict()},
                           os.path.join(ckpt_folder, f"ckpt_{episode_id:08d}.pt"))
                print(f"💾 Saved at episode {episode_id}")

    # ==========================================================================
    # [MODE 3] SAC
    # ==========================================================================
    elif args.mode == 'SAC':
        batch_size = 64; buffer_capacity = 100000
        agent = DiscreteSAC(env.state_dims, env.action_dims, lr=args.lr, gamma=args.gamma, tau=args.tau, alpha=args.alpha).to(device)
        buffer = ReplayBuffer(buffer_capacity)

        ckpt_list = glob.glob(os.path.join(ckpt_folder, '*.pt'))
        if len(ckpt_list) > 0:
            ckpt_list.sort()
            checkpoint = torch.load(ckpt_list[-1], map_location=device, weights_only=False)
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.q1.load_state_dict(checkpoint['q1_state_dict'])
            agent.q2.load_state_dict(checkpoint['q2_state_dict'])
            agent.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
            agent.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
            last_episode_id = int(checkpoint['episode_id']) + 1
            REWARDS = checkpoint['REWARDS']
            print(f"🔄 SAC Loaded: {ckpt_list[-1]} (Next episode: {last_episode_id})")

        # [수정] max_m_episode 번호까지 포함해서 실행 (+1)
        target_episode = args.max_m_episode
        print(f"▶️  Running episodes: {last_episode_id} ~ {target_episode}")

        for episode_id in range(last_episode_id, target_episode + 1):
            state = env.reset(); ep_reward = 0; frames = []; 
            
            # [수정] 저장 조건 변경
            do_save = (episode_id % args.save_interval == 0) or (episode_id == target_episode)

            for step_id in range(args.max_steps):
                action, _, _ = agent.get_action(state, deterministic=False)
                next_state, reward, done, _ = env.step(action)
                buffer.push(state, action, reward, next_state, done)
                state = next_state; ep_reward += reward

                if do_save:
                    res = env.render(); img = res[1] if isinstance(res, (tuple, list)) else res
                    if isinstance(img, np.ndarray): frames.append(img)

                if len(buffer) > batch_size: agent.update(buffer, batch_size)
                if done: break

            REWARDS.append(ep_reward)
            if episode_id % 10 == 0: print(f"[{args.mode}] Ep: {episode_id}, Reward: {ep_reward:.3f}")

            if do_save:
                if frames: save_video(frames, os.path.join(ckpt_folder, f"video_{episode_id:08d}.gif"))
                plt.figure(); plt.plot(REWARDS); plt.plot(utils.moving_avg(REWARDS))
                plt.savefig(os.path.join(ckpt_folder, f"rewards_{episode_id:08d}.jpg")); plt.close()
                torch.save({'episode_id': episode_id, 'REWARDS': REWARDS, 
                            'actor_state_dict': agent.actor.state_dict(),
                            'q1_state_dict': agent.q1.state_dict(), 'q2_state_dict': agent.q2.state_dict(),
                            'q1_target_state_dict': agent.q1_target.state_dict(), 'q2_target_state_dict': agent.q2_target.state_dict()}, 
                           os.path.join(ckpt_folder, f"ckpt_{episode_id:08d}.pt"))
                print(f"💾 Saved at episode {episode_id}")