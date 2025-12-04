import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque

# 장치 설정
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- [공통] Positional Mapping ---
class PositionalMapping(nn.Module):
    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        if self.L == 0:
            return x
        h = [x]
        PI = 3.1415927410125732
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)
        return torch.cat(h, dim=-1) / self.scale

# --- [공통] Replay Buffer (DQN, SAC 사용) ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), action, reward, np.array(next_state), done

    def __len__(self):
        return len(self.buffer)


# ==============================================================================
# [MODE 1] A2C 관련 클래스
# ==============================================================================
def calculate_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=7)
        h_dim = 128
        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class ActorCritic(nn.Module):
    # [수정됨] lr 인자를 받을 수 있도록 변경 (기본값 5e-5)
    def __init__(self, input_dim, output_dim, lr=5e-5):
        super().__init__()
        self.output_dim = output_dim
        self.actor = MLP(input_dim=input_dim, output_dim=output_dim)
        self.critic = MLP(input_dim=input_dim, output_dim=1)
        self.softmax = nn.Softmax(dim=-1)
        
        # [수정됨] 입력받은 lr을 옵티마이저에 적용
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, x):
        y = self.actor(x)
        probs = self.softmax(y)
        value = self.critic(x)
        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        probs, value = self.forward(state)
        probs = probs[0, :]
        value = value[0]
        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            if random.random() < exploration:
                action_id = random.randint(0, self.output_dim - 1)
            else:
                action_id = np.random.choice(self.output_dim, p=np.squeeze(probs.detach().cpu().numpy()))
        log_prob = torch.log(probs[action_id] + 1e-9)
        return action_id, log_prob, value

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=0.99):
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()
        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss
        network.optimizer.zero_grad()
        ac_loss.backward()
        network.optimizer.step()


# ==============================================================================
# [MODE 2] DQN 관련 클래스
# ==============================================================================
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, lr=1e-3):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==============================================================================
# [MODE 3] SAC 관련 클래스
# ==============================================================================
class Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.mapping = PositionalMapping(input_dim=input_dim, L=7)
        self.linear1 = nn.Linear(self.mapping.output_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.mapping(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)
    
class DiscreteSAC(nn.Module):
    def __init__(self, input_dim, output_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.actor = Network(input_dim, output_dim).to(device)
        self.q1 = Network(input_dim, output_dim).to(device)
        self.q2 = Network(input_dim, output_dim).to(device)
        self.q1_target = Network(input_dim, output_dim).to(device)
        self.q2_target = Network(input_dim, output_dim).to(device)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if state.dim() == 1: state = state.unsqueeze(0)
            
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            return action.item(), 0, 0 

    def update(self, replay_buffer, batch_size=64):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-8)
            
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            q_target = rewards + (1 - dones) * self.gamma * next_v

        q1 = self.q1(states).gather(1, actions)
        q2 = self.q2(states).gather(1, actions)
        
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        logits = self.actor(states)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        
        with torch.no_grad():
            q1_val = self.q1(states)
            q2_val = self.q2(states)
            min_q = torch.min(q1_val, q2_val)
        
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)