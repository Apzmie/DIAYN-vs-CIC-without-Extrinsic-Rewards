from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter       

BASE_DIR = ""


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))       
        mean = self.mean(x)
        
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std

    def sample(self, state, z):
        mean, log_std = self.forward(state, z)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mean, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(pre_tanh)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
        
    def deterministic(self, state, z):
        mean, _ = self.forward(state, z)
        return torch.tanh(mean)
        

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action, z):
        x = torch.cat([state, action, z], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.q(x)
        return q


class Discriminator(nn.Module):
    def __init__(self, state_dim, z_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, z_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits
                

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, z_dim, max_size=int(1e6), batch_size=256):
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)
        self.z = np.zeros((max_size, z_dim), dtype=np.float32)

    def add(self, state, action, reward, next_state, done, z):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        self.z[self.ptr] = z

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idx = np.random.randint(0, self.size, size=self.batch_size)

        return {
            "state": self.state[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
            "next_state": self.next_state[idx],
            "done": self.done[idx],
            "z": self.z[idx]
        }
        
        
class SACAgent:
    def __init__(self, state_dim, action_dim, z_dim, lr=3e-4):
        self.actor = PolicyNetwork(state_dim, action_dim, z_dim)
        self.critic1 = QNetwork(state_dim, action_dim, z_dim)
        self.critic2 = QNetwork(state_dim, action_dim, z_dim)
        self.critic1_target = QNetwork(state_dim, action_dim, z_dim)
        self.critic2_target = QNetwork(state_dim, action_dim, z_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.discriminator = Discriminator(state_dim, z_dim)
        
        ###########################################
        ### Load Actor (fc1, fc2, mean) ###
        # Set random_exploration_steps to 0, learning_starts to the minimum
        ###########################################
        
        #state_dict = torch.load(f"{BASE_DIR}/previous_model.pth")
        #self.actor.fc1.load_state_dict({"weight": state_dict["fc1.weight"], "bias": state_dict["fc1.bias"]})
        #self.actor.fc2.load_state_dict({"weight": state_dict["fc2.weight"], "bias": state_dict["fc2.bias"]})
        #self.actor.mean.load_state_dict({"weight": state_dict["mean.weight"], "bias": state_dict["mean.bias"]})
        
        #with torch.no_grad():        
        #    self.actor.log_std.weight.zero_()
        #    self.actor.log_std.bias.fill_(-2)        
        #self.log_alpha = nn.Parameter(torch.tensor([-9.0]))
        
        #==========================================
        
        ###########################################
        ### Load one DIAYN skill without z_dim input ###
        ###########################################
        
        #state_dict = torch.load(f"{BASE_DIR}/previous_model.pth")
        
        #z_dim = ?
        #z = torch.zeros(z_dim)
        #z[?] = 1.0
        
        #old_weight = state_dict["fc1.weight"]
        #old_bias = state_dict["fc1.bias"]
        
        #state_weight = old_weight[:, :state_dim]
        #z_weight = old_weight[:, state_dim:]
        
        #new_bias = old_bias + z_weight @ z
        
        #with torch.no_grad():
        #    self.model.fc1.weight.copy_(state_weight)
        #    self.model.fc1.bias.copy_(new_bias)
        
        #self.actor.fc2.load_state_dict({
        #    "weight": state_dict["fc2.weight"],
        #    "bias": state_dict["fc2.bias"]
        #})

        #self.actor.mean.load_state_dict({
        #    "weight": state_dict["mean.weight"],
        #    "bias": state_dict["mean.bias"]
        #})
        
        #==========================================
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        self.log_alpha = nn.Parameter(torch.zeros(1))  #
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        self.target_entropy = -action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.z_dim = z_dim
        
    def sample_z(self):
        idx = np.random.randint(self.z_dim)
        z = np.zeros(self.z_dim, dtype=np.float32)
        z[idx] = 1.0
        return z
        
    def update_target(self, net, target_net):
        with torch.no_grad():
            for param, target_param in zip(net.parameters(), target_net.parameters()):
                target_param.copy_(
                    self.tau * param + (1 - self.tau) * target_param
                )

    def update(self, batch):
        state = torch.FloatTensor(batch['state'])
        action = torch.FloatTensor(batch['action'])
        reward = torch.FloatTensor(batch['reward'])
        next_state = torch.FloatTensor(batch['next_state'])
        done = torch.FloatTensor(batch['done'])
        z = torch.FloatTensor(batch["z"])
        
        #==========================================

        disc_logits = self.discriminator(next_state)
        skill = z.argmax(dim=-1)
        disc_loss = F.cross_entropy(disc_logits, skill)  # include softmax
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.disc_optimizer.step()
        
        with torch.no_grad():
            log_q = F.log_softmax(disc_logits, dim=-1)
            intrinsic_reward = (log_q.gather(1, skill.unsqueeze(1)) - np.log(1 / self.z_dim))
            total_reward = reward + intrinsic_reward
        
        #==========================================
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state, z)
            
            next_q1 = self.critic1_target(next_state, next_action, z)
            next_q2 = self.critic2_target(next_state, next_action, z)
            next_q = torch.min(next_q1, next_q2)
            
            alpha = self.log_alpha.exp()            
            target_q = total_reward + self.gamma * (1 - done) * (next_q - alpha * next_log_prob)
            
        q1 = self.critic1(state, action, z)
        q2 = self.critic2(state, action, z)
        
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()
        
        #==========================================
        
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False
        
        action_new, log_prob = self.actor.sample(state, z)
        
        q1_new = self.critic1(state, action_new, z)
        q2_new = self.critic2(state, action_new, z)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = self.log_alpha.exp().detach()    
        actor_loss = -(q_new - alpha * log_prob).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True
        
        #==========================================

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        #==========================================
        
        self.update_target(self.critic1, self.critic1_target)
        self.update_target(self.critic2, self.critic2_target)
        
        return {
            "discriminator_loss": disc_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "intrinsic_reward": intrinsic_reward.mean().item(),
            "total_reward": total_reward.mean().item()
        }


def save_checkpoint(path, agent, buffer):
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic1": agent.critic1.state_dict(),
        "critic2": agent.critic2.state_dict(),
        "critic1_target": agent.critic1_target.state_dict(),
        "critic2_target": agent.critic2_target.state_dict(),
        "discriminator": agent.discriminator.state_dict(),

        "actor_optimizer": agent.actor_optimizer.state_dict(),
        "critic1_optimizer": agent.critic1_optimizer.state_dict(),
        "critic2_optimizer": agent.critic2_optimizer.state_dict(),
        "disc_optimizer": agent.disc_optimizer.state_dict(),
        "alpha_optimizer": agent.alpha_optimizer.state_dict(),

        "log_alpha": agent.log_alpha.detach().cpu(),

        "replay_buffer": {
            "state": buffer.state,
            "action": buffer.action,
            "reward": buffer.reward,
            "next_state": buffer.next_state,
            "done": buffer.done,
            "z": buffer.z,
            "ptr": buffer.ptr,
            "size": buffer.size
        }
    }, path)
    
    
def load_checkpoint(path, agent, buffer):
    ckpt = torch.load(path)

    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic1.load_state_dict(ckpt["critic1"])
    agent.critic2.load_state_dict(ckpt["critic2"])
    agent.critic1_target.load_state_dict(ckpt["critic1_target"])
    agent.critic2_target.load_state_dict(ckpt["critic2_target"])
    agent.discriminator.load_state_dict(ckpt["discriminator"])

    agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
    agent.critic1_optimizer.load_state_dict(ckpt["critic1_optimizer"])
    agent.critic2_optimizer.load_state_dict(ckpt["critic2_optimizer"])
    agent.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
    agent.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
    
    with torch.no_grad():
        agent.log_alpha.copy_(ckpt["log_alpha"])

    buffer.state = ckpt["replay_buffer"]["state"]
    buffer.action = ckpt["replay_buffer"]["action"]
    buffer.reward = ckpt["replay_buffer"]["reward"]
    buffer.next_state = ckpt["replay_buffer"]["next_state"]
    buffer.done = ckpt["replay_buffer"]["done"]
    buffer.z = ckpt["replay_buffer"]["z"]
    buffer.ptr = ckpt["replay_buffer"]["ptr"]
    buffer.size = ckpt["replay_buffer"]["size"]
    

if __name__ == "__main__":
    channel1 = EngineConfigurationChannel()
    channel1.set_configuration_parameters(time_scale=20.0)
    channel2 = EngineConfigurationChannel()
    channel2.set_configuration_parameters(time_scale=20.0)
    env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel1], no_graphics=True, worker_id=0)
    test_env = UnityEnvironment(file_name=f"{BASE_DIR}/Build.x86_64", side_channels=[channel2], no_graphics=True, worker_id=1)
    env.reset()
    test_env.reset()
    
    behavior_name = list(env.behavior_specs.keys())[0]
    t_behavior_name = list(test_env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    state_dim = spec.observation_specs[0].shape[0]
    action_dim = spec.action_spec.continuous_size    
    z_dim = 8    
    agent = SACAgent(state_dim, action_dim, z_dim)
    buffer = ReplayBuffer(state_dim, action_dim, z_dim)
    writer = SummaryWriter(log_dir=BASE_DIR)
    
    # Set random_exploration_steps, learning_starts to 0
    #load_checkpoint(f"{BASE_DIR}/checkpoint.pth", agent, buffer)
    
    #agent.actor.load_state_dict(torch.load(f"{BASE_DIR}/previous_model.pth"))
    
    random_exploration_steps = 0
    learning_starts = 256
    test_interval = 1000
    max_collected = 5000
    
    total_steps = 0
    update_count = 0
    save_idx = 0
    best_test_score = -float('inf')
    current_z = {}
    
    while True:
        decision_steps, terminal_steps = env.get_steps(behavior_name)

        agent_ids = decision_steps.agent_id
        if len(agent_ids) > 0:
            states_tensor = torch.from_numpy(decision_steps.obs[0]).to(torch.float32)
            
            z_batch = []
            for agent_id in agent_ids:
                if agent_id not in current_z:
                    current_z[agent_id] = agent.sample_z()
                z_batch.append(current_z[agent_id])
            z_tensor = torch.tensor(np.array(z_batch), dtype=torch.float32)
            
            if total_steps < random_exploration_steps:
                actions = np.random.uniform(low=-1.0, high=1.0, size=(len(agent_ids), action_dim)).astype(np.float32)
            else:
                with torch.no_grad():
                    actions, _ = agent.actor.sample(states_tensor, z_tensor)   
                actions = actions.cpu().numpy().astype(np.float32)
                
            env.set_actions(behavior_name, ActionTuple(continuous=actions))
            
        env.step()
        next_decision_steps, terminal_steps = env.get_steps(behavior_name)
        
        for i, agent_id in enumerate(agent_ids):
            state = states_tensor[i].cpu().numpy()
            action = actions[i]

            if agent_id in terminal_steps:
                reward = terminal_steps[agent_id].reward
                done = 1.0
                next_state = np.zeros_like(state)
            elif agent_id in next_decision_steps:
                reward = next_decision_steps[agent_id].reward
                done = 0.0
                next_state = next_decision_steps[agent_id].obs[0]
            else:
                continue
                
            buffer.add(state, action, reward, next_state, done, current_z[agent_id])
            if done:
                current_z.pop(agent_id, None)
            total_steps += 1
         
        if total_steps >= learning_starts:
             batch = buffer.sample()
             metrics = agent.update(batch) 
             update_count += 1           
             for k, v in metrics.items():
                 writer.add_scalar(f"Train/{k}", v, update_count)               
             
             if update_count % test_interval == 0:
                 print(f"Update Count {update_count}")
                 test_states = {z: [] for z in range(z_dim)}

                 for z in range(z_dim):
                     test_env.reset()
                     collected = 0
                     
                     while collected < max_collected:
                         t_decision_steps, _ = test_env.get_steps(t_behavior_name)
                         t_agent_ids = t_decision_steps.agent_id
                         if len(t_agent_ids) > 0:
                             t_states_tensor = torch.from_numpy(t_decision_steps.obs[0]).to(torch.float32) 
                             tz_tensor = torch.zeros((len(t_agent_ids), z_dim), dtype=torch.float32)
                             tz_tensor[:, z] = 1.0                        
                             with torch.no_grad():
                                 t_actions = agent.actor.deterministic(t_states_tensor, tz_tensor)                    
                             t_actions = t_actions.cpu().numpy().astype(np.float32)
                             test_env.set_actions(t_behavior_name, ActionTuple(continuous=t_actions))
                             
                             for s in t_states_tensor:
                                 if collected >= max_collected:
                                     break
                                 test_states[z].append(s.detach().cpu().numpy())
                                 collected += 1
                         
                         test_env.step()
                 
                 tz_means = {}
                 for z in range(z_dim):                         
                     tz_means[z] = np.mean(np.array(test_states[z]), axis=0)
                 
                 means = np.stack([tz_means[z] for z in range(z_dim)])
                 global_mean = np.mean(means, axis=0)
                 distance_score = np.mean(np.linalg.norm(means - global_mean, axis=1))
             
                 writer.add_scalar("Test/distance_score", distance_score, update_count)
                 print(f"{distance_score:.4f}")
                 torch.save(agent.actor.state_dict(), f"{BASE_DIR}/period_model.pth")
                 save_checkpoint(f"{BASE_DIR}/checkpoint.pth", agent, buffer)                    
                         
                 if distance_score > best_test_score:
                     best_test_score = distance_score
                     save_idx += 1
                     torch.save(agent.actor.state_dict(), f"{BASE_DIR}/#({save_idx})best_{best_test_score:.4f}.pth") 
                     print(f"[Test] Model saved at new best score {best_test_score:.4f}")
                     
