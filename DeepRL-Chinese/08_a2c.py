"""8.3节A2C算法实现。"""
import argparse
import os
from collections import defaultdict
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter


class ValueNet(nn.Module):
    def __init__(self, dim_state,dim_hiden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, dim_hiden)
        self.fc2 = nn.Linear(dim_hiden, int(dim_hiden/2))
        self.fc3 = nn.Linear(int(dim_hiden/2), 1)
        self.neure_l1_av=torch.zeros((1,256))
        self.neure_l2_av=torch.zeros((1,128))
        self.neure_l3_av=torch.zeros((1,1))
        self.is_reset_die_neure=False
        self.count_=0

    def forward(self, state):#      bs*1=>
        l1_a = F.relu(self.fc1(state))#1*256=bs*256=>
        l2_a = F.relu(self.fc2(l1_a))#256*128=bs*128=>
        x = self.fc3(l2_a)          #128*1=bs*1                                 
        #-----------重置死掉的神经元---------------------------
        with torch.no_grad():
            self.count_+=1
            self.neure_l1_av+=torch.abs(l1_a).mean(dim=0)
            self.neure_l2_av+=torch.abs(l2_a).mean(dim=0)
            self.neure_l3_av+=torch.abs(x).mean(dim=0)
            if self.is_reset_die_neure:
                self.ReInitNerunk(self.neure_l1_av/self.count_,self.fc1.weight,self.fc2.weight)
                self.ReInitNerunk(self.neure_l2_av/self.count_,self.fc2.weight,self.fc3.weight)
                self.ReInitNerunk(self.neure_l3_av/self.count_,self.fc3.weight,None)
                self.neure_l1_av=torch.zeros((1,256))
                self.neure_l2_av=torch.zeros((1,128))
                self.neure_l3_av=torch.zeros((1,1))
                self.is_reset_die_neure=False
                self.count_=0
        #-------------------------------------
        return x
    def ReInitNerunk(self,l1_a,l1_weight,lnext_weight):
        l1_a_w=l1_a/l1_a.mean()#计算每个神经元输出占比
        condition_l1_a=l1_a_w>0.001#神经元输出比重小于0.001，判定为死掉,torch.randint(1,100,dtype=float)/100
        #直接修改data可行吗？
        #测试替换是否如预期
        l1_weight.data=torch.where(condition_l1_a.T,l1_weight.data,torch.rand_like(l1_weight.data))#重置输入权重【也可以使用nonzero实现】
        if(lnext_weight==None):
            return
        #测试多替换是否可行
        lnext_weight.data=torch.where(condition_l1_a,lnext_weight.data,0)


class PolicyNet(nn.Module):
    def __init__(self, dim_state, num_action,dim_hiden=256):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, dim_hiden)
        self.fc2 = nn.Linear(dim_hiden, int(dim_hiden/2))
        self.fc3 = nn.Linear(int(dim_hiden/2), num_action)
        self.neure_l1_av=torch.zeros((1,256))
        self.neure_l2_av=torch.zeros((1,128))
        self.neure_l3_av=torch.zeros((1,4))
        self.is_reset_die_neure=False
        self.count_=0
    def forward(self, state):
        l1_a = F.relu(self.fc1(state))
        l2_a = F.relu(self.fc2(l1_a))
        l3 = self.fc3(l2_a)
        prob = F.softmax(l3, dim=-1)
                #-----------重置死掉的神经元---------------------------
        with torch.no_grad():
            self.count_+=1
            self.neure_l1_av+=torch.abs(l1_a).mean(dim=0)
            self.neure_l2_av+=torch.abs(l2_a).mean(dim=0)
            self.neure_l3_av+=torch.abs(prob).mean(dim=0)
            if self.is_reset_die_neure:
                self.ReInitNerunk(self.neure_l1_av/self.count_,self.fc1.weight,self.fc2.weight)
                self.ReInitNerunk(self.neure_l2_av/self.count_,self.fc2.weight,self.fc3.weight)
                self.ReInitNerunk(self.neure_l3_av/self.count_,self.fc3.weight,None)
                self.neure_l1_av=torch.zeros((1,256))
                self.neure_l2_av=torch.zeros((1,128))
                self.neure_l3_av=torch.zeros((1,4))
                self.is_reset_die_neure=False
                self.count_=0
        #-------------------------------------

        return prob
    def ReInitNerunk(self,l1_a,l1_weight,lnext_weight):
        l1_a_w=l1_a/l1_a.mean()#计算每个神经元输出占比
        condition_l1_a=l1_a_w>0.001#神经元输出比重小于0.001，判定为死掉,torch.randint(1,100,dtype=float)/100
        #直接修改data可行吗？
        #测试替换是否如预期
        l1_weight.data=torch.where(condition_l1_a.T,l1_weight.data,torch.rand_like(l1_weight.data))#重置输入权重【也可以使用nonzero实现】
        if(lnext_weight==None):
            return
        #测试多替换是否可行
        lnext_weight.data=torch.where(condition_l1_a,lnext_weight.data,0)


class A2C:
    def __init__(self, args):
        self.args = args
        self.V = ValueNet(args.dim_state)
        self.V_target = ValueNet(args.dim_state)
        self.pi = PolicyNet(args.dim_state, args.num_action)
        self.V_target.load_state_dict(self.V.state_dict())

    def get_action(self, state):
        '''
        模拟一个行为
        '''
        probs = self.pi(state)
        m = Categorical(probs)
        action = m.sample()
        logp_action = m.log_prob(action)
        return action, logp_action

    def compute_value_loss(self, bs, blogp_a, br, bd, bns):
        # 目标价值。计算优势函数的v和t_v
        with torch.no_grad():
            target_value = br + self.args.discount  * self.V_target(bns.unsqueeze(-1)).squeeze()

        # 计算value loss。
        value_loss = F.mse_loss(self.V(bs.unsqueeze(-1)).squeeze(), target_value)
        return value_loss

    def compute_policy_loss(self, bs, blogp_a, br, bd, bns):
        # 目标价值。
        with torch.no_grad():
            #gs:即模拟s_value的s_next_value
            #需要注意的是这里只从s+1算了s,而不是对每个s+n算到s0
            target_value = br + self.args.discount  * self.V_target(bns.unsqueeze(-1)).squeeze()

        # 计算policy loss。
        with torch.no_grad():
            advantage = target_value - self.V(bs.unsqueeze(-1)).squeeze()
        policy_loss = 0
        for i, logp_a in enumerate(blogp_a):
            policy_loss += -logp_a * advantage[i]
        policy_loss = policy_loss.mean()
        return policy_loss

    def soft_update(self, tau=0.1):
        def soft_update_(target, source, tau_=0.01):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau_) + param.data * tau_)

        soft_update_(self.V_target, self.V, tau)

class Rollout:
    '''用于处理一个存于队列的episode'''
    def __init__(self):
        self.state_lst = []
        self.action_lst = []
        self.logp_action_lst = []
        self.reward_lst = []
        self.done_lst = []
        self.next_state_lst = []

    def put(self, state, action, logp_action, reward, done, next_state):
        '''放入s,a,log_a,r,done,next_s'''
        self.state_lst.append(state)
        self.action_lst.append(action)
        self.logp_action_lst.append(logp_action)
        self.reward_lst.append(reward)
        self.done_lst.append(done)
        self.next_state_lst.append(next_state)

    def tensor(self):
        '''返回放入s,a,log_a,r,done,next_s的list'''
        bs = torch.as_tensor(self.state_lst).float()
        ba = torch.as_tensor(self.action_lst).float()
        blogp_a = self.logp_action_lst
        br = torch.as_tensor(self.reward_lst).float()
        bd = torch.as_tensor(self.done_lst)
        bns = torch.as_tensor(self.next_state_lst).float()
        return bs, ba, blogp_a, br, bd, bns


class INFO:
    def __init__(self):
        self.log = defaultdict(list)
        self.episode_length = 0
        self.episode_reward = 0
        self.max_episode_reward = -float("inf")

    def put(self, done, reward):
        if done is True:
            self.episode_length += 1
            self.episode_reward += reward
            self.log["episode_length"].append(self.episode_length)
            self.log["episode_reward"].append(self.episode_reward)

            if self.episode_reward > self.max_episode_reward:
                self.max_episode_reward = self.episode_reward

            self.episode_length = 0
            self.episode_reward = 0

        else:
            self.episode_length += 1
            self.episode_reward += reward


def train(args, env, agent: A2C):
    V_optimizer = torch.optim.Adam(agent.V.parameters(), lr=1e-3)
    pi_optimizer = torch.optim.Adam(agent.pi.parameters(), lr=1e-3)
    info = INFO()

    rollout = Rollout()
    state= env.reset()
    false_num=0
    train_num=0
    for step in range(args.max_steps):
        #env.render()
        action, logp_action = agent.get_action(torch.tensor([state]).float())
        next_state, reward, terminated, _ = env.step(action.item())
        done = terminated and (reward!=0.0)
        if done:
            reward=5
        else:
             reward-=1
        rollout.put(
            state,
            action,
            logp_action,
            reward,
            done,
            next_state,
        )
        state = next_state
        if(not done):
            false_num+=1
            if false_num>50:
                false_num=0
                #done=True#方案2，如果长度超过100则按照截断处理
                #重置环境。
                state= env.reset()
                rollout = Rollout()
                #不使用infog.put(done=ture,r)，直接归零episode_length和episode_reward
                #当做无事发生
                info.episode_length = 0
                info.episode_reward = 0
        info.put(done, reward)
        #怎么感觉是离线算法，先看下更新和使用的是否为同一个模型
        if done is True:
            train_num+=1
            # 模型训练。
            bs, ba, blogp_a, br, bd, bns = rollout.tensor()

            value_loss = agent.compute_value_loss(bs, blogp_a, br, bd, bns)
            V_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            V_optimizer.step()

            policy_loss = agent.compute_policy_loss(bs, blogp_a, br, bd, bns)
            pi_optimizer.zero_grad()
            policy_loss.backward()
            pi_optimizer.step()

            agent.soft_update()

            # 打印信息。
            info.log["value_loss"].append(value_loss.item())
            info.log["policy_loss"].append(policy_loss.item())

            episode_reward = info.log["episode_reward"][-1]
            episode_length = info.log["episode_length"][-1]
            value_loss = info.log["value_loss"][-1]
            print(f"step={step}, reward={episode_reward:.0f}, length={episode_length}, max_reward={info.max_episode_reward}, value_loss={value_loss:.1e}")
            writer.add_scalars("reward",{"train":episode_reward},episode_length)
            writer.add_scalars("loss_value",{"train":value_loss},episode_length)
            writer.add_scalars("loss_policy",{"train":policy_loss},episode_length)
            # 重置环境。
            state= env.reset()
            rollout = Rollout()
            false_num=0
            if(train_num>200):
                agent.pi.is_reset_die_neure=True
                agent.V.is_reset_die_neure=True
                train_num=0
            # 保存模型。
            if episode_reward == info.max_episode_reward:
                save_path = os.path.join(args.output_dir, "model.bin")
                torch.save(agent.pi.state_dict(), save_path)
    
        if step % 10000 == 0:
            plt.plot(info.log["value_loss"], label="value loss")
            plt.legend()
            if os.path.exists(args.output_dir) is False:
                os.mkdir(args.output_dir)
            plt.savefig(f"{args.output_dir}/value_loss.png", bbox_inches="tight")
            plt.close()

            plt.plot(info.log["episode_reward"])
            plt.savefig(f"{args.output_dir}/episode_reward.png", bbox_inches="tight")
            plt.close()

            plt.plot(info.log["policy_loss"])
            plt.savefig(f"{args.output_dir}/policy_loss.png", bbox_inches="tight")
            plt.close()

            plt.plot(info.log["episode_length"])
            plt.savefig(f"{args.output_dir}/episode_length.png", bbox_inches="tight")
            plt.close()

def eval(args, env, agent):
    agent = A2C(args)
    model_path = os.path.join(args.output_dir, "model.bin")
    agent.pi.load_state_dict(torch.load(model_path))

    episode_length = 0
    episode_reward = 0
    state = env.reset()
    for i in range(5000):
        env.render()
        episode_length += 1
        action, _ = agent.get_action(torch.tensor([state]).float())
        next_state, reward, terminated, info = env.step(action.item())
        done = terminated
        episode_reward += reward

        state = next_state
        if done is True:
            print(f"episode reward={episode_reward}, length={episode_length}")
            state = env.reset()
            episode_length = 0
            episode_reward = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="FrozenLake-v1", type=str, help="Environment name.")
    parser.add_argument("--dim_state", default=1, type=int, help="Dimension of state.")
    parser.add_argument("--num_action", default=4, type=int, help="Number of action.")
    parser.add_argument("--output_dir", default="output", type=str, help="Output directory.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")

    parser.add_argument("--max_steps", default=100_000_00, type=int, help="Maximum steps for interaction.")
    parser.add_argument("--discount", default=0.99, type=float, help="Discount coefficient.")
    parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--do_train",default=True,action="store_true", help="Train policy.")
    parser.add_argument("--do_eval",  action="store_true", help="Evaluate policy.")
    args = parser.parse_args()
    writer=SummaryWriter()
    env = gym.make(args.env)
    agent = A2C(args)

    if args.do_train:
        train(args, env, agent)

    if args.do_eval:
        eval(args, env, agent)
# 1，迁移到冰湖环境测试可行否
# 2，发现奖励太过稀疏，于是只使用done的数据训练，发现loss一直在增加
# 3，取消只是用done的数据，但限制最大长度为30#--------------ing
# 4，如果不行就从2开始分析，加入死神经元激活#