import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
import cv2
from PIL import Image
import io

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

class VideoRecorder:
    """Video recording utility for PyBullet simulations"""
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frames = []
        self.recording = False
        
    def start_recording(self):
        """Start video recording"""
        self.frames = []
        self.recording = True
        print("Started video recording...")
        
    def stop_recording(self):
        """Stop video recording"""
        self.recording = False
        print(f"Stopped recording. Captured {len(self.frames)} frames.")
        
    def capture_frame(self):
        """Capture a single frame from PyBullet"""
        if not self.recording:
            return
            
        # Get camera image from PyBullet
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0.5],
            distance=2.0,
            yaw=45,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width/self.height,
            nearVal=0.1,
            farVal=100.0
        )
        
        # Capture image
        _, _, rgba_img, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to RGB
        rgb_img = np.array(rgba_img)[:, :, :3]
        self.frames.append(rgb_img)
        
    def save_video(self, filename):
        """Save recorded frames as MP4 video"""
        if not self.frames:
            print("No frames to save!")
            return
            
        print(f"Saving video with {len(self.frames)} frames...")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
        
        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
            
        out.release()
        print(f"Video saved as {filename}")
        
    def create_gif(self, filename, duration=None):
        """Create GIF from recorded frames"""
        if not self.frames:
            print("No frames to save!")
            return
            
        print(f"Creating GIF with {len(self.frames)} frames...")
        
        # Convert frames to PIL Images
        pil_frames = [Image.fromarray(frame) for frame in self.frames]
        
        # Calculate duration
        if duration is None:
            duration = int(1000 / self.fps)  # ms per frame
            
        # Save as GIF
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved as {filename}")

class RobotWaveEnvironment:
    def __init__(self, robot_urdf_path=None, gui=False, record_video=False):
        """
        Initialize the robot waving environment
        """
        self.gui = gui
        self.record_video = record_video
        
        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        
        # If no robot URDF provided, use a simple robot for demonstration
        if robot_urdf_path is None:
            # Create a simple robot for testing (you should replace with your actual robot)
            self.robot_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])
        else:
            self.robot_id = p.loadURDF(robot_urdf_path, [0, 0, 0.5])
        
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"Robot has {self.num_joints} joints")
        
        # Get joint info
        self.joint_info = []
        self.controllable_joints = []
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            self.joint_info.append(info)
            # Only consider revolute and prismatic joints
            if info[2] in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.controllable_joints.append(i)
        
        print(f"Controllable joints: {len(self.controllable_joints)}")
        
        # Define waving parameters
        self.wave_frequency = 2.0  # Hz
        self.wave_amplitude = 0.5  # radians
        self.episode_length = 240  # simulation steps (4 seconds at 60 FPS)
        self.current_step = 0
        
        # Target waving pattern for primary joint (usually shoulder or elbow)
        self.primary_joint_idx = 0 if len(self.controllable_joints) > 0 else 0
        
        # Initialize joint positions
        self.initial_positions = [0.0] * self.num_joints
        
        # Video recorder
        self.video_recorder = VideoRecorder() if record_video else None
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        self.current_step = 0
        
        # Reset robot to initial position
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, self.initial_positions[i])
        
        # Let physics settle
        for _ in range(10):
            p.stepSimulation()
        
        return self.get_state()
    
    def get_state(self):
        """Get current state of the robot"""
        joint_states = []
        for i in range(self.num_joints):
            pos, vel, _, _ = p.getJointState(self.robot_id, i)
            joint_states.extend([pos, vel])
        
        # Add time information for temporal awareness
        time_normalized = (self.current_step / self.episode_length) * 2 * np.pi
        joint_states.extend([np.sin(time_normalized), np.cos(time_normalized)])
        
        return np.array(joint_states, dtype=np.float32)
    
    def get_target_wave_position(self, joint_idx):
        """Generate target waving position for a given joint"""
        if joint_idx == self.primary_joint_idx:
            # Primary waving motion
            t = self.current_step / 60.0  # Convert to seconds (assuming 60 FPS)
            return self.wave_amplitude * np.sin(2 * np.pi * self.wave_frequency * t)
        else:
            # Secondary joints can have complementary motions
            t = self.current_step / 60.0
            return 0.1 * np.sin(2 * np.pi * self.wave_frequency * t + np.pi/4)
    
    def calculate_reward(self, action):
        """Calculate reward based on how well the robot is waving"""
        reward = 0.0
        
        # Get current joint positions
        current_positions = []
        for i in range(self.num_joints):
            pos, vel, _, _ = p.getJointState(self.robot_id, i)
            current_positions.append(pos)
        
        # Reward for following the waving pattern
        total_error = 0
        for i, joint_idx in enumerate(self.controllable_joints):
            target_pos = self.get_target_wave_position(joint_idx)
            actual_pos = current_positions[joint_idx]
            
            # Distance reward (exponential decay instead of linear penalty)
            distance_error = abs(target_pos - actual_pos)
            if joint_idx == self.primary_joint_idx:
                # Reward good tracking, penalize bad tracking
                reward += max(0, 1.0 - distance_error) * 5  # Positive reward for good tracking
                total_error += distance_error * 2
            else:
                reward += max(0, 0.5 - distance_error) * 2
                total_error += distance_error
        
        # Apply total error penalty (less harsh)
        reward -= total_error
        
        # Smoother action penalty (less harsh)
        action_penalty = sum(abs(a) for a in action) * 0.01
        reward -= action_penalty
        
        # Time-based completion bonus
        progress_bonus = (self.current_step / self.episode_length) * 0.5
        reward += progress_bonus
        
        # Completion bonus
        if self.current_step >= self.episode_length - 1:
            reward += 10.0
        
        return reward

    def step(self, action):
        """Execute one step in the environment"""
        # Apply actions to controllable joints
        for i, joint_idx in enumerate(self.controllable_joints):
            if i < len(action):
                p.setJointMotorControl2(
                    self.robot_id, 
                    joint_idx, 
                    p.POSITION_CONTROL, 
                    targetPosition=action[i],
                    force=50
                )
        
        # Step physics
        p.stepSimulation()
        if self.gui:
            time.sleep(1./60.)  # 60 FPS
        
        # Capture frame if recording
        if self.video_recorder:
            self.video_recorder.capture_frame()
        
        self.current_step += 1
        
        # Get new state and calculate reward
        new_state = self.get_state()
        reward = self.calculate_reward(action)
        done = self.current_step >= self.episode_length
        
        return new_state, reward, done, {}

class PolicyNetwork(nn.Module):
    """GPU-accelerated policy network using PyTorch"""
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))  # Output between -1 and 1
        return x

class ValueNetwork(nn.Module):
    """Value network for actor-critic"""
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class PPOTrainer:
    """GPU-accelerated PPO trainer"""
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        self.device = device
        
        # Networks
        self.policy_net = PolicyNetwork(input_size, hidden_size, output_size).to(device)
        self.value_net = ValueNetwork(input_size, hidden_size).to(device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-3)
        
        # PPO parameters
        self.gamma = 0.99
        self.tau = 0.95
        self.epsilon = 0.2
        self.batch_size = 64
        self.update_epochs = 4
        
        # Storage
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        self.episode_rewards = []
        self.best_reward = float('-inf')
    
    def get_action(self, state, training=True):
        """Get action from policy network"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_mean = self.policy_net(state_tensor)
            value = self.value_net(state_tensor)
        
        if training:
            # Add exploration noise
            std = torch.ones_like(action_mean) * 0.1
            dist = Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            action = torch.clamp(action, -1, 1)
            return action.cpu().numpy().flatten(), log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]
        else:
            return action_mean.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        returns = []
        advantages = []
        
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                nextnonterminal = 1.0 - self.dones[step]
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[step + 1]
                nextvalues = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * nextvalues * nextnonterminal - self.values[step]
            gae = delta + self.gamma * self.tau * nextnonterminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
        
        return returns, advantages
    
    def update_networks(self):
        """Update policy and value networks using PPO"""
        if len(self.states) < self.batch_size:
            return
        
        # Get final value for GAE computation
        final_state = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.value_net(final_state).cpu().numpy()[0][0]
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.FloatTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update networks
        for _ in range(self.update_epochs):
            # Random mini-batches
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Policy update
                action_mean = self.policy_net(batch_states)
                std = torch.ones_like(action_mean) * 0.1
                dist = Normal(action_mean, std)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                
                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()
                
                # Value update
                values = self.value_net(batch_states).squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()
        
        # Clear memory
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def train_episode(self, record_video=False):
        """Train one episode"""
        state = self.env.reset()
        episode_reward = 0
        done = False
        
        # Start video recording if requested
        if record_video and self.env.video_recorder:
            self.env.video_recorder.start_recording()
        
        while not done:
            action, log_prob, value = self.get_action(state, training=True)
            next_state, reward, done, _ = self.env.step(action)
            
            self.store_transition(state, action, reward, log_prob, value, done)
            
            state = next_state
            episode_reward += reward
        
        # Stop video recording if it was started
        if record_video and self.env.video_recorder:
            self.env.video_recorder.stop_recording()
        
        return episode_reward
    
    def train(self, num_episodes=1000, update_frequency=10, video_frequency=100):
        """Train the policy using PPO with video recording"""
        print("Starting GPU-accelerated PPO training...")
        
        # Create videos directory
        os.makedirs('training_videos', exist_ok=True)
        
        for episode in range(num_episodes):
            # Record video for specific episodes
            record_this_episode = (episode % video_frequency == 0) or (episode == num_episodes - 1)
            
            episode_reward = self.train_episode(record_video=record_this_episode)
            self.episode_rewards.append(episode_reward)
            
            # Save video if recorded
            if record_this_episode and self.env.video_recorder:
                video_filename = f'training_videos/episode_{episode:04d}.mp4'
                gif_filename = f'training_videos/episode_{episode:04d}.gif'
                
                self.env.video_recorder.save_video(video_filename)
                self.env.video_recorder.create_gif(gif_filename)
            
            # Update networks periodically
            if (episode + 1) % update_frequency == 0:
                self.update_networks()
            
            # Track best performance
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model('best_model.pth')
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Avg={avg_reward:.2f}, Best={self.best_reward:.2f}")
        
        print(f"Training completed! Best reward: {self.best_reward:.2f}")
        return self.policy_net
    
    def save_model(self, filename):
        """Save the trained model"""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, filename)
    
    def load_model(self, filename):
        """Load a trained model"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

def record_final_demonstration(env, policy_net, episodes=3):
    """Record final demonstration videos"""
    print("Recording final demonstration...")
    policy_net.eval()
    
    os.makedirs('final_demos', exist_ok=True)
    
    with torch.no_grad():
        for episode in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            # Start recording
            env.video_recorder.start_recording()
            
            print(f"Recording demo episode {episode + 1}")
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).cpu().numpy().flatten()
                
                state, reward, done, _ = env.step(action)
                episode_reward += reward
            
            # Stop recording and save
            env.video_recorder.stop_recording()
            
            video_filename = f'final_demos/final_demo_{episode + 1}.mp4'
            gif_filename = f'final_demos/final_demo_{episode + 1}.gif'
            
            env.video_recorder.save_video(video_filename)
            env.video_recorder.create_gif(gif_filename)
            
            print(f"Demo {episode + 1} saved. Total reward: {episode_reward:.2f}")

def main():
    """Main training and demonstration function"""
    print("Initializing GPU-Accelerated Robot Waving Training Environment...")
    
    # Check if running on Kaggle
    is_kaggle = '/kaggle' in os.getcwd()
    
    # Create environment with video recording enabled
    env = RobotWaveEnvironment(gui=False, record_video=True)
    
    # Calculate network parameters
    state = env.reset()
    input_size = len(state)
    hidden_size = 128
    output_size = len(env.controllable_joints)
    
    print(f"Network architecture: {input_size} -> {hidden_size} -> {output_size}")
    
    # Create GPU-accelerated trainer
    trainer = PPOTrainer(env, input_size, hidden_size, output_size)
    
    # Train the network with video recording
    print("Training with video recording...")
    episodes = 300 if is_kaggle else 500
    video_freq = 50 if is_kaggle else 100
    
    best_policy = trainer.train(
        num_episodes=episodes, 
        update_frequency=10, 
        video_frequency=video_freq
    )
    
    # Plot training progress
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(trainer.episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    # Moving average
    window = 50
    if len(trainer.episode_rewards) > window:
        moving_avg = np.convolve(trainer.episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(moving_avg)
        plt.title(f'Moving Average (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.text(0.5, 0.5, f'Training Device: {device}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.3, f'Total Episodes: {episodes}', ha='center', va='center', transform=plt.gca().transAxes)
    plt.text(0.5, 0.1, f'Video Frequency: Every {video_freq} episodes', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Training Info')
    
    plt.subplot(2, 3, 4)
    # Final performance
    final_performance = trainer.episode_rewards[-50:] if len(trainer.episode_rewards) > 50 else trainer.episode_rewards
    plt.hist(final_performance, bins=20, alpha=0.7)
    plt.title('Final Performance Distribution')
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    # Best reward progression
    best_rewards = []
    current_best = float('-inf')
    for reward in trainer.episode_rewards:
        if reward > current_best:
            current_best = reward
        best_rewards.append(current_best)
    plt.plot(best_rewards)
    plt.title('Best Reward Progression')
    plt.xlabel('Episode')
    plt.ylabel('Best Reward')
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    # Video recording info
    video_episodes = list(range(0, episodes, video_freq)) + [episodes - 1]
    video_rewards = [trainer.episode_rewards[i] for i in video_episodes if i < len(trainer.episode_rewards)]
    plt.scatter(video_episodes[:len(video_rewards)], video_rewards, color='red', s=50, alpha=0.7)
    plt.plot(trainer.episode_rewards, alpha=0.3)
    plt.title('Video Recorded Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend(['All Episodes', 'Video Recorded'])
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress_with_videos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Record final demonstration videos
    print("\nRecording final demonstration videos...")
    record_final_demonstration(env, best_policy, episodes=3)
    
    # Save final model
    trainer.save_model('final_robot_wave_model.pth')
    
    # Clean up
    p.disconnect()
    
    print("\nTraining completed successfully!")
    print(f"Best reward achieved: {trainer.best_reward:.2f}")
    print(f"Model saved as 'final_robot_wave_model.pth'")
    print(f"Training videos saved in 'training_videos/' directory")
    print(f"Final demonstration videos saved in 'final_demos/' directory")
    
    # Create a summary video list
    training_videos = [f for f in os.listdir('training_videos') if f.endswith('.mp4')]
    final_videos = [f for f in os.listdir('final_demos') if f.endswith('.mp4')]
    
    print(f"\nVideo Summary:")
    print(f"Training videos: {len(training_videos)}")
    print(f"Final demo videos: {len(final_videos)}")
    
    return trainer, best_policy

if __name__ == "__main__":
    trainer, policy = main()