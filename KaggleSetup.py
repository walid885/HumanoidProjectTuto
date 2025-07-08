import sys
import subprocess
import os
import time
from IPython.display import clear_output

print("🚀 Starting Kaggle Humanoid Training Environment Setup...")
print("=" * 60)

# ===================================================================
# STEP 1: ENVIRONMENT VERIFICATION
# ===================================================================
print("📋 STEP 1: Verifying Kaggle Environment...")

# Check if we're in Kaggle
if os.path.exists('/kaggle/input'):
    print("✅ Running in Kaggle environment")
    print(f"Working directory: {os.getcwd()}")
    print(f"Available space: {os.statvfs('/kaggle/working').f_bavail * os.statvfs('/kaggle/working').f_frsize / (1024**3):.1f} GB")
else:
    print("❌ Not in Kaggle environment!")
    sys.exit(1)

# Check GPU availability
print("\n🔍 Checking GPU availability...")
try:
    gpu_info = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if gpu_info.returncode == 0:
        print("✅ GPU detected:")
        # Extract GPU info
        lines = gpu_info.stdout.split('\n')
        for line in lines:
            if 'Tesla' in line or 'T4' in line or 'P100' in line:
                print(f"   {line.strip()}")
    else:
        print("❌ No GPU detected! Enable GPU in Settings → Accelerator")
        print("   Go to Settings (right panel) → Accelerator → Select 'GPU T4 x2'")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error checking GPU: {e}")
    sys.exit(1)

print("\n" + "=" * 60)

# ===================================================================
# STEP 2: PACKAGE INSTALLATION
# ===================================================================
print("📦 STEP 2: Installing Required Packages...")

# Function to install packages with progress
def install_package(package_name, pip_args): # Changed pip_name to pip_args to reflect it's a list
    print(f"Installing {package_name}...")
    try:
        command = [sys.executable, '-m', 'pip', 'install', '-q'] + pip_args
        result = subprocess.run(command,
                                capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {package_name} installed successfully")
        else:
            print(f"❌ Failed to install {package_name}")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Installation of {package_name} timed out")
        return False
    except Exception as e:
        print(f"❌ Error installing {package_name}: {e}")
        return False
    return True

# Install packages in order
packages = [
    ("PyTorch with CUDA", ["torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]),
    ("PyBullet", ["pybullet"]),
    ("Stable Baselines3", ["stable-baselines3[extra]"]),
    ("Gymnasium", ["gymnasium"]),
    ("TensorBoard", ["tensorboard"]),
    ("Weights & Biases", ["wandb"]),
    ("Additional ML libraries", ["matplotlib", "seaborn", "pandas", "numpy", "opencv-python"]),
    ("MuJoCo", ["mujoco"]),
    ("Plotting utilities", ["plotly"])
]

failed_packages = []
for package_name, pip_args in packages: # Changed pip_command to pip_args
    if not install_package(package_name, pip_args):
        failed_packages.append(package_name)
    time.sleep(1)  # Small delay between installations

if failed_packages:
    print(f"\n❌ Failed to install: {', '.join(failed_packages)}")
    print("You may need to install these manually or continue without them")
else:
    print("\n✅ All packages installed successfully!")

print("\n" + "=" * 60)

# ===================================================================
# STEP 3: IMPORT AND COMPATIBILITY TESTING
# ===================================================================
print("🧪 STEP 3: Testing Package Imports and Compatibility...")

# Test imports
imports_to_test = [
    ("torch", "PyTorch"),
    ("pybullet", "PyBullet"),
    ("stable_baselines3", "Stable Baselines3"),
    ("gymnasium", "Gymnasium"),
    ("tensorboard", "TensorBoard"),
    ("matplotlib", "Matplotlib"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas")
]

failed_imports = []
for module, name in imports_to_test:
    try:
        __import__(module)
        print(f"✅ {name} import: OK")
    except ImportError as e:
        print(f"❌ {name} import: FAILED - {e}")
        failed_imports.append(name)

if failed_imports:
    print(f"\n❌ Failed imports: {', '.join(failed_imports)}")
    print("Some functionality may not work correctly")

print("\n" + "=" * 60)

# ===================================================================
# STEP 4: GPU AND PYTORCH TESTING
# ===================================================================
print("🔥 STEP 4: Testing GPU and PyTorch Integration...")

import torch
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Test GPU tensor operations
    print("\n🧮 Testing GPU tensor operations...")
    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # Perform matrix multiplication
        start_time = time.time()
        z = torch.mm(x, y)
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU tensor operations: OK")
        print(f"  Matrix multiplication (1000x1000): {gpu_time:.4f}s")
        
        # Test CPU vs GPU speed
        x_cpu = torch.randn(1000, 1000)
        y_cpu = torch.randn(1000, 1000)
        start_time = time.time()
        z_cpu = torch.mm(x_cpu, y_cpu)
        cpu_time = time.time() - start_time
        
        print(f"  CPU time: {cpu_time:.4f}s")
        print(f"  GPU speedup: {cpu_time/gpu_time:.1f}x")
        
    except Exception as e:
        print(f"❌ GPU tensor operations failed: {e}")
else:
    print("❌ CUDA not available - check GPU settings!")

print("\n" + "=" * 60)

# ===================================================================
# STEP 5: PYBULLET TESTING
# ===================================================================
print("🎯 STEP 5: Testing PyBullet Physics Engine...")

try:
    import pybullet as p
    import pybullet_data
    
    print(f"PyBullet version: {p.__version__}")
    
    # Test basic PyBullet functionality
    print("Testing PyBullet connection...")
    physics_client = p.connect(p.DIRECT)
    print(f"✅ PyBullet connection: OK (Client ID: {physics_client})")
    
    # Set search path for URDF files
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print("✅ URDF search path set")
    
    # Test loading basic objects
    print("Testing URDF loading...")
    try:
        plane_id = p.loadURDF("plane.urdf")
        print(f"✅ Plane loaded (ID: {plane_id})")
        
        cube_id = p.loadURDF("cube.urdf", [0, 0, 1])
        print(f"✅ Cube loaded (ID: {cube_id})")
        
        # Test humanoid loading
        try:
            humanoid_id = p.loadURDF("humanoid/nao.urdf", [0, 0, 0.5])
            print(f"✅ Humanoid (NAO) loaded (ID: {humanoid_id})")
            
            # Get joint info
            num_joints = p.getNumJoints(humanoid_id)
            print(f"    Number of joints: {num_joints}")
            
        except Exception as e:
            print(f"⚠️ Humanoid loading failed: {e}")
            print("    Will use alternative humanoid model")
            
    except Exception as e:
        print(f"❌ URDF loading failed: {e}")
    
    # Test basic physics simulation
    print("Testing physics simulation...")
    p.setGravity(0, 0, -9.81)
    
    # Run a few simulation steps
    for i in range(10):
        p.stepSimulation()
    
    print("✅ Physics simulation: OK")
    
    # Disconnect
    p.disconnect()
    print("✅ PyBullet cleanup: OK")
    
except Exception as e:
    print(f"❌ PyBullet testing failed: {e}")

print("\n" + "=" * 60)

# ===================================================================
# STEP 6: STABLE BASELINES3 TESTING
# ===================================================================
print("🤖 STEP 6: Testing Stable Baselines3 RL Framework...")

try:
    import stable_baselines3 as sb3
    import gymnasium as gym
    
    print(f"Stable Baselines3 version: {sb3.__version__}")
    print(f"Gymnasium version: {gym.__version__}")
    
    # Test basic environment
    print("Testing basic Gymnasium environment...")
    env = gym.make('CartPole-v1')
    print(f"✅ CartPole environment created")
    print(f"    Observation space: {env.observation_space}")
    print(f"    Action space: {env.action_space}")
    
    # Test basic PPO model
    print("Testing PPO model creation...")
    model = sb3.PPO('MlpPolicy', env, verbose=0, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ PPO model created")
    print(f"    Device: {model.device}")
    
    # Test short training
    print("Testing short training run...")
    model.learn(total_timesteps=1000)
    print("✅ Short training: OK")
    
    # Test model prediction
    obs, _ = env.reset()
    action, _ = model.predict(obs)
    print(f"✅ Model prediction: OK (action: {action})")
    
    env.close()
    
except Exception as e:
    print(f"❌ Stable Baselines3 testing failed: {e}")

print("\n" + "=" * 60)

# ===================================================================
# STEP 7: CUSTOM ENVIRONMENT TESTING
# ===================================================================
print("🏗️ STEP 7: Testing Custom Humanoid Environment...")

try:
    # Create a simple test environment
    class SimpleHumanoidEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
            self.physics_client = None
            
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if self.physics_client is None:
                self.physics_client = p.connect(p.DIRECT)
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            p.resetSimulation()
            p.setGravity(0, 0, -9.81)
            p.loadURDF("plane.urdf")
            
            observation = np.random.randn(20).astype(np.float32)
            return observation, {}
            
        def step(self, action):
            observation = np.random.randn(20).astype(np.float32)
            reward = np.random.randn()
            terminated = False
            truncated = False
            info = {}
            return observation, reward, terminated, truncated, info
            
        def close(self):
            if self.physics_client is not None:
                p.disconnect(self.physics_client)
    
    # Test custom environment
    print("Creating custom humanoid environment...")
    custom_env = SimpleHumanoidEnv()
    obs, _ = custom_env.reset()
    print(f"✅ Custom environment reset: OK")
    print(f"    Observation shape: {obs.shape}")
    
    # Test environment step
    action = custom_env.action_space.sample()
    obs, reward, terminated, truncated, info = custom_env.step(action)
    print(f"✅ Environment step: OK")
    print(f"    Reward: {reward:.3f}")
    
    custom_env.close()
    
except Exception as e:
    print(f"❌ Custom environment testing failed: {e}")

print("\n" + "=" * 60)

# ===================================================================
# STEP 8: FINAL SYSTEM REPORT
# ===================================================================
print("📊 STEP 8: Final System Report")
print("=" * 60)

print("🖥️  HARDWARE:")
print(f"    Platform: Kaggle")
print(f"    GPU: {'Available' if torch.cuda.is_available() else 'Not Available'}")
if torch.cuda.is_available():
    print(f"    GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\n📚 SOFTWARE:")
print(f"    Python: {sys.version.split()[0]}")
print(f"    PyTorch: {torch.__version__}")
try:
    print(f"    PyBullet: {p.__version__}")
except:
    print(f"    PyBullet: Not available")
try:
    print(f"    Stable Baselines3: {sb3.__version__}")
except:
    print(f"    Stable Baselines3: Not available")
try:
    print(f"    Gymnasium: {gym.__version__}")
except:
    print(f"    Gymnasium: Not available")

print(f"\n💾 STORAGE:")
print(f"    Working directory: /kaggle/working")
print(f"    Available space: {os.statvfs('/kaggle/working').f_bavail * os.statvfs('/kaggle/working').f_frsize / (1024**3):.1f} GB")

print(f"\n⏱️  RESOURCE LIMITS:")
print(f"    GPU quota: 30 hours/week")
print(f"    Session limit: 12 hours maximum")
print(f"    Recommended training blocks: 6 hours")

print("\n🎯 RECOMMENDED NEXT STEPS:")
print("1. Start with simple wave gesture environment")
print("2. Use PPO algorithm with reduced batch size (128-256)")
print("3. Train in 6-hour blocks to maximize quota usage")
print("4. Save checkpoints every 10,000 steps")
print("5. Monitor training with TensorBoard")

print("\n" + "=" * 60)
print("🎉 SETUP COMPLETE!")
print("Your Kaggle environment is ready for humanoid training!")
print("=" * 60)

# Save environment info to file
with open('/kaggle/working/environment_info.txt', 'w') as f:
    f.write("Kaggle Humanoid Training Environment Setup\n")
    f.write("=" * 50 + "\n")
    f.write(f"Setup completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Python version: {sys.version}\n")
    f.write(f"PyTorch version: {torch.__version__}\n")
    f.write(f"CUDA available: {torch.cuda.is_available()}\n")
    if torch.cuda.is_available():
        f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
    f.write(f"Working directory: {os.getcwd()}\n")

print("📝 Environment info saved to: /kaggle/working/environment_info.txt")