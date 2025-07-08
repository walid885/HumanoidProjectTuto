import pybullet as p
import pybullet_data
import time

print("\n--- Verifying PyBullet Functionality ---")
try:
    # 1. Connect to the physics server in DIRECT mode (no GUI needed)
    physicsClient = p.connect(p.DIRECT)
    print(f"PyBullet connection established. Client ID: {physicsClient}")

    # 2. Set additional search path for URDF files (like 'plane.urdf')
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    print("PyBullet data path set.")

    # 3. Load a simple object (e.g., a plane)
    planeId = p.loadURDF("plane.urdf")
    print(f"Plane loaded successfully (ID: {planeId}).")

    # 4. Set gravity
    p.setGravity(0, 0, -9.81)
    print("Gravity set.")

    # 5. Run a few simulation steps
    for _ in range(5):
        p.stepSimulation()
    print("5 simulation steps executed successfully.")

    # 6. Disconnect from the physics server
    p.disconnect()
    print("PyBullet disconnected successfully.")

    print("✅ PyBullet is working correctly.")

except Exception as e:
    print(f"❌ PyBullet functionality test failed: {e}")
    print("   This indicates a deeper issue with PyBullet setup.")