# demo_single_comparison.py
# Simple demonstration of individual warm-start vs cold-start comparison

def run_single_comparison():
    """Run a single comparison to demonstrate the difference"""
    
    print("="*70)
    print("DEMONSTRATION: Single Warm-Start vs Cold-Start Comparison")
    print("="*70)
    
    # Configuration
    layout = 'smallGrid'
    training_episodes = 30
    test_episodes = 5
    python_exec = "C:/Users/eldad/OneDrive/Desktop/pacman-merged/.venv/Scripts/python.exe"
    
    print(f"Layout: {layout}")
    print(f"Training Episodes: {training_episodes}")
    print(f"Test Episodes: {test_episodes}")
    print()
    
    # 1. Run cold start
    print("1. Running Cold-Start Q-Learning...")
    print("-" * 50)
    
    import subprocess
    cmd = [
        python_exec, "pacman.py",
        "-p", "PacmanQAgent",
        "-l", layout,
        "-x", str(training_episodes),
        "-n", str(training_episodes + test_episodes)
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Running...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Extract average score
            output = result.stdout
            for line in output.split('\n'):
                if "Average Score:" in line:
                    cold_score = float(line.split("Average Score:")[1].strip())
                    print(f"Cold Start Result: {cold_score}")
                    break
        else:
            print("Cold start failed!")
            cold_score = None
    except Exception as e:
        print(f"Error running cold start: {e}")
        cold_score = None
    
    # 2. Run warm start
    print("\n2. Running Warm-Start Q-Learning...")
    print("-" * 50)
    
    warm_file = f'demo_trajectory_closest_{layout}.pkl'
    
    if not os.path.exists(warm_file):
        print(f"Warm-start file {warm_file} not found. Skipping warm start.")
        return
    
    cmd_warm = [
        python_exec, "pacman.py",
        "-p", "PacmanQAgent",
        "-l", layout,
        "-x", str(training_episodes),
        "-n", str(training_episodes + test_episodes),
        "-a", f"warmStart=True,warmStartFile={warm_file}"
    ]
    
    print(f"Command: {' '.join(cmd_warm)}")
    print("Running...")
    
    try:
        result = subprocess.run(cmd_warm, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Extract average score
            output = result.stdout
            for line in output.split('\n'):
                if "Average Score:" in line:
                    warm_score = float(line.split("Average Score:")[1].strip())
                    print(f"Warm Start Result: {warm_score}")
                    break
        else:
            print("Warm start failed!")
            warm_score = None
    except Exception as e:
        print(f"Error running warm start: {e}")
        warm_score = None
    
    # 3. Comparison
    print("\n3. Comparison Results")
    print("="*50)
    
    if cold_score is not None and warm_score is not None:
        improvement = warm_score - cold_score
        improvement_pct = (improvement / abs(cold_score)) * 100 if cold_score != 0 else 0
        
        print(f"Cold Start Score:  {cold_score:.1f}")
        print(f"Warm Start Score:  {warm_score:.1f}")
        print(f"Improvement:       {improvement:+.1f} points ({improvement_pct:+.1f}%)")
        
        if improvement > 0:
            print("✓ Warm start IMPROVED performance!")
        else:
            print("✗ Warm start did not improve performance in this run.")
        
        print("\nNote: Results may vary between runs due to randomness in Q-learning.")
        print("The comprehensive experiment shows consistent improvements across multiple runs.")
    else:
        print("Could not complete comparison - one or both experiments failed.")

if __name__ == '__main__':
    import os
    run_single_comparison()
