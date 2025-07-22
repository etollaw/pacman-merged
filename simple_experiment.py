# simple_experiment.py
# Simple experiment to demonstrate warm-start vs cold-start Q-learning

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Run a basic Q-learning experiment
def run_simple_experiment():
    """Run a simple comparison between cold-start and warm-start Q-learning"""
    
    print("=== Simple Q-Learning Comparison Experiment ===")
    
    # Test different layouts and configurations
    configs = [
        {
            'layout': 'smallGrid',
            'episodes': 200,
            'test_episodes': 5,
            'warm_start_file': 'demo_trajectory_dfs_smallGrid.pkl'
        }
    ]
    
    results = {}
    
    for config in configs:
        layout = config['layout']
        episodes = config['episodes']
        test_episodes = config['test_episodes']
        warm_start_file = config['warm_start_file']
        
        print(f"\nTesting on {layout} with {episodes} training episodes")
        
        # Run cold-start experiment
        print("Running cold-start Q-learning...")
        try:
            cmd = f'python pacman.py -p PacmanQAgent -x {episodes} -n {episodes + test_episodes} -l {layout}'
            os.system(cmd)
            print("Cold-start completed")
        except Exception as e:
            print(f"Cold-start failed: {e}")
        
        # Run warm-start experiment if trajectory exists
        if os.path.exists(warm_start_file):
            print(f"Running warm-start Q-learning with {warm_start_file}...")
            try:
                cmd = f'python pacman.py -p PacmanQAgent -x {episodes} -n {episodes + test_episodes} -l {layout} -a warmStart=True,warmStartFile={warm_start_file}'
                os.system(cmd)
                print("Warm-start completed")
            except Exception as e:
                print(f"Warm-start failed: {e}")
        else:
            print(f"Warm-start file {warm_start_file} not found")
        
        # Run search-only baseline
        print("Running search-only baseline...")
        try:
            cmd = f'python pacman.py -p SearchAgent -l {layout} -n {test_episodes}'
            os.system(cmd)
            print("Search-only completed")
        except Exception as e:
            print(f"Search-only failed: {e}")

if __name__ == '__main__':
    run_simple_experiment()
