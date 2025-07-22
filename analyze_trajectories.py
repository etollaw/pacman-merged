# analyze_trajectories.py
# Analyzes existing trajectory files and provides insights

import pickle
import sys
from collections import Counter

def analyze_trajectory(filename):
    """Analyze a trajectory file and print statistics"""
    try:
        with open(filename, 'rb') as f:
            trajectory = pickle.load(f)
        
        print(f"\n=== Analysis of {filename} ===")
        print(f"Total steps: {len(trajectory)}")
        
        if not trajectory:
            print("Empty trajectory!")
            return
        
        # Analyze actions
        actions = [action for state, action in trajectory]
        action_counts = Counter(actions)
        
        print(f"Action distribution:")
        for action, count in action_counts.most_common():
            percentage = (count / len(actions)) * 100
            print(f"  {action}: {count} ({percentage:.1f}%)")
        
        # Sample some state-action pairs
        print(f"\nFirst 5 state-action pairs:")
        for i, (state, action) in enumerate(trajectory[:5]):
            print(f"  Step {i+1}: {action}")
            
        print(f"\nLast 5 state-action pairs:")
        for i, (state, action) in enumerate(trajectory[-5:]):
            print(f"  Step {len(trajectory)-5+i+1}: {action}")
        
        return {
            'length': len(trajectory),
            'actions': dict(action_counts),
            'unique_actions': len(action_counts)
        }
        
    except FileNotFoundError:
        print(f"File {filename} not found!")
        return None
    except Exception as e:
        print(f"Error analyzing {filename}: {e}")
        return None

def main():
    """Analyze all available trajectory files"""
    trajectory_files = [
        'demo_trajectory.pkl',
        'demo_trajectory_dfs_mediumClassic.pkl',
        'demo_trajectory_closest_mediumClassic.pkl',
        'demo_trajectory_random_mediumClassic.pkl',
        'demo_trajectory_dfs_smallGrid.pkl',
        'demo_trajectory_closest_smallGrid.pkl'
    ]
    
    results = {}
    
    for filename in trajectory_files:
        result = analyze_trajectory(filename)
        if result:
            results[filename] = result
    
    if results:
        print(f"\n{'='*60}")
        print("TRAJECTORY COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        print(f"{'File':<40} {'Length':<10} {'Unique Actions':<15}")
        print("-" * 65)
        
        for filename, stats in results.items():
            print(f"{filename:<40} {stats['length']:<10} {stats['unique_actions']:<15}")

if __name__ == '__main__':
    main()
