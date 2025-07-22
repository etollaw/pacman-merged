# trajectory_generator.py
# Simplified trajectory generation for warm-start demonstrations

import pickle
import random
from game import Directions

class SimpleTrajectoryGenerator:
    """
    Simple trajectory generator that creates demonstration data
    without running full Pacman games (which can be complex to set up).
    """
    
    def generate_search_like_trajectory(self, num_steps=200, strategy='dfs'):
        """
        Generate a trajectory that mimics a search algorithm.
        This creates (state_hash, action) pairs that can be used for warm-starting.
        """
        trajectory = []
        
        if strategy == 'dfs':
            # DFS-like: prefer going deep in one direction
            actions = [Directions.EAST, Directions.NORTH, Directions.WEST, Directions.SOUTH]
            current_dir = 0
            
            for step in range(num_steps):
                # Create a simple state representation (just step number as hash)
                state = f"state_{step}"
                
                # DFS-like behavior: stick to one direction for a while, then change
                if step % 20 == 0 and step > 0:  # Change direction every 20 steps
                    current_dir = (current_dir + 1) % 4
                
                action = actions[current_dir]
                
                # Add some randomness
                if random.random() < 0.1:  # 10% chance to pick random action
                    action = random.choice(actions)
                
                trajectory.append((state, action))
        
        elif strategy == 'closest':
            # ClosestDot-like: more varied movement pattern
            actions = [Directions.EAST, Directions.NORTH, Directions.WEST, Directions.SOUTH]
            
            for step in range(num_steps):
                state = f"state_{step}"
                
                # Simulate "going to closest food" behavior
                if step < num_steps // 4:
                    # First quarter: mostly East
                    action = random.choices(actions, weights=[0.5, 0.2, 0.1, 0.2])[0]
                elif step < num_steps // 2:
                    # Second quarter: mostly North
                    action = random.choices(actions, weights=[0.2, 0.5, 0.2, 0.1])[0]
                elif step < 3 * num_steps // 4:
                    # Third quarter: mostly West
                    action = random.choices(actions, weights=[0.1, 0.2, 0.5, 0.2])[0]
                else:
                    # Final quarter: mostly South
                    action = random.choices(actions, weights=[0.2, 0.1, 0.2, 0.5])[0]
                
                trajectory.append((state, action))
        
        elif strategy == 'random':
            # Random baseline
            actions = [Directions.EAST, Directions.NORTH, Directions.WEST, Directions.SOUTH]
            
            for step in range(num_steps):
                state = f"state_{step}"
                action = random.choice(actions)
                trajectory.append((state, action))
        
        return trajectory
    
    def save_trajectory(self, trajectory, filename):
        """Save trajectory to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(trajectory, f)
        print(f"Generated and saved trajectory with {len(trajectory)} steps to {filename}")

def main():
    """Generate demonstration trajectories for warm-starting"""
    generator = SimpleTrajectoryGenerator()
    
    layouts = ['mediumClassic', 'smallGrid']
    
    for layout in layouts:
        print(f"\n=== Generating trajectories for {layout} ===")
        
        # Generate DFS-like trajectory
        dfs_traj = generator.generate_search_like_trajectory(num_steps=200, strategy='dfs')
        generator.save_trajectory(dfs_traj, f'demo_trajectory_dfs_{layout}.pkl')
        
        # Generate ClosestDot-like trajectory
        closest_traj = generator.generate_search_like_trajectory(num_steps=200, strategy='closest')
        generator.save_trajectory(closest_traj, f'demo_trajectory_closest_{layout}.pkl')
        
        # Generate random trajectory for comparison
        random_traj = generator.generate_search_like_trajectory(num_steps=200, strategy='random')
        generator.save_trajectory(random_traj, f'demo_trajectory_random_{layout}.pkl')

if __name__ == '__main__':
    main()
