# experiment_runner.py
# Runs comprehensive experiments comparing warm-start vs cold-start Q-learning

import os
import sys
import time
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Import Pacman modules
from pacman import ClassicGameRules
from layout import getLayout
import qlearningAgents
import searchAgents
import pacmanAgents

class ExperimentRunner:
    """
    Runs experiments comparing different learning approaches:
    1. Cold-start Q-learning (random initialization)
    2. Warm-start Q-learning (initialized from search demonstrations)  
    3. Search-only agents (no learning)
    """
    
    def __init__(self, layout_name='mediumClassic', num_training_episodes=500, num_test_episodes=10):
        self.layout_name = layout_name
        self.num_training_episodes = num_training_episodes
        self.num_test_episodes = num_test_episodes
        self.results = {}
        
    def run_qlearning_experiment(self, agent_type, warm_start_file=None, experiment_name=""):
        """Run Q-learning experiment with specified configuration"""
        print(f"\n=== Running {experiment_name} Q-Learning Experiment ===")
        
        layout = getLayout(self.layout_name)
        rules = ClassicGameRules(timeout=30)
        
        # Create Q-learning agent
        if warm_start_file and os.path.exists(warm_start_file):
            agent = qlearningAgents.PacmanQAgent(
                alpha=0.2, epsilon=0.05, gamma=0.8,
                numTraining=self.num_training_episodes,
                warmStart=True, 
                warmStartFile=warm_start_file,
                warmStartValue=100.0
            )
            print(f"Using warm start from: {warm_start_file}")
        else:
            agent = qlearningAgents.PacmanQAgent(
                alpha=0.2, epsilon=0.05, gamma=0.8,
                numTraining=self.num_training_episodes,
                warmStart=False
            )
            print("Using cold start (random initialization)")
        
        # Training phase
        print(f"Training for {self.num_training_episodes} episodes...")
        training_scores = []
        training_times = []
        
        start_time = time.time()
        
        for episode in range(self.num_training_episodes):
            game = rules.newGame(layout, [agent], [], False, False)  # No graphics during training
            game.run()
            
            score = game.state.getScore()
            training_scores.append(score)
            
            if episode % 100 == 0:
                elapsed = time.time() - start_time
                training_times.append(elapsed)
                avg_score = np.mean(training_scores[-100:]) if len(training_scores) >= 100 else np.mean(training_scores)
                print(f"Episode {episode}: Avg Score = {avg_score:.2f}, Time = {elapsed:.1f}s")
        
        total_training_time = time.time() - start_time
        
        # Testing phase (with learning disabled)
        print(f"Testing for {self.num_test_episodes} episodes...")
        agent.epsilon = 0.0  # No exploration during testing
        agent.alpha = 0.0    # No learning during testing
        
        test_scores = []
        test_start_time = time.time()
        
        for episode in range(self.num_test_episodes):
            game = rules.newGame(layout, [agent], [], False, False)
            game.run()
            score = game.state.getScore()
            test_scores.append(score)
        
        total_test_time = time.time() - test_start_time
        
        # Collect results
        performance_stats = agent.getPerformanceStats()
        
        results = {
            'experiment_name': experiment_name,
            'agent_type': agent_type,
            'warm_start': agent.warmStart,
            'warm_start_file': warm_start_file,
            'training_scores': training_scores,
            'test_scores': test_scores,
            'training_time': total_training_time,
            'test_time': total_test_time,
            'convergence_episode': performance_stats.get('convergence_episode'),
            'final_q_values_count': len(performance_stats.get('final_q_values', {})),
            'avg_training_score': np.mean(training_scores),
            'avg_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'max_test_score': np.max(test_scores),
            'min_test_score': np.min(test_scores)
        }
        
        self.results[experiment_name] = results
        return results
    
    def run_search_only_experiment(self, search_agent_class, experiment_name=""):
        """Run search-only experiment (no learning)"""
        print(f"\n=== Running {experiment_name} Search-Only Experiment ===")
        
        layout = getLayout(self.layout_name)
        rules = ClassicGameRules(timeout=30)
        
        # Create search agent
        if search_agent_class == 'ClosestDotSearchAgent':
            agent = searchAgents.ClosestDotSearchAgent()
        elif search_agent_class == 'SearchAgent_DFS':
            agent = searchAgents.SearchAgent(fn='depthFirstSearch')
        else:
            raise ValueError(f"Unknown search agent: {search_agent_class}")
        
        # Run test episodes (no training needed for search agents)
        test_scores = []
        test_times = []
        
        start_time = time.time()
        
        for episode in range(self.num_test_episodes):
            episode_start = time.time()
            game = rules.newGame(layout, [agent], [], False, False)
            game.run()
            
            score = game.state.getScore()
            episode_time = time.time() - episode_start
            
            test_scores.append(score)
            test_times.append(episode_time)
            
            print(f"Episode {episode + 1}: Score = {score}, Time = {episode_time:.2f}s")
        
        total_time = time.time() - start_time
        
        results = {
            'experiment_name': experiment_name,
            'agent_type': 'search_only',
            'warm_start': False,
            'warm_start_file': None,
            'training_scores': [],  # No training for search agents
            'test_scores': test_scores,
            'training_time': 0,
            'test_time': total_time,
            'convergence_episode': 0,  # Search agents don't converge
            'final_q_values_count': 0,
            'avg_training_score': 0,
            'avg_test_score': np.mean(test_scores),
            'std_test_score': np.std(test_scores),
            'max_test_score': np.max(test_scores),
            'min_test_score': np.min(test_scores)
        }
        
        self.results[experiment_name] = results
        return results
    
    def run_all_experiments(self):
        """Run comprehensive comparison experiments"""
        print(f"Starting comprehensive experiments on {self.layout_name}")
        
        # 1. Cold-start Q-learning
        self.run_qlearning_experiment(
            agent_type='qlearning_cold',
            warm_start_file=None,
            experiment_name='Cold Start Q-Learning'
        )
        
        # 2. Warm-start Q-learning with DFS demonstration
        dfs_file = f'demo_trajectory_dfs_{self.layout_name}.pkl'
        if os.path.exists(dfs_file):
            self.run_qlearning_experiment(
                agent_type='qlearning_warm_dfs',
                warm_start_file=dfs_file,
                experiment_name='Warm Start Q-Learning (DFS)'
            )
        else:
            print(f"Warning: {dfs_file} not found. Skipping DFS warm start experiment.")
        
        # 3. Warm-start Q-learning with ClosestDot demonstration
        closest_file = f'demo_trajectory_closest_{self.layout_name}.pkl'
        if os.path.exists(closest_file):
            self.run_qlearning_experiment(
                agent_type='qlearning_warm_closest',
                warm_start_file=closest_file,
                experiment_name='Warm Start Q-Learning (ClosestDot)'
            )
        else:
            print(f"Warning: {closest_file} not found. Skipping ClosestDot warm start experiment.")
        
        # 4. Search-only baselines
        try:
            self.run_search_only_experiment(
                search_agent_class='ClosestDotSearchAgent',
                experiment_name='Search Only (ClosestDot)'
            )
        except Exception as e:
            print(f"Error running ClosestDot search experiment: {e}")
        
        # 5. Warm-start with existing demo file
        if os.path.exists('demo_trajectory.pkl'):
            self.run_qlearning_experiment(
                agent_type='qlearning_warm_demo',
                warm_start_file='demo_trajectory.pkl',
                experiment_name='Warm Start Q-Learning (Demo)'
            )
    
    def save_results(self, filename='experiment_results.pkl'):
        """Save results to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(self.results, f)
        print(f"Results saved to {filename}")
    
    def generate_report(self):
        """Generate comparison report"""
        if not self.results:
            print("No results to report!")
            return
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPARISON REPORT")
        print("="*80)
        
        # Summary table
        print(f"\nLayout: {self.layout_name}")
        print(f"Training Episodes: {self.num_training_episodes}")
        print(f"Test Episodes: {self.num_test_episodes}")
        print()
        
        print(f"{'Experiment':<35} {'Avg Test Score':<15} {'Std Dev':<10} {'Convergence':<12} {'Training Time':<15}")
        print("-" * 90)
        
        for name, result in self.results.items():
            convergence = result['convergence_episode'] if result['convergence_episode'] else 'N/A'
            training_time = f"{result['training_time']:.1f}s" if result['training_time'] > 0 else 'N/A'
            
            print(f"{name:<35} {result['avg_test_score']:<15.2f} {result['std_test_score']:<10.2f} "
                  f"{convergence:<12} {training_time:<15}")
        
        # Detailed analysis
        print("\n" + "="*50)
        print("DETAILED ANALYSIS")
        print("="*50)
        
        # Find best performers
        qlearning_results = {name: result for name, result in self.results.items() 
                           if 'Q-Learning' in name}
        
        if qlearning_results:
            best_ql = max(qlearning_results.values(), key=lambda x: x['avg_test_score'])
            print(f"\nBest Q-Learning Agent: {best_ql['experiment_name']}")
            print(f"  Average Test Score: {best_ql['avg_test_score']:.2f} ± {best_ql['std_test_score']:.2f}")
            
            # Convergence analysis
            converged = [r for r in qlearning_results.values() if r['convergence_episode']]
            if converged:
                fastest_convergence = min(converged, key=lambda x: x['convergence_episode'])
                print(f"\nFastest Convergence: {fastest_convergence['experiment_name']}")
                print(f"  Converged at episode: {fastest_convergence['convergence_episode']}")
        
        # Warm start vs cold start comparison
        cold_start = next((r for r in self.results.values() if not r['warm_start'] and 'Q-Learning' in r['experiment_name']), None)
        warm_starts = [r for r in self.results.values() if r['warm_start']]
        
        if cold_start and warm_starts:
            print(f"\n{'Warm Start Comparison:'}")
            print(f"Cold Start Score: {cold_start['avg_test_score']:.2f}")
            for warm in warm_starts:
                improvement = ((warm['avg_test_score'] - cold_start['avg_test_score']) / 
                             abs(cold_start['avg_test_score'])) * 100
                print(f"{warm['experiment_name']}: {warm['avg_test_score']:.2f} "
                      f"({improvement:+.1f}% vs cold start)")
    
    def plot_results(self, save_plots=True):
        """Generate visualization plots"""
        if not self.results:
            print("No results to plot!")
            return
        
        # Plot 1: Average test scores comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Q-Learning Comparison Results ({self.layout_name})', fontsize=16)
        
        # Bar plot of average test scores
        names = list(self.results.keys())
        scores = [self.results[name]['avg_test_score'] for name in names]
        errors = [self.results[name]['std_test_score'] for name in names]
        
        axes[0, 0].bar(range(len(names)), scores, yerr=errors, capsize=5, alpha=0.7)
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average Test Score')
        axes[0, 0].set_title('Average Test Performance')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Training curves for Q-learning agents
        for name, result in self.results.items():
            if result['training_scores']:
                # Smooth the training curve
                scores = result['training_scores']
                if len(scores) > 100:
                    window = 50
                    smoothed = np.convolve(scores, np.ones(window)/window, mode='valid')
                    x_smooth = range(window-1, len(scores))
                    axes[0, 1].plot(x_smooth, smoothed, label=name, alpha=0.8)
                else:
                    axes[0, 1].plot(scores, label=name, alpha=0.8)
        
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Training Progress (Smoothed)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Convergence comparison
        convergence_data = [(name, result['convergence_episode']) 
                          for name, result in self.results.items() 
                          if result['convergence_episode']]
        
        if convergence_data:
            names_conv, episodes = zip(*convergence_data)
            axes[1, 0].bar(range(len(names_conv)), episodes, alpha=0.7, color='green')
            axes[1, 0].set_xticks(range(len(names_conv)))
            axes[1, 0].set_xticklabels(names_conv, rotation=45, ha='right')
            axes[1, 0].set_ylabel('Episodes to Convergence')
            axes[1, 0].set_title('Convergence Speed')
            axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Training time comparison
        times = [self.results[name]['training_time'] for name in names if self.results[name]['training_time'] > 0]
        time_names = [name for name in names if self.results[name]['training_time'] > 0]
        
        if times:
            axes[1, 1].bar(range(len(time_names)), times, alpha=0.7, color='orange')
            axes[1, 1].set_xticks(range(len(time_names)))
            axes[1, 1].set_xticklabels(time_names, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Training Time (seconds)')
            axes[1, 1].set_title('Training Time Comparison')
            axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'qlearning_comparison_{self.layout_name}.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved as qlearning_comparison_{self.layout_name}.png")
        
        plt.show()

def main():
    """Run the complete experiment suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Q-learning comparison experiments')
    parser.add_argument('--layout', default='mediumClassic', help='Layout to use for experiments')
    parser.add_argument('--episodes', type=int, default=300, help='Number of training episodes')
    parser.add_argument('--test-episodes', type=int, default=10, help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner(
        layout_name=args.layout,
        num_training_episodes=args.episodes,
        num_test_episodes=args.test_episodes
    )
    
    # Generate trajectories first if they don't exist
    trajectory_files = [
        f'demo_trajectory_dfs_{args.layout}.pkl',
        f'demo_trajectory_closest_{args.layout}.pkl'
    ]
    
    missing_files = [f for f in trajectory_files if not os.path.exists(f)]
    
    if missing_files:
        print("Missing trajectory files. Generating them first...")
        from trajectory_generator import TrajectoryGenerator
        gen = TrajectoryGenerator()
        
        if f'demo_trajectory_dfs_{args.layout}.pkl' in missing_files:
            try:
                dfs_traj = gen.generate_dfs_trajectory(args.layout)
                gen.save_trajectory(dfs_traj, f'demo_trajectory_dfs_{args.layout}.pkl')
            except Exception as e:
                print(f"Failed to generate DFS trajectory: {e}")
        
        if f'demo_trajectory_closest_{args.layout}.pkl' in missing_files:
            try:
                closest_traj = gen.generate_closest_dot_trajectory(args.layout)
                gen.save_trajectory(closest_traj, f'demo_trajectory_closest_{args.layout}.pkl')
            except Exception as e:
                print(f"Failed to generate ClosestDot trajectory: {e}")
    
    # Run experiments
    print(f"Starting experiments with {args.episodes} training episodes on {args.layout}")
    runner.run_all_experiments()
    
    # Generate report and save results
    runner.generate_report()
    runner.save_results(f'experiment_results_{args.layout}.pkl')
    runner.plot_results()

if __name__ == '__main__':
    main()
