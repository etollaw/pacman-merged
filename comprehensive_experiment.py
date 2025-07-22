# comprehensive_experiment.py
# Comprehensive experiment runner for Q-learning comparison

import subprocess
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class PacmanExperiment:
    """Run comprehensive Pacman Q-learning experiments"""
    
    def __init__(self, python_executable=None):
        if python_executable:
            self.python_exec = python_executable
        else:
            # Try to detect the correct python executable
            potential_paths = [
                "C:/Users/eldad/OneDrive/Desktop/pacman-merged/.venv/Scripts/python.exe",
                "python",
                "python3"
            ]
            self.python_exec = None
            for path in potential_paths:
                try:
                    result = subprocess.run([path, "--version"], capture_output=True, text=True)
                    if result.returncode == 0:
                        self.python_exec = path
                        break
                except:
                    continue
            
            if not self.python_exec:
                self.python_exec = "python"
    
    def run_qlearning_experiment(self, layout='smallGrid', training_episodes=100, 
                                test_episodes=10, warm_start_file=None, experiment_name=""):
        """Run a Q-learning experiment and parse results"""
        
        print(f"\n=== {experiment_name} ===")
        
        # Build command
        cmd = [
            self.python_exec, "pacman.py",
            "-p", "PacmanQAgent",
            "-l", layout,
            "-x", str(training_episodes),  # Training episodes
            "-n", str(training_episodes + test_episodes),  # Total episodes
            "-q"  # Quiet mode
        ]
        
        # Add warm start parameters if specified
        if warm_start_file and os.path.exists(warm_start_file):
            cmd.extend(["-a", f"warmStart=True,warmStartFile={warm_start_file},warmStartValue=50.0"])
            print(f"Using warm start from: {warm_start_file}")
        else:
            print("Using cold start (random initialization)")
        
        try:
            # Run the experiment
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Error running experiment: {result.stderr}")
                return None
            
            # Parse output
            output = result.stdout
            return self.parse_qlearning_output(output, experiment_name)
            
        except subprocess.TimeoutExpired:
            print(f"Experiment timed out after 300 seconds")
            return None
        except Exception as e:
            print(f"Experiment failed: {e}")
            return None
    
    def run_search_experiment(self, layout='smallGrid', test_episodes=10, search_type='SearchAgent'):
        """Run a search-only experiment"""
        
        print(f"\n=== Search Only ({search_type}) ===")
        
        cmd = [
            self.python_exec, "pacman.py",
            "-p", search_type,
            "-l", layout,
            "-n", str(test_episodes),
            "-q"
        ]
        
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"Error running search experiment: {result.stderr}")
                return None
            
            output = result.stdout
            return self.parse_search_output(output, f"Search Only ({search_type})")
            
        except Exception as e:
            print(f"Search experiment failed: {e}")
            return None
    
    def parse_qlearning_output(self, output, experiment_name):
        """Parse Q-learning experiment output"""
        lines = output.split('\\n')
        
        # Extract key metrics
        convergence_episode = None
        average_score = None
        scores = []
        win_rate = None
        
        for line in lines:
            # Look for convergence
            if "[Convergence]" in line:
                match = re.search(r"episode (\\d+)", line)
                if match:
                    convergence_episode = int(match.group(1))
            
            # Look for average score
            if "Average Score:" in line:
                match = re.search(r"Average Score: ([\\-\\d\\.]+)", line)
                if match:
                    average_score = float(match.group(1))
            
            # Look for scores
            if "Scores:" in line:
                scores_match = re.findall(r"([\\-\\d\\.]+)", line)
                scores = [float(s) for s in scores_match if s.replace('-', '').replace('.', '').isdigit()]
            
            # Look for win rate
            if "Win Rate:" in line:
                match = re.search(r"(\\d+)/(\\d+)", line)
                if match:
                    wins = int(match.group(1))
                    total = int(match.group(2))
                    win_rate = wins / total if total > 0 else 0
        
        result = {
            'experiment_name': experiment_name,
            'convergence_episode': convergence_episode,
            'average_score': average_score,
            'scores': scores,
            'win_rate': win_rate,
            'std_score': np.std(scores) if scores else 0,
            'max_score': max(scores) if scores else None,
            'min_score': min(scores) if scores else None
        }
        
        print(f"Results: Avg Score = {average_score}, Win Rate = {win_rate}, Convergence = {convergence_episode}")
        return result
    
    def parse_search_output(self, output, experiment_name):
        """Parse search experiment output"""
        lines = output.split('\\n')
        
        average_score = None
        scores = []
        win_rate = None
        
        for line in lines:
            if "Average Score:" in line:
                match = re.search(r"Average Score: ([\\-\\d\\.]+)", line)
                if match:
                    average_score = float(match.group(1))
            
            if "Scores:" in line:
                scores_match = re.findall(r"([\\-\\d\\.]+)", line)
                scores = [float(s) for s in scores_match if s.replace('-', '').replace('.', '').isdigit()]
            
            if "Win Rate:" in line:
                match = re.search(r"(\\d+)/(\\d+)", line)
                if match:
                    wins = int(match.group(1))
                    total = int(match.group(2))
                    win_rate = wins / total if total > 0 else 0
        
        result = {
            'experiment_name': experiment_name,
            'convergence_episode': 0,  # Search doesn't converge
            'average_score': average_score,
            'scores': scores,
            'win_rate': win_rate,
            'std_score': np.std(scores) if scores else 0,
            'max_score': max(scores) if scores else None,
            'min_score': min(scores) if scores else None
        }
        
        print(f"Results: Avg Score = {average_score}, Win Rate = {win_rate}")
        return result

def main():
    """Run comprehensive experiments"""
    
    print("Starting Comprehensive Q-Learning vs Search Comparison")
    print("="*60)
    
    experiment = PacmanExperiment()
    results = []
    
    # Experiment parameters
    layout = 'smallGrid'  # Start with simple layout
    training_episodes = 50  # Reasonable number for quick testing
    test_episodes = 5
    
    # 1. Cold-start Q-learning
    result = experiment.run_qlearning_experiment(
        layout=layout,
        training_episodes=training_episodes,
        test_episodes=test_episodes,
        warm_start_file=None,
        experiment_name="Cold Start Q-Learning"
    )
    if result:
        results.append(result)
    
    # 2. Warm-start Q-learning with DFS trajectory
    dfs_file = f'demo_trajectory_dfs_{layout}.pkl'
    result = experiment.run_qlearning_experiment(
        layout=layout,
        training_episodes=training_episodes,
        test_episodes=test_episodes,
        warm_start_file=dfs_file,
        experiment_name="Warm Start Q-Learning (DFS)"
    )
    if result:
        results.append(result)
    
    # 3. Warm-start Q-learning with ClosestDot trajectory
    closest_file = f'demo_trajectory_closest_{layout}.pkl'
    result = experiment.run_qlearning_experiment(
        layout=layout,
        training_episodes=training_episodes,
        test_episodes=test_episodes,
        warm_start_file=closest_file,
        experiment_name="Warm Start Q-Learning (ClosestDot)"
    )
    if result:
        results.append(result)
    
    # 4. Warm-start with original demo file
    if os.path.exists('demo_trajectory.pkl'):
        result = experiment.run_qlearning_experiment(
            layout=layout,
            training_episodes=training_episodes,
            test_episodes=test_episodes,
            warm_start_file='demo_trajectory.pkl',
            experiment_name="Warm Start Q-Learning (Original Demo)"
        )
        if result:
            results.append(result)
    
    # 5. Search-only baseline
    result = experiment.run_search_experiment(
        layout=layout,
        test_episodes=test_episodes,
        search_type='SearchAgent'
    )
    if result:
        results.append(result)
    
    # Generate report
    generate_report(results, layout, training_episodes, test_episodes)

def generate_report(results, layout, training_episodes, test_episodes):
    """Generate comprehensive comparison report"""
    
    if not results:
        print("No results to report!")
        return
    
    print("\\n" + "="*80)
    print("COMPREHENSIVE EXPERIMENT RESULTS")
    print("="*80)
    
    print(f"Layout: {layout}")
    print(f"Training Episodes: {training_episodes}")
    print(f"Test Episodes: {test_episodes}")
    print()
    
    # Results table
    print(f"{'Experiment':<40} {'Avg Score':<12} {'Win Rate':<10} {'Std Dev':<10} {'Convergence':<12}")
    print("-" * 85)
    
    for result in results:
        name = result['experiment_name']
        avg_score = result['average_score'] if result['average_score'] is not None else 'N/A'
        win_rate = f"{result['win_rate']:.2f}" if result['win_rate'] is not None else 'N/A'
        std_score = f"{result['std_score']:.2f}"
        convergence = result['convergence_episode'] if result['convergence_episode'] else 'N/A'
        
        print(f"{name:<40} {avg_score:<12} {win_rate:<10} {std_score:<10} {convergence:<12}")
    
    # Analysis
    print("\\n" + "="*50)
    print("ANALYSIS")
    print("="*50)
    
    # Find Q-learning results
    qlearning_results = [r for r in results if 'Q-Learning' in r['experiment_name']]
    
    if len(qlearning_results) >= 2:
        # Compare cold start vs warm start
        cold_start = next((r for r in qlearning_results if 'Cold Start' in r['experiment_name']), None)
        warm_starts = [r for r in qlearning_results if 'Warm Start' in r['experiment_name']]
        
        if cold_start and warm_starts:
            print(f"\\nCold Start Performance: {cold_start['average_score']:.2f}")
            print("Warm Start Comparisons:")
            
            for warm in warm_starts:
                if warm['average_score'] is not None and cold_start['average_score'] is not None:
                    improvement = warm['average_score'] - cold_start['average_score']
                    improvement_pct = (improvement / abs(cold_start['average_score'])) * 100
                    
                    print(f"  {warm['experiment_name']}: {warm['average_score']:.2f} "
                          f"({improvement:+.2f}, {improvement_pct:+.1f}%)")
    
    # Compare with search-only
    search_results = [r for r in results if 'Search' in r['experiment_name']]
    if search_results and qlearning_results:
        search_score = search_results[0]['average_score']
        best_ql = max(qlearning_results, key=lambda x: x['average_score'] or float('-inf'))
        
        if search_score is not None and best_ql['average_score'] is not None:
            print(f"\\nSearch-Only Performance: {search_score:.2f}")
            print(f"Best Q-Learning Performance: {best_ql['average_score']:.2f}")
            
            if best_ql['average_score'] > search_score:
                print("✓ Q-Learning outperforms Search-Only!")
            else:
                print("✗ Search-Only still better than Q-Learning")
    
    # Plot results
    plot_results(results, layout)

def plot_results(results, layout):
    """Plot comparison results"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average scores
        names = [r['experiment_name'] for r in results]
        scores = [r['average_score'] if r['average_score'] is not None else 0 for r in results]
        colors = ['red' if 'Cold Start' in name else 'green' if 'Warm Start' in name else 'blue' for name in names]
        
        bars1 = ax1.bar(range(len(names)), scores, color=colors, alpha=0.7)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Average Score')
        ax1.set_title(f'Average Performance ({layout})')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars1, scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.1f}', ha='center', va='bottom')
        
        # Plot 2: Win rates
        win_rates = [r['win_rate'] if r['win_rate'] is not None else 0 for r in results]
        bars2 = ax2.bar(range(len(names)), win_rates, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Win Rate')
        ax2.set_title(f'Win Rate Comparison ({layout})')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, win_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'qlearning_experiment_results_{layout}.png', dpi=300, bbox_inches='tight')
        print(f"\\nPlot saved as qlearning_experiment_results_{layout}.png")
        plt.show()
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == '__main__':
    main()
