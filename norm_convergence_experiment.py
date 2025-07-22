# norm_convergence_experiment.py
# Experiment focused on Q-matrix norm-based convergence analysis

import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def run_norm_experiment(layout='smallGrid', training_episodes=100, test_episodes=5, 
                       warm_start_file=None, experiment_name="", python_exec=None):
    """Run experiment with focus on norm-based convergence tracking"""
    
    if python_exec is None:
        python_exec = "C:/Users/eldad/OneDrive/Desktop/pacman-merged/.venv/Scripts/python.exe"
    
    print(f"\n{'='*70}")
    print(f"Norm Convergence Analysis: {experiment_name}")
    print(f"Layout: {layout}, Training: {training_episodes}, Test: {test_episodes}")
    
    # Build command
    cmd = [
        python_exec, "pacman.py",
        "-p", "PacmanQAgent",
        "-l", layout,
        "-x", str(training_episodes),
        "-n", str(training_episodes + test_episodes),
        "-q"
    ]
    
    if warm_start_file and os.path.exists(warm_start_file):
        cmd.extend(["-a", f"warmStart=True,warmStartFile={warm_start_file}"])
        print(f"Using warm-start: {warm_start_file}")
    else:
        print("Using cold start")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        output = result.stdout
        
        # Parse norm convergence information from output
        metrics = {
            'experiment_name': experiment_name,
            'execution_time': end_time - start_time,
            'layout': layout,
            'training_episodes': training_episodes
        }
        
        # Parse standard metrics
        lines = output.split('\n')
        for line in lines:
            if "Average Score:" in line:
                try:
                    metrics['average_score'] = float(line.split("Average Score:")[1].strip())
                except:
                    pass
            
            if "Win Rate:" in line:
                try:
                    parts = line.split("Win Rate:")[1].strip()
                    if "(" in parts:
                        rate_str = parts.split("(")[1].split(")")[0]
                        metrics['win_rate'] = float(rate_str)
                except:
                    metrics['win_rate'] = 0.0
            
            if "[Convergence]" in line:
                try:
                    import re
                    match = re.search(r'episode (\d+)', line)
                    if match:
                        metrics['convergence_episode'] = int(match.group(1))
                except:
                    pass
            
            if "[Norm Convergence]" in line:
                try:
                    import re
                    match = re.search(r'episode (\d+)', line)
                    if match:
                        metrics['norm_convergence_episode'] = int(match.group(1))
                        print(f"✓ Norm-based convergence detected at episode {match.group(1)}")
                except:
                    pass
            
            if "Q-matrix residual norm:" in line:
                try:
                    import re
                    match = re.search(r'Q-matrix residual norm: ([\d\.]+)', line)
                    if match:
                        norm_value = float(match.group(1))
                        if 'norm_values' not in metrics:
                            metrics['norm_values'] = []
                        metrics['norm_values'].append(norm_value)
                except:
                    pass
        
        # Print summary
        print(f"Results:")
        print(f"  Average Score: {metrics.get('average_score', 'N/A')}")
        print(f"  Win Rate: {metrics.get('win_rate', 'N/A'):.2f}")
        print(f"  Standard Convergence: {metrics.get('convergence_episode', 'N/A')}")
        print(f"  Norm Convergence: {metrics.get('norm_convergence_episode', 'N/A')}")
        print(f"  Execution Time: {metrics['execution_time']:.1f}s")
        
        if 'norm_values' in metrics:
            print(f"  Final Norm Value: {metrics['norm_values'][-1]:.4f}")
            print(f"  Norm Measurements: {len(metrics['norm_values'])}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("Experiment timed out!")
        return None
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None

def run_norm_comparison_experiment():
    """Run comprehensive norm-based convergence comparison"""
    
    print("Q-MATRIX NORM CONVERGENCE ANALYSIS")
    print("="*80)
    
    results = []
    layout = 'smallGrid'
    training_episodes = 80  # Longer training to see convergence
    test_episodes = 5
    
    # 1. Cold start
    result = run_norm_experiment(
        layout=layout,
        training_episodes=training_episodes,
        test_episodes=test_episodes,
        warm_start_file=None,
        experiment_name="Cold Start Q-Learning"
    )
    if result:
        results.append(result)
    
    # 2. Warm start with DFS
    dfs_file = f'demo_trajectory_dfs_{layout}.pkl'
    if os.path.exists(dfs_file):
        result = run_norm_experiment(
            layout=layout,
            training_episodes=training_episodes,
            test_episodes=test_episodes,
            warm_start_file=dfs_file,
            experiment_name="Warm Start (DFS)"
        )
        if result:
            results.append(result)
    
    # 3. Warm start with ClosestDot
    closest_file = f'demo_trajectory_closest_{layout}.pkl'
    if os.path.exists(closest_file):
        result = run_norm_experiment(
            layout=layout,
            training_episodes=training_episodes,
            test_episodes=test_episodes,
            warm_start_file=closest_file,
            experiment_name="Warm Start (ClosestDot)"
        )
        if result:
            results.append(result)
    
    # Generate analysis
    analyze_norm_results(results, layout, training_episodes)
    
    return results

def analyze_norm_results(results, layout, training_episodes):
    """Analyze norm convergence results"""
    
    print(f"\n{'='*80}")
    print("NORM-BASED CONVERGENCE ANALYSIS")
    print(f"{'='*80}")
    
    if not results:
        print("No results to analyze!")
        return
    
    print(f"Layout: {layout}")
    print(f"Training Episodes: {training_episodes}")
    print()
    
    # Results table
    print(f"{'Experiment':<25} {'Avg Score':<12} {'Std Conv':<12} {'Norm Conv':<12} {'Final Norm':<12}")
    print("-" * 75)
    
    for result in results:
        name = result['experiment_name']
        avg_score = result.get('average_score', 'N/A')
        std_conv = result.get('convergence_episode', 'N/A')
        norm_conv = result.get('norm_convergence_episode', 'N/A')
        
        final_norm = 'N/A'
        if 'norm_values' in result and result['norm_values']:
            final_norm = f"{result['norm_values'][-1]:.4f}"
        
        avg_str = f"{avg_score:.1f}" if isinstance(avg_score, (int, float)) else str(avg_score)
        
        print(f"{name:<25} {avg_str:<12} {std_conv:<12} {norm_conv:<12} {final_norm:<12}")
    
    # Analysis
    print(f"\n{'='*50}")
    print("KEY INSIGHTS")
    print(f"{'='*50}")
    
    # Compare convergence speeds
    valid_results = [r for r in results if r.get('norm_convergence_episode')]
    
    if valid_results:
        print("\nNorm-Based Convergence Analysis:")
        
        fastest_norm = min(valid_results, key=lambda x: x['norm_convergence_episode'])
        print(f"  Fastest norm convergence: {fastest_norm['experiment_name']} "
              f"at episode {fastest_norm['norm_convergence_episode']}")
        
        # Compare cold vs warm start
        cold_result = next((r for r in valid_results if 'Cold Start' in r['experiment_name']), None)
        warm_results = [r for r in valid_results if 'Warm Start' in r['experiment_name']]
        
        if cold_result and warm_results:
            cold_conv = cold_result['norm_convergence_episode']
            print(f"  Cold start convergence: Episode {cold_conv}")
            print("  Warm start convergence improvements:")
            
            for warm in warm_results:
                warm_conv = warm['norm_convergence_episode']
                improvement = cold_conv - warm_conv
                improvement_pct = (improvement / cold_conv) * 100 if cold_conv > 0 else 0
                print(f"    {warm['experiment_name']}: Episode {warm_conv} "
                      f"({improvement:+d} episodes, {improvement_pct:+.1f}%)")
    
    # Plot norm convergence if we have norm values
    plot_norm_convergence(results, layout)

def plot_norm_convergence(results, layout):
    """Plot Q-matrix norm convergence over time"""
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Performance comparison
        names = [r['experiment_name'] for r in results]
        scores = [r.get('average_score', 0) for r in results if isinstance(r.get('average_score'), (int, float))]
        valid_names = [names[i] for i, r in enumerate(results) if isinstance(r.get('average_score'), (int, float))]
        
        if scores:
            colors = ['red' if 'Cold Start' in name else 'green' for name in valid_names]
            bars = ax1.bar(range(len(valid_names)), scores, color=colors, alpha=0.7)
            ax1.set_xticks(range(len(valid_names)))
            ax1.set_xticklabels(valid_names, rotation=45, ha='right')
            ax1.set_ylabel('Average Score')
            ax1.set_title(f'Performance Comparison ({layout})')
            ax1.grid(axis='y', alpha=0.3)
            
            for bar, score in zip(bars, scores):
                ax1.text(bar.get_x() + bar.get_width()/2., score + max(scores) * 0.01,
                        f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Convergence comparison
        conv_data = []
        for result in results:
            name = result['experiment_name']
            std_conv = result.get('convergence_episode')
            norm_conv = result.get('norm_convergence_episode')
            
            if std_conv or norm_conv:
                conv_data.append({
                    'name': name,
                    'standard': std_conv if std_conv else 0,
                    'norm': norm_conv if norm_conv else 0
                })
        
        if conv_data:
            names = [d['name'] for d in conv_data]
            std_convs = [d['standard'] for d in conv_data]
            norm_convs = [d['norm'] for d in conv_data]
            
            x = np.arange(len(names))
            width = 0.35
            
            ax2.bar(x - width/2, std_convs, width, label='Standard Convergence', alpha=0.7, color='blue')
            ax2.bar(x + width/2, norm_convs, width, label='Norm Convergence', alpha=0.7, color='orange')
            
            ax2.set_xlabel('Experiment')
            ax2.set_ylabel('Episodes to Convergence')
            ax2.set_title('Convergence Speed Comparison')
            ax2.set_xticks(x)
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'norm_convergence_analysis_{layout}.png', dpi=300, bbox_inches='tight')
        print(f"\nNorm convergence plot saved as 'norm_convergence_analysis_{layout}.png'")
        plt.show()
        
    except Exception as e:
        print(f"Could not generate plot: {e}")

def main():
    """Run the norm convergence experiment"""
    results = run_norm_comparison_experiment()
    
    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total experiments: {len(results)}")
    print("The Q-matrix norm provides a more rigorous measure of convergence")
    print("by comparing the entire Q-table rather than individual Q-values.")
    print()
    print("Key benefits of norm-based convergence:")
    print("- Captures global changes in the Q-table")
    print("- More mathematically rigorous than max delta tracking")
    print("- Better indicates when the policy has truly stabilized")
    print("- Follows your mentor's suggested approach!")

if __name__ == '__main__':
    main()
