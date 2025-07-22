# enhanced_experiment.py
# Enhanced experiment with more comprehensive testing

import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np

def run_experiment_enhanced(layout='mediumClassic', training_episodes=100, test_episodes=10, 
                           warm_start_file=None, experiment_name="", python_exec=None):
    """Enhanced experiment runner with better metrics"""
    
    if python_exec is None:
        python_exec = "C:/Users/eldad/OneDrive/Desktop/pacman-merged/.venv/Scripts/python.exe"
    
    print(f"\n{'='*70}")
    print(f"Running: {experiment_name}")
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
        cmd.extend(["-a", f"warmStart=True,warmStartFile={warm_start_file},warmStartValue=50.0"])
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
        
        # Parse metrics
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            if "Average Score:" in line:
                try:
                    score = float(line.split("Average Score:")[1].strip())
                    metrics['average_score'] = score
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
            
            if "Scores:" in line:
                try:
                    import re
                    score_matches = re.findall(r'-?\d+\.?\d*', line.split("Scores:")[1])
                    metrics['scores'] = [float(s) for s in score_matches]
                    if metrics['scores']:
                        metrics['std_score'] = np.std(metrics['scores'])
                        metrics['max_score'] = max(metrics['scores'])
                        metrics['min_score'] = min(metrics['scores'])
                except:
                    pass
            
            if "[Convergence]" in line:
                try:
                    import re
                    match = re.search(r'episode (\d+)', line)
                    if match:
                        metrics['convergence_episode'] = int(match.group(1))
                except:
                    pass
        
        metrics['experiment_name'] = experiment_name
        metrics['execution_time'] = end_time - start_time
        metrics['layout'] = layout
        metrics['training_episodes'] = training_episodes
        
        print(f"Results: Score={metrics.get('average_score', 'N/A'):.1f}, "
              f"WinRate={metrics.get('win_rate', 0):.2f}, "
              f"Time={metrics['execution_time']:.1f}s")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print("Experiment timed out!")
        return None
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None

def run_multiple_layouts():
    """Run experiments on multiple layouts for comprehensive comparison"""
    
    print("COMPREHENSIVE Q-LEARNING WARM-START EXPERIMENT")
    print("="*80)
    
    # Test different layouts
    layouts = ['smallGrid', 'mediumClassic']
    
    # Experiment configurations
    configs = [
        {'training': 50, 'test': 10, 'name': 'Quick Test'},
        {'training': 100, 'test': 10, 'name': 'Standard Test'}
    ]
    
    all_results = []
    
    for layout in layouts:
        for config in configs:
            training_eps = config['training']
            test_eps = config['test']
            config_name = config['name']
            
            print(f"\n{'#'*50}")
            print(f"TESTING LAYOUT: {layout} ({config_name})")
            print(f"{'#'*50}")
            
            layout_results = []
            
            # 1. Cold start
            result = run_experiment_enhanced(
                layout=layout,
                training_episodes=training_eps,
                test_episodes=test_eps,
                warm_start_file=None,
                experiment_name=f"Cold Start"
            )
            if result:
                layout_results.append(result)
            
            # 2. Warm start with DFS
            dfs_file = f'demo_trajectory_dfs_{layout}.pkl'
            if os.path.exists(dfs_file):
                result = run_experiment_enhanced(
                    layout=layout,
                    training_episodes=training_eps,
                    test_episodes=test_eps,
                    warm_start_file=dfs_file,
                    experiment_name=f"Warm Start (DFS)"
                )
                if result:
                    layout_results.append(result)
            
            # 3. Warm start with ClosestDot
            closest_file = f'demo_trajectory_closest_{layout}.pkl'
            if os.path.exists(closest_file):
                result = run_experiment_enhanced(
                    layout=layout,
                    training_episodes=training_eps,
                    test_episodes=test_eps,
                    warm_start_file=closest_file,
                    experiment_name=f"Warm Start (ClosestDot)"
                )
                if result:
                    layout_results.append(result)
            
            # 4. Original demo (if available)
            if os.path.exists('demo_trajectory.pkl'):
                result = run_experiment_enhanced(
                    layout=layout,
                    training_episodes=training_eps,
                    test_episodes=test_eps,
                    warm_start_file='demo_trajectory.pkl',
                    experiment_name=f"Warm Start (Original)"
                )
                if result:
                    layout_results.append(result)
            
            # Analyze this configuration
            if layout_results:
                analyze_layout_results(layout_results, layout, config_name)
                all_results.extend(layout_results)
    
    # Final comprehensive analysis
    if all_results:
        generate_comprehensive_report(all_results)
        plot_comprehensive_results(all_results)

def analyze_layout_results(results, layout, config_name):
    """Analyze results for a specific layout and configuration"""
    
    print(f"\n{'-'*60}")
    print(f"ANALYSIS FOR {layout.upper()} ({config_name})")
    print(f"{'-'*60}")
    
    if not results:
        print("No results to analyze!")
        return
    
    # Table format
    print(f"{'Method':<25} {'Avg Score':<12} {'Win Rate':<10} {'Std Dev':<10} {'Time (s)':<10}")
    print("-" * 68)
    
    for result in results:
        name = result['experiment_name']
        avg_score = result.get('average_score', 'N/A')
        win_rate = result.get('win_rate', 'N/A')
        std_score = result.get('std_score', 'N/A')
        exec_time = result.get('execution_time', 'N/A')
        
        avg_str = f"{avg_score:.1f}" if isinstance(avg_score, (int, float)) else str(avg_score)
        win_str = f"{win_rate:.2f}" if isinstance(win_rate, (int, float)) else str(win_rate)
        std_str = f"{std_score:.1f}" if isinstance(std_score, (int, float)) else str(std_score)
        time_str = f"{exec_time:.1f}" if isinstance(exec_time, (int, float)) else str(exec_time)
        
        print(f"{name:<25} {avg_str:<12} {win_str:<10} {std_str:<10} {time_str:<10}")
    
    # Find best and worst
    valid_results = [r for r in results if isinstance(r.get('average_score'), (int, float))]
    
    if len(valid_results) >= 2:
        best_result = max(valid_results, key=lambda x: x['average_score'])
        cold_start = next((r for r in valid_results if 'Cold Start' in r['experiment_name']), None)
        
        print(f"\nBest Method: {best_result['experiment_name']} (Score: {best_result['average_score']:.1f})")
        
        if cold_start:
            warm_starts = [r for r in valid_results if 'Warm Start' in r['experiment_name']]
            if warm_starts:
                print(f"Cold Start Baseline: {cold_start['average_score']:.1f}")
                print("Warm Start Improvements:")
                
                for warm in warm_starts:
                    improvement = warm['average_score'] - cold_start['average_score']
                    improvement_pct = (improvement / abs(cold_start['average_score'])) * 100 if cold_start['average_score'] != 0 else 0
                    print(f"  {warm['experiment_name']}: {improvement:+.1f} ({improvement_pct:+.1f}%)")

def generate_comprehensive_report(all_results):
    """Generate comprehensive report across all experiments"""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    # Group by layout
    layout_groups = {}
    for result in all_results:
        layout = result.get('layout', 'unknown')
        if layout not in layout_groups:
            layout_groups[layout] = []
        layout_groups[layout].append(result)
    
    print(f"Total experiments conducted: {len(all_results)}")
    print(f"Layouts tested: {list(layout_groups.keys())}")
    
    # Overall statistics
    cold_starts = [r for r in all_results if 'Cold Start' in r['experiment_name']]
    warm_starts = [r for r in all_results if 'Warm Start' in r['experiment_name']]
    
    if cold_starts and warm_starts:
        cold_avg = np.mean([r['average_score'] for r in cold_starts if isinstance(r.get('average_score'), (int, float))])
        warm_avg = np.mean([r['average_score'] for r in warm_starts if isinstance(r.get('average_score'), (int, float))])
        
        print(f"\nOverall Performance:")
        print(f"  Cold Start Average: {cold_avg:.2f}")
        print(f"  Warm Start Average: {warm_avg:.2f}")
        print(f"  Average Improvement: {warm_avg - cold_avg:.2f} ({((warm_avg - cold_avg) / abs(cold_avg)) * 100:+.1f}%)")
    
    # Best performing configurations
    print(f"\nTop Performing Configurations:")
    valid_results = [r for r in all_results if isinstance(r.get('average_score'), (int, float))]
    if valid_results:
        top_results = sorted(valid_results, key=lambda x: x['average_score'], reverse=True)[:5]
        
        for i, result in enumerate(top_results, 1):
            print(f"  {i}. {result['experiment_name']} on {result.get('layout', 'unknown')}: {result['average_score']:.2f}")

def plot_comprehensive_results(all_results):
    """Plot comprehensive results"""
    
    try:
        # Group results by layout
        layout_groups = {}
        for result in all_results:
            layout = result.get('layout', 'unknown')
            if layout not in layout_groups:
                layout_groups[layout] = []
            layout_groups[layout].append(result)
        
        # Create subplots for each layout
        n_layouts = len(layout_groups)
        fig, axes = plt.subplots(1, n_layouts, figsize=(6 * n_layouts, 6))
        
        if n_layouts == 1:
            axes = [axes]
        
        for i, (layout, results) in enumerate(layout_groups.items()):
            ax = axes[i]
            
            # Extract data for plotting
            names = [r['experiment_name'] for r in results]
            scores = [r.get('average_score', 0) for r in results if isinstance(r.get('average_score'), (int, float))]
            valid_names = [names[j] for j, r in enumerate(results) if isinstance(r.get('average_score'), (int, float))]
            
            if scores:
                # Color coding: red for cold start, green for warm start
                colors = ['red' if 'Cold Start' in name else 'green' for name in valid_names]
                
                bars = ax.bar(range(len(valid_names)), scores, color=colors, alpha=0.7)
                ax.set_xticks(range(len(valid_names)))
                ax.set_xticklabels(valid_names, rotation=45, ha='right')
                ax.set_ylabel('Average Score')
                ax.set_title(f'Performance on {layout}')
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars, scores):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(scores) * 0.01,
                           f'{score:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('comprehensive_qlearning_results.png', dpi=300, bbox_inches='tight')
        print(f"\nComprehensive results plot saved as 'comprehensive_qlearning_results.png'")
        plt.show()
        
    except Exception as e:
        print(f"Could not generate comprehensive plot: {e}")

if __name__ == '__main__':
    run_multiple_layouts()
