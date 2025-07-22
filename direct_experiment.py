# direct_experiment.py
# Direct experiment runner that captures Pacman output properly

import os
import subprocess
import tempfile
import time

def run_experiment_direct(layout='smallGrid', training_episodes=50, test_episodes=5, 
                         warm_start_file=None, experiment_name="", python_exec=None):
    """Run experiment and capture output directly"""
    
    if python_exec is None:
        python_exec = "C:/Users/eldad/OneDrive/Desktop/pacman-merged/.venv/Scripts/python.exe"
    
    print(f"\n{'='*60}")
    print(f"Running: {experiment_name}")
    print(f"Layout: {layout}, Training: {training_episodes}, Test: {test_episodes}")
    
    # Build command
    cmd = [
        python_exec, "pacman.py",
        "-p", "PacmanQAgent",
        "-l", layout,
        "-x", str(training_episodes),
        "-n", str(training_episodes + test_episodes),
        "-q"  # Quiet graphics
    ]
    
    if warm_start_file and os.path.exists(warm_start_file):
        cmd.extend(["-a", f"warmStart=True,warmStartFile={warm_start_file}"])
        print(f"Using warm-start: {warm_start_file}")
    else:
        print("Using cold start")
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        end_time = time.time()
        
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
        if result.returncode != 0:
            print(f"Error (return code {result.returncode}):")
            print(result.stderr)
            return None
        
        # Print and parse the output
        output = result.stdout
        print("Raw output:")
        print(output)
        
        # Parse key metrics
        lines = output.split('\n')
        metrics = {}
        
        for line in lines:
            if "Average Score:" in line:
                try:
                    score = float(line.split("Average Score:")[1].strip())
                    metrics['average_score'] = score
                except:
                    pass
            
            if "Win Rate:" in line:
                try:
                    # Extract win rate like "0/5 (0.00)"
                    parts = line.split("Win Rate:")[1].strip()
                    if "(" in parts:
                        rate_str = parts.split("(")[1].split(")")[0]
                        metrics['win_rate'] = float(rate_str)
                    else:
                        # Try to parse "X/Y" format
                        frac_parts = parts.split()[0].split("/")
                        if len(frac_parts) == 2:
                            wins = int(frac_parts[0])
                            total = int(frac_parts[1])
                            metrics['win_rate'] = wins / total if total > 0 else 0
                except:
                    pass
            
            if "Scores:" in line:
                try:
                    # Extract individual scores
                    scores_part = line.split("Scores:")[1].strip()
                    # Remove any non-numeric characters except dots, minus signs, and commas
                    import re
                    score_matches = re.findall(r'-?\d+\.?\d*', scores_part)
                    metrics['scores'] = [float(s) for s in score_matches]
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
        
        print(f"Parsed metrics: {metrics}")
        return metrics
        
    except subprocess.TimeoutExpired:
        print("Experiment timed out!")
        return None
    except Exception as e:
        print(f"Experiment failed: {e}")
        return None

def run_search_experiment(layout='smallGrid', test_episodes=5, python_exec=None):
    """Run search-only experiment"""
    
    if python_exec is None:
        python_exec = "C:/Users/eldad/OneDrive/Desktop/pacman-merged/.venv/Scripts/python.exe"
    
    print(f"\n{'='*60}")
    print(f"Running: Search Only Baseline")
    print(f"Layout: {layout}, Episodes: {test_episodes}")
    
    cmd = [
        python_exec, "pacman.py",
        "-p", "SearchAgent",
        "-l", layout,
        "-n", str(test_episodes),
        "-q"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end_time = time.time()
        
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
        
        output = result.stdout
        print("Raw output:")
        print(output)
        
        # Parse metrics (similar to Q-learning)
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
                    pass
        
        metrics['experiment_name'] = 'Search Only'
        metrics['execution_time'] = end_time - start_time
        metrics['convergence_episode'] = 0  # Search doesn't converge
        
        print(f"Parsed metrics: {metrics}")
        return metrics
        
    except Exception as e:
        print(f"Search experiment failed: {e}")
        return None

def main():
    """Run all experiments"""
    
    print("DIRECT Q-LEARNING COMPARISON EXPERIMENT")
    print("="*70)
    
    # Experiment configuration
    layout = 'smallGrid'
    training_episodes = 30  # Smaller for faster testing
    test_episodes = 5
    
    results = []
    
    # 1. Cold start Q-learning
    result = run_experiment_direct(
        layout=layout,
        training_episodes=training_episodes,
        test_episodes=test_episodes,
        warm_start_file=None,
        experiment_name="Cold Start Q-Learning"
    )
    if result:
        results.append(result)
    
    # 2. Warm start with DFS trajectory
    dfs_file = f'demo_trajectory_dfs_{layout}.pkl'
    if os.path.exists(dfs_file):
        result = run_experiment_direct(
            layout=layout,
            training_episodes=training_episodes,
            test_episodes=test_episodes,
            warm_start_file=dfs_file,
            experiment_name="Warm Start Q-Learning (DFS)"
        )
        if result:
            results.append(result)
    
    # 3. Warm start with ClosestDot trajectory
    closest_file = f'demo_trajectory_closest_{layout}.pkl'
    if os.path.exists(closest_file):
        result = run_experiment_direct(
            layout=layout,
            training_episodes=training_episodes,
            test_episodes=test_episodes,
            warm_start_file=closest_file,
            experiment_name="Warm Start Q-Learning (ClosestDot)"
        )
        if result:
            results.append(result)
    
    # 4. Search only baseline
    search_result = run_search_experiment(
        layout=layout,
        test_episodes=test_episodes
    )
    if search_result:
        results.append(search_result)
    
    # Generate final comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    print("="*80)
    
    if not results:
        print("No results collected!")
        return
    
    print(f"Layout: {layout}")
    print(f"Training Episodes: {training_episodes}")
    print(f"Test Episodes: {test_episodes}")
    print()
    
    # Results table
    print(f"{'Experiment':<35} {'Avg Score':<12} {'Win Rate':<10} {'Time (s)':<10} {'Convergence':<12}")
    print("-" * 80)
    
    for result in results:
        name = result.get('experiment_name', 'Unknown')
        avg_score = result.get('average_score', 'N/A')
        win_rate = result.get('win_rate', 'N/A')
        exec_time = result.get('execution_time', 'N/A')
        convergence = result.get('convergence_episode', 'N/A')
        
        # Format the values
        avg_score_str = f"{avg_score:.1f}" if isinstance(avg_score, (int, float)) else str(avg_score)
        win_rate_str = f"{win_rate:.2f}" if isinstance(win_rate, (int, float)) else str(win_rate)
        time_str = f"{exec_time:.1f}" if isinstance(exec_time, (int, float)) else str(exec_time)
        conv_str = str(convergence) if convergence != 'N/A' else 'N/A'
        
        print(f"{name:<35} {avg_score_str:<12} {win_rate_str:<10} {time_str:<10} {conv_str:<12}")
    
    # Analysis
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    
    # Find best performing method
    qlearning_results = [r for r in results if 'Q-Learning' in r.get('experiment_name', '')]
    
    if qlearning_results:
        valid_ql_results = [r for r in qlearning_results if isinstance(r.get('average_score'), (int, float))]
        
        if valid_ql_results:
            best_ql = max(valid_ql_results, key=lambda x: x['average_score'])
            print(f"Best Q-Learning Method: {best_ql['experiment_name']}")
            print(f"  Average Score: {best_ql['average_score']:.2f}")
            
            # Compare warm vs cold start
            cold_start = next((r for r in valid_ql_results if 'Cold Start' in r['experiment_name']), None)
            warm_starts = [r for r in valid_ql_results if 'Warm Start' in r['experiment_name']]
            
            if cold_start and warm_starts:
                print(f"\nCold Start Score: {cold_start['average_score']:.2f}")
                print("Warm Start Improvements:")
                
                for warm in warm_starts:
                    improvement = warm['average_score'] - cold_start['average_score']
                    improvement_pct = (improvement / abs(cold_start['average_score'])) * 100
                    print(f"  {warm['experiment_name']}: {improvement:+.2f} ({improvement_pct:+.1f}%)")
    
    # Compare with search
    search_results = [r for r in results if 'Search' in r.get('experiment_name', '')]
    if search_results and qlearning_results:
        search_score = search_results[0].get('average_score')
        if isinstance(search_score, (int, float)):
            print(f"\nSearch-Only Performance: {search_score:.2f}")
            
            best_ql_score = best_ql.get('average_score') if 'best_ql' in locals() else None
            if isinstance(best_ql_score, (int, float)):
                if best_ql_score > search_score:
                    diff = best_ql_score - search_score
                    print(f"✓ Best Q-Learning beats Search by {diff:.2f} points!")
                else:
                    diff = search_score - best_ql_score
                    print(f"✗ Search still beats Q-Learning by {diff:.2f} points")

if __name__ == '__main__':
    main()
