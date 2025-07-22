# plotconvergence.py
# Detailed plotting of Q-matrix norm convergence over time

import subprocess
import matplotlib.pyplot as plt
import numpy as np
import re
import os

def extract_detailed_norm_data(output_text):
    """Extract norm data from experiment output"""
    lines = output_text.split('\n')
    
    norm_data = []
    episode_data = []
    
    for line in lines:
        # Extract norm values with episode numbers
        if "Q-matrix residual norm:" in line:
            try:
                # Extract episode number from the line
                episode_match = re.search(r'\[Episode (\d+)\]', line)
                # Extract norm value
                norm_match = re.search(r'Q-matrix residual norm: ([\d\.]+)', line)
                
                if episode_match and norm_match:
                    episode = int(episode_match.group(1))
                    norm_value = float(norm_match.group(1))
                    
                    episode_data.append(episode)
                    norm_data.append(norm_value)
            except:
                pass
    
    return episode_data, norm_data

def run_detailed_norm_experiment(layout='smallGrid', training_episodes=100):
    """Run experiment with detailed norm tracking"""
    
    python_exec = "C:/Users/eldad/OneDrive/Desktop/pacman-merged/.venv/Scripts/python.exe"
    
    experiments = [
        {
            'name': 'Cold Start',
            'warm_file': None,
            'color': 'red'
        },
        {
            'name': 'Warm Start (DFS)',
            'warm_file': f'demo_trajectory_dfs_{layout}.pkl',
            'color': 'blue'
        },
        {
            'name': 'Warm Start (ClosestDot)', 
            'warm_file': f'demo_trajectory_closest_{layout}.pkl',
            'color': 'green'
        }
    ]
    
    all_results = {}
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Running {exp['name']} experiment...")
        
        cmd = [
            python_exec, "pacman.py",
            "-p", "PacmanQAgent",
            "-l", layout,
            "-x", str(training_episodes),
            "-n", str(training_episodes + 5),
            "-q"
        ]
        
        if exp['warm_file'] and os.path.exists(exp['warm_file']):
            cmd.extend(["-a", f"warmStart=True,warmStartFile={exp['warm_file']}"])
            print(f"Using warm-start file: {exp['warm_file']}")
        else:
            print("Using cold start")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                output = result.stdout
                episodes, norms = extract_detailed_norm_data(output)
                
                if episodes and norms:
                    all_results[exp['name']] = {
                        'episodes': episodes,
                        'norms': norms,
                        'color': exp['color']
                    }
                    print(f"✓ Extracted {len(norms)} norm measurements")
                    print(f"  Episode range: {min(episodes)} - {max(episodes)}")
                    print(f"  Norm range: {min(norms):.4f} - {max(norms):.4f}")
                else:
                    print("✗ No norm data extracted")
            else:
                print(f"✗ Experiment failed: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Error running experiment: {e}")
    
    return all_results

def plot_detailed_convergence(results, layout, training_episodes):
    """Plot detailed norm convergence curves"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Q-Matrix Norm Convergence Analysis ({layout}, {training_episodes} episodes)', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Raw norm values over time
    for name, data in results.items():
        episodes = data['episodes']
        norms = data['norms']
        color = data['color']
        
        ax1.plot(episodes, norms, label=name, color=color, marker='o', markersize=4, alpha=0.8)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Q-Matrix Residual Norm')
    ax1.set_title('Residual Norm Over Time')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')  # Log scale to better see convergence
    
    # Plot 2: Smoothed convergence curves
    for name, data in results.items():
        episodes = data['episodes']
        norms = data['norms']
        color = data['color']
        
        if len(norms) > 3:
            # Simple moving average
            window = min(3, len(norms) // 3)
            if window > 1:
                smoothed_norms = np.convolve(norms, np.ones(window)/window, mode='valid')
                smoothed_episodes = episodes[window-1:]
                ax2.plot(smoothed_episodes, smoothed_norms, label=f'{name} (smoothed)', 
                        color=color, linewidth=2, alpha=0.8)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Smoothed Q-Matrix Residual Norm')
    ax2.set_title('Smoothed Convergence Curves')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Final norm values comparison
    names = list(results.keys())
    final_norms = [results[name]['norms'][-1] if results[name]['norms'] else 0 for name in names]
    colors = [results[name]['color'] for name in names]
    
    bars = ax3.bar(range(len(names)), final_norms, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(names)))
    ax3.set_xticklabels(names, rotation=15, ha='right')
    ax3.set_ylabel('Final Residual Norm')
    ax3.set_title('Final Convergence State')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, norm in zip(bars, final_norms):
        ax3.text(bar.get_x() + bar.get_width()/2., norm + max(final_norms) * 0.02,
                f'{norm:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 4: Convergence rate analysis
    for name, data in results.items():
        episodes = data['episodes']
        norms = data['norms']
        color = data['color']
        
        if len(norms) > 2:
            # Calculate rate of change
            rates = []
            rate_episodes = []
            
            for i in range(1, len(norms)):
                if episodes[i] != episodes[i-1]:  # Avoid division by zero
                    rate = -(norms[i] - norms[i-1]) / (episodes[i] - episodes[i-1])
                    rates.append(rate)
                    rate_episodes.append(episodes[i])
            
            if rates:
                ax4.plot(rate_episodes, rates, label=f'{name} convergence rate', 
                        color=color, marker='s', markersize=3, alpha=0.8)
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Convergence Rate (Norm Decrease per Episode)')
    ax4.set_title('Convergence Rate Analysis')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'detailed_norm_convergence_{layout}.png', dpi=300, bbox_inches='tight')
    print(f"\nDetailed convergence plot saved as 'detailed_norm_convergence_{layout}.png'")
    plt.show()

def analyze_convergence_behavior(results):
    """Analyze convergence behavior from norm data"""
    
    print(f"\n{'='*60}")
    print("CONVERGENCE BEHAVIOR ANALYSIS")
    print(f"{'='*60}")
    
    for name, data in results.items():
        episodes = data['episodes']
        norms = data['norms']
        
        print(f"\n{name}:")
        print(f"  Measurements: {len(norms)}")
        
        if norms:
            initial_norm = norms[0]
            final_norm = norms[-1]
            reduction = initial_norm - final_norm
            reduction_pct = (reduction / initial_norm) * 100 if initial_norm > 0 else 0
            
            print(f"  Initial norm: {initial_norm:.6f}")
            print(f"  Final norm: {final_norm:.6f}")
            print(f"  Total reduction: {reduction:.6f} ({reduction_pct:.2f}%)")
            
            # Check for convergence using different thresholds
            thresholds = [0.1, 0.01, 0.001]
            for threshold in thresholds:
                converged_episodes = [ep for ep, norm in zip(episodes, norms) if norm < threshold]
                if converged_episodes:
                    first_convergence = min(converged_episodes)
                    print(f"  Converged at norm < {threshold}: Episode {first_convergence}")
                else:
                    print(f"  Never converged to norm < {threshold}")
    
    # Compare convergence speeds
    print(f"\n{'='*40}")
    print("CONVERGENCE COMPARISON")
    print(f"{'='*40}")
    
    convergence_data = {}
    threshold = 0.01  # Use a reasonable threshold
    
    for name, data in results.items():
        episodes = data['episodes']
        norms = data['norms']
        
        # Find first episode where norm drops below threshold
        for ep, norm in zip(episodes, norms):
            if norm < threshold:
                convergence_data[name] = ep
                break
    
    if convergence_data:
        print(f"Episodes to convergence (norm < {threshold}):")
        sorted_convergence = sorted(convergence_data.items(), key=lambda x: x[1])
        
        for name, episode in sorted_convergence:
            print(f"  {name}: Episode {episode}")
        
        fastest = sorted_convergence[0]
        print(f"\nFastest convergence: {fastest[0]} at episode {fastest[1]}")
    else:
        print("No experiments achieved convergence with the current threshold.")
        print("Consider running longer experiments or adjusting convergence criteria.")

def main():
    """Run detailed norm convergence analysis"""
    
    print("DETAILED Q-MATRIX NORM CONVERGENCE ANALYSIS")
    print("="*80)
    
    layout = 'smallGrid'
    training_episodes = 150  # Longer training for better convergence analysis
    
    print(f"Layout: {layout}")
    print(f"Training Episodes: {training_episodes}")
    print(f"Analysis Focus: Q-matrix residual norm (Q_new - Q_old)")
    
    # Run experiments
    results = run_detailed_norm_experiment(layout, training_episodes)
    
    if results:
        # Generate plots and analysis
        plot_detailed_convergence(results, layout, training_episodes)
        analyze_convergence_behavior(results)
        
        print(f"\n{'='*80}")
        print("MENTOR'S APPROACH IMPLEMENTED!")
        print(f"{'='*80}")
        print("✓ Element-wise subtraction: Q_residual = Q_new - Q_old")
        print("✓ Matrix norm calculation: np.linalg.norm(Q_residual)")
        print("✓ Convergence tracking over episodes")
        print("✓ Comparison between warm-start and cold-start approaches")
        print()
        print("This provides a rigorous mathematical measure of Q-learning convergence")
        print("by comparing the entire Q-table rather than individual Q-values.")
        
    else:
        print("No experimental data collected. Check the setup and try again.")

if __name__ == '__main__':
    main()
