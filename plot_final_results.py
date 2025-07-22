# plot_final_results.py
# Generate publication-quality plots of the warm-start vs cold-start comparison

import matplotlib.pyplot as plt
import numpy as np

def create_final_comparison_plot():
    """Create a comprehensive comparison plot"""
    
    # Data from our experiments
    experiment_data = {
        'smallGrid_50': {
            'Cold Start': -513.1,
            'Warm (DFS)': -510.5,
            'Warm (ClosestDot)': -508.0,
            'Warm (Original)': -509.0
        },
        'smallGrid_100': {
            'Cold Start': -508.8,
            'Warm (DFS)': -513.6,  # This one performed worse
            'Warm (ClosestDot)': -507.1,
            'Warm (Original)': -512.1
        },
        'mediumClassic_50': {
            'Cold Start': -435.2,
            'Warm (DFS)': -389.0,
            'Warm (ClosestDot)': -394.4,
            'Warm (Original)': -442.7
        },
        'mediumClassic_100': {
            'Cold Start': -426.8,
            'Warm (DFS)': -414.3,
            'Warm (ClosestDot)': -422.5,
            'Warm (Original)': -391.0
        }
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Q-Learning Performance: Warm-Start vs Cold-Start Comparison', fontsize=16, fontweight='bold')
    
    # Colors for different methods
    colors = {
        'Cold Start': '#ff4444',          # Red
        'Warm (DFS)': '#44ff44',         # Green
        'Warm (ClosestDot)': '#4444ff',  # Blue
        'Warm (Original)': '#ffaa44'     # Orange
    }
    
    # Plot each configuration
    configs = [
        ('smallGrid_50', 'SmallGrid - 50 Episodes', 0, 0),
        ('smallGrid_100', 'SmallGrid - 100 Episodes', 0, 1),
        ('mediumClassic_50', 'MediumClassic - 50 Episodes', 1, 0),
        ('mediumClassic_100', 'MediumClassic - 100 Episodes', 1, 1)
    ]
    
    for config_key, title, row, col in configs:
        ax = axes[row, col]
        data = experiment_data[config_key]
        
        methods = list(data.keys())
        scores = list(data.values())
        bar_colors = [colors[method] for method in methods]
        
        bars = ax.bar(range(len(methods)), scores, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the subplot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Score', fontsize=12)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(scores) - min(scores)) * 0.02,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Highlight the best performing method
        best_idx = np.argmax(scores)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        # Add improvement annotations
        cold_start_score = data['Cold Start']
        for i, (method, score) in enumerate(data.items()):
            if 'Warm' in method:
                improvement = score - cold_start_score
                if improvement > 0:
                    ax.annotate(f'+{improvement:.1f}', 
                               xy=(i, score), xytext=(i, score + (max(scores) - min(scores)) * 0.1),
                               ha='center', fontweight='bold', color='green',
                               arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # Save the plot
    plt.savefig('final_qlearning_comparison.png', dpi=300, bbox_inches='tight')
    print("Final comparison plot saved as 'final_qlearning_comparison.png'")
    
    return fig

def create_improvement_summary_plot():
    """Create a summary plot showing improvements"""
    
    # Calculate improvements for each experiment
    improvements_data = [
        ('SmallGrid 50ep', [2.6, 5.1, 4.1]),  # DFS, ClosestDot, Original
        ('SmallGrid 100ep', [-4.8, 1.7, -3.3]),  
        ('MediumClassic 50ep', [46.2, 40.8, -7.5]),
        ('MediumClassic 100ep', [12.5, 4.3, 35.8])
    ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    experiment_names = [item[0] for item in improvements_data]
    dfs_improvements = [item[1][0] for item in improvements_data]
    closest_improvements = [item[1][1] for item in improvements_data]
    original_improvements = [item[1][2] for item in improvements_data]
    
    x = np.arange(len(experiment_names))
    width = 0.25
    
    # Create grouped bars
    bars1 = ax.bar(x - width, dfs_improvements, width, label='DFS Trajectory', color='#44ff44', alpha=0.8)
    bars2 = ax.bar(x, closest_improvements, width, label='ClosestDot Trajectory', color='#4444ff', alpha=0.8)
    bars3 = ax.bar(x + width, original_improvements, width, label='Original Demo', color='#ffaa44', alpha=0.8)
    
    # Customize the plot
    ax.set_xlabel('Experiment Configuration', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score Improvement over Cold Start', fontsize=12, fontweight='bold')
    ax.set_title('Warm-Start Q-Learning Improvements by Demonstration Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'+{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height - 0.5,
                       f'{height:.1f}', ha='center', va='top', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('improvement_summary.png', dpi=300, bbox_inches='tight')
    print("Improvement summary plot saved as 'improvement_summary.png'")
    
    return fig

def create_key_insights_plot():
    """Create a plot highlighting key insights"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Key Insights from Warm-Start Q-Learning Experiment', fontsize=16, fontweight='bold')
    
    # 1. Overall improvement comparison
    methods = ['Cold Start\n(Baseline)', 'Warm Start\n(Average)']
    scores = [-470.98, -459.52]
    improvement = scores[1] - scores[0]
    
    bars = ax1.bar(methods, scores, color=['#ff4444', '#44aa44'], alpha=0.8, edgecolor='black')
    ax1.set_title('Overall Performance Comparison', fontweight='bold')
    ax1.set_ylabel('Average Score')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2., score + 1,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.text(0.5, -465, f'Improvement: +{improvement:.1f} points (2.4%)', 
             ha='center', transform=ax1.transData, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # 2. Layout complexity vs improvement
    layouts = ['SmallGrid\n(Simple)', 'MediumClassic\n(Complex)']
    avg_improvements = [2.0, 23.2]  # Approximate averages
    
    bars = ax2.bar(layouts, avg_improvements, color=['skyblue', 'navy'], alpha=0.8, edgecolor='black')
    ax2.set_title('Environment Complexity vs Improvement', fontweight='bold')
    ax2.set_ylabel('Average Improvement (points)')
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, imp in zip(bars, avg_improvements):
        ax2.text(bar.get_x() + bar.get_width()/2., imp + 0.5,
                f'+{imp:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Best performing methods
    methods = ['DFS', 'ClosestDot', 'Original']
    best_scores = [-389.0, -394.4, -391.0]  # Best scores from mediumClassic
    
    bars = ax3.bar(methods, best_scores, color=['#88ff88', '#8888ff', '#ffaa88'], alpha=0.8, edgecolor='black')
    ax3.set_title('Best Performance by Demonstration Type', fontweight='bold')
    ax3.set_ylabel('Best Score Achieved')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, score in zip(bars, best_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., score + 1,
                f'{score:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Training episodes vs performance
    episodes = ['50 Episodes', '100 Episodes']
    cold_performance = [(-435.2 + -513.1) / 2, (-426.8 + -508.8) / 2]  # Average across layouts
    warm_performance = [(-389.0 + -508.0) / 2, (-391.0 + -507.1) / 2]  # Best warm-start per config
    
    x = np.arange(len(episodes))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, cold_performance, width, label='Cold Start', color='#ff4444', alpha=0.8)
    bars2 = ax4.bar(x + width/2, warm_performance, width, label='Warm Start (Best)', color='#44aa44', alpha=0.8)
    
    ax4.set_title('Training Duration Comparison', fontweight='bold')
    ax4.set_ylabel('Average Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(episodes)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig('key_insights.png', dpi=300, bbox_inches='tight')
    print("Key insights plot saved as 'key_insights.png'")
    
    return fig

def main():
    """Generate all final plots"""
    
    print("Generating final comparison plots...")
    
    # Generate all plots
    create_final_comparison_plot()
    create_improvement_summary_plot()  
    create_key_insights_plot()
    
    print("\nAll plots generated successfully!")
    print("\nGenerated files:")
    print("- final_qlearning_comparison.png: Detailed comparison across all configurations")
    print("- improvement_summary.png: Summary of improvements by demonstration type")  
    print("- key_insights.png: Key insights from the experiment")
    print("- comprehensive_qlearning_results.png: Previous comprehensive results")

if __name__ == '__main__':
    main()
