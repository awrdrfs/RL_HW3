import matplotlib.pyplot as plt
import re

def parse_record(file_path):
    data = {}
    current_variant = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Check for variant header (e.g., "Static DQN:")
            if line.endswith(':'):
                current_variant = line[:-1]
                data[current_variant] = {'episodes': [], 'rewards': []}
                continue
            
            # Match "Episode X, Reward: Y, Epsilon: Z"
            match = re.search(r'Episode (\d+), Reward: ([\d\.-]+)', line)
            if match and current_variant:
                episode = int(match.group(1))
                reward = float(match.group(2))
                data[current_variant]['episodes'].append(episode)
                data[current_variant]['rewards'].append(reward)
                
    return data

def plot_data(data, output_file='dqn_performance.png'):
    plt.figure(figsize=(10, 6))
    
    colors = {'Static DQN': '#3498db', 'Double DQN': '#e74c3c', 'Dueling DQN': '#2ecc71'}
    
    for variant, results in data.items():
        plt.plot(results['episodes'], results['rewards'], 
                 label=variant, 
                 color=colors.get(variant),
                 marker='o', markersize=4, linewidth=2)
    
    plt.title('DQN Variants Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.legend(frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Premium styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    records = parse_record('record.txt')
    if records:
        plot_data(records)
    else:
        print("No data found in record.txt")
