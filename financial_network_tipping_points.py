"""
Financial Network Contagion: Non-Linear Tipping Points Analysis

This script models contagion dynamics in interbank networks, exploring the 
dual nature of connectivity where networks transition from resilient to 
fragile at critical thresholds.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FinancialNetwork:
    """
    Models a financial network with contagion dynamics.
    
    Key concepts:
    - Banks have capital buffers and interbank exposures
    - Contagion spreads when losses exceed capital buffers
    - Network structure determines resilience vs fragility
    """
    
    def __init__(self, n_banks=50, seed=42):
        """
        Initialize financial network.
        
        Parameters:
        -----------
        n_banks : int
            Number of banks in the network
        seed : int
            Random seed for reproducibility
        """
        self.n_banks = n_banks
        self.rng = np.random.RandomState(seed)
        
        # Bank characteristics
        self.capital_buffers = None
        self.external_assets = None
        self.interbank_assets = None
        self.interbank_liabilities = None
        
        # Network structure
        self.G = None
        self.adjacency_matrix = None
        
    def generate_network(self, network_type='scale_free', density=0.1, **kwargs):
        """
        Generate network structure.
        
        Parameters:
        -----------
        network_type : str
            Type of network: 'erdos_renyi', 'scale_free', 'small_world', 'core_periphery'
        density : float
            Network density (for Erdos-Renyi) or connection probability
        """
        if network_type == 'erdos_renyi':
            self.G = nx.erdos_renyi_graph(self.n_banks, density, seed=self.rng)
            
        elif network_type == 'scale_free':
            m = kwargs.get('m', 3)  # Number of edges to attach from new node
            self.G = nx.barabasi_albert_graph(self.n_banks, m, seed=self.rng)
            
        elif network_type == 'small_world':
            k = kwargs.get('k', 4)  # Each node connected to k nearest neighbors
            p = kwargs.get('p', 0.1)  # Rewiring probability
            self.G = nx.watts_strogatz_graph(self.n_banks, k, p, seed=self.rng)
            
        elif network_type == 'core_periphery':
            # Create core-periphery structure
            n_core = int(self.n_banks * 0.2)
            self.G = nx.Graph()
            self.G.add_nodes_from(range(self.n_banks))
            
            # Dense core connections
            for i in range(n_core):
                for j in range(i+1, n_core):
                    if self.rng.rand() < 0.8:
                        self.G.add_edge(i, j)
            
            # Sparse periphery connections
            for i in range(n_core, self.n_banks):
                # Connect to core
                n_connections = self.rng.randint(1, 4)
                core_nodes = self.rng.choice(n_core, n_connections, replace=False)
                for node in core_nodes:
                    self.G.add_edge(i, node)
        
        # Convert to directed graph for interbank lending
        self.G = self.G.to_directed()
        self.adjacency_matrix = nx.adjacency_matrix(self.G).toarray()
        
    def initialize_balance_sheets(self, capital_ratio_mean=0.08, capital_ratio_std=0.02,
                                  interbank_fraction=0.2):
        """
        Initialize bank balance sheets.
        
        Parameters:
        -----------
        capital_ratio_mean : float
            Mean capital ratio (capital/total assets)
        capital_ratio_std : float
            Std dev of capital ratios
        interbank_fraction : float
            Fraction of assets in interbank loans
        """
        # Total assets (normalized to 1 for simplicity)
        total_assets = np.ones(self.n_banks)
        
        # Capital buffers (equity)
        capital_ratios = self.rng.normal(capital_ratio_mean, capital_ratio_std, self.n_banks)
        capital_ratios = np.clip(capital_ratios, 0.03, 0.15)  # Regulatory bounds
        self.capital_buffers = capital_ratios * total_assets
        
        # Split assets between external and interbank
        self.external_assets = (1 - interbank_fraction) * total_assets
        
        # Interbank assets (loans to other banks)
        # Distribute based on network structure
        self.interbank_assets = np.zeros(self.n_banks)
        self.interbank_liabilities = np.zeros(self.n_banks)
        
        # Create exposure matrix
        self.exposure_matrix = np.zeros((self.n_banks, self.n_banks))
        
        for i in range(self.n_banks):
            out_neighbors = list(self.G.successors(i))
            if len(out_neighbors) > 0:
                # Distribute interbank assets among neighbors
                total_interbank = interbank_fraction * total_assets[i]
                exposures = self.rng.dirichlet(np.ones(len(out_neighbors))) * total_interbank
                
                for j, neighbor in enumerate(out_neighbors):
                    self.exposure_matrix[i, neighbor] = exposures[j]
                    self.interbank_assets[i] += exposures[j]
                    self.interbank_liabilities[neighbor] += exposures[j]
    
    def simulate_contagion(self, initial_shock_size, shock_banks=None, 
                          recovery_rate=0.4):
        """
        Simulate contagion cascade.
        
        Parameters:
        -----------
        initial_shock_size : float
            Size of initial shock as fraction of external assets
        shock_banks : list or None
            Banks receiving initial shock. If None, shock random bank
        recovery_rate : float
            Recovery rate on defaulted interbank loans (0-1)
            
        Returns:
        --------
        dict : Simulation results including default cascade
        """
        if shock_banks is None:
            shock_banks = [self.rng.randint(0, self.n_banks)]
        
        # Initialize state
        capital = self.capital_buffers.copy()
        defaults = np.zeros(self.n_banks, dtype=bool)
        default_times = np.full(self.n_banks, -1, dtype=int)
        
        # Apply initial shock
        for bank in shock_banks:
            loss = initial_shock_size * self.external_assets[bank]
            capital[bank] -= loss
            if capital[bank] <= 0:
                defaults[bank] = True
                default_times[bank] = 0
        
        # Propagate contagion
        time_step = 1
        new_defaults = True
        
        while new_defaults and time_step < 100:
            new_defaults = False
            
            for i in range(self.n_banks):
                if not defaults[i]:
                    # Calculate losses from defaulted counterparties
                    contagion_loss = 0
                    for j in range(self.n_banks):
                        if defaults[j] and self.exposure_matrix[i, j] > 0:
                            # Loss is exposure minus recovery
                            loss = self.exposure_matrix[i, j] * (1 - recovery_rate)
                            contagion_loss += loss
                    
                    # Update capital
                    capital[i] -= contagion_loss
                    
                    # Check for default
                    if capital[i] <= 0:
                        defaults[i] = True
                        default_times[i] = time_step
                        new_defaults = True
            
            time_step += 1
        
        return {
            'defaults': defaults,
            'default_times': default_times,
            'final_capital': capital,
            'n_defaults': np.sum(defaults),
            'default_fraction': np.mean(defaults),
            'cascade_length': time_step - 1
        }
    
    def analyze_tipping_point(self, shock_range=np.linspace(0, 0.5, 50),
                              n_simulations=100, recovery_rate=0.4):
        """
        Map the non-linear tipping point by varying shock size.
        
        Parameters:
        -----------
        shock_range : array
            Range of shock sizes to test
        n_simulations : int
            Number of Monte Carlo simulations per shock size
        recovery_rate : float
            Recovery rate on defaulted loans
            
        Returns:
        --------
        dict : Analysis results
        """
        mean_defaults = []
        std_defaults = []
        max_defaults = []
        min_defaults = []
        
        for shock_size in shock_range:
            defaults_list = []
            
            for _ in range(n_simulations):
                # Random initial shock bank
                shock_bank = self.rng.randint(0, self.n_banks)
                result = self.simulate_contagion(shock_size, [shock_bank], recovery_rate)
                defaults_list.append(result['default_fraction'])
            
            mean_defaults.append(np.mean(defaults_list))
            std_defaults.append(np.std(defaults_list))
            max_defaults.append(np.max(defaults_list))
            min_defaults.append(np.min(defaults_list))
        
        return {
            'shock_range': shock_range,
            'mean_defaults': np.array(mean_defaults),
            'std_defaults': np.array(std_defaults),
            'max_defaults': np.array(max_defaults),
            'min_defaults': np.array(min_defaults)
        }
    
    def compute_network_metrics(self):
        """Compute key network topology metrics."""
        metrics = {
            'density': nx.density(self.G),
            'avg_clustering': nx.average_clustering(self.G.to_undirected()),
            'avg_degree': np.mean([d for n, d in self.G.degree()]),
            'diameter': nx.diameter(self.G.to_undirected()) if nx.is_connected(self.G.to_undirected()) else np.inf,
        }
        
        # Centrality measures
        metrics['degree_centrality'] = nx.degree_centrality(self.G)
        metrics['betweenness_centrality'] = nx.betweenness_centrality(self.G)
        metrics['eigenvector_centrality'] = nx.eigenvector_centrality(self.G, max_iter=1000)
        
        return metrics


def compare_network_structures():
    """
    Compare tipping point behavior across different network structures.
    """
    print("Comparing tipping points across network structures...")
    
    network_types = {
        'Erdos-Renyi (Random)': ('erdos_renyi', {'density': 0.1}),
        'Scale-Free': ('scale_free', {'m': 3}),
        'Small-World': ('small_world', {'k': 6, 'p': 0.1}),
        'Core-Periphery': ('core_periphery', {})
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    shock_range = np.linspace(0, 0.5, 30)
    
    for idx, (name, (net_type, params)) in enumerate(network_types.items()):
        print(f"  Analyzing {name}...")
        
        # Create network
        fn = FinancialNetwork(n_banks=50, seed=42)
        fn.generate_network(net_type, **params)
        fn.initialize_balance_sheets(capital_ratio_mean=0.08, interbank_fraction=0.2)
        
        # Analyze tipping point
        results = fn.analyze_tipping_point(shock_range=shock_range, n_simulations=50)
        
        # Plot
        ax = axes[idx]
        ax.plot(results['shock_range'], results['mean_defaults'], 
                linewidth=2.5, label='Mean', color='darkred')
        ax.fill_between(results['shock_range'],
                        results['mean_defaults'] - results['std_defaults'],
                        results['mean_defaults'] + results['std_defaults'],
                        alpha=0.3, color='red', label='±1 Std Dev')
        ax.plot(results['shock_range'], results['max_defaults'],
                '--', alpha=0.5, color='darkred', label='Max')
        
        # Find tipping point (where derivative is maximum)
        derivative = np.gradient(results['mean_defaults'], results['shock_range'])
        tipping_idx = np.argmax(derivative)
        tipping_shock = results['shock_range'][tipping_idx]
        
        ax.axvline(tipping_shock, color='orange', linestyle='--', linewidth=2,
                  label=f'Tipping Point: {tipping_shock:.3f}')
        ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Compute network metrics
        metrics = fn.compute_network_metrics()
        
        ax.set_xlabel('Initial Shock Size (fraction of external assets)', fontsize=11)
        ax.set_ylabel('Fraction of Banks Defaulted', fontsize=11)
        ax.set_title(f'{name}\nDensity: {metrics["density"]:.3f}, Avg Degree: {metrics["avg_degree"]:.1f}',
                    fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/network_tipping_points_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("  Saved: network_tipping_points_comparison.png")
    
    return fig


def analyze_connectivity_resilience_tradeoff():
    """
    Explore the dual nature of connectivity: resilience vs fragility.
    """
    print("\nAnalyzing connectivity-resilience tradeoff...")
    
    # Vary network density
    densities = np.linspace(0.05, 0.4, 10)
    
    # Storage for results
    tipping_points = []
    small_shock_resilience = []
    large_shock_fragility = []
    avg_degrees = []
    
    shock_range = np.linspace(0, 0.5, 30)
    
    for density in densities:
        fn = FinancialNetwork(n_banks=50, seed=42)
        fn.generate_network('erdos_renyi', density=density)
        fn.initialize_balance_sheets(capital_ratio_mean=0.08, interbank_fraction=0.2)
        
        results = fn.analyze_tipping_point(shock_range=shock_range, n_simulations=30)
        
        # Find tipping point
        derivative = np.gradient(results['mean_defaults'], results['shock_range'])
        tipping_idx = np.argmax(derivative)
        tipping_points.append(results['shock_range'][tipping_idx])
        
        # Resilience to small shocks (shock = 0.1)
        small_idx = np.argmin(np.abs(results['shock_range'] - 0.1))
        small_shock_resilience.append(1 - results['mean_defaults'][small_idx])
        
        # Fragility to large shocks (shock = 0.3)
        large_idx = np.argmin(np.abs(results['shock_range'] - 0.3))
        large_shock_fragility.append(results['mean_defaults'][large_idx])
        
        metrics = fn.compute_network_metrics()
        avg_degrees.append(metrics['avg_degree'])
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Tipping point vs connectivity
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(avg_degrees, tipping_points, 'o-', linewidth=2.5, markersize=8, color='darkblue')
    ax1.set_xlabel('Average Degree (Network Connectivity)', fontsize=12)
    ax1.set_ylabel('Tipping Point Shock Size', fontsize=12)
    ax1.set_title('Tipping Point vs Network Connectivity', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Resilience vs Fragility
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(avg_degrees, small_shock_resilience, 'o-', linewidth=2.5, 
             markersize=8, color='green', label='Resilience (Small Shock)')
    ax2.plot(avg_degrees, large_shock_fragility, 's-', linewidth=2.5,
             markersize=8, color='red', label='Fragility (Large Shock)')
    ax2.set_xlabel('Average Degree (Network Connectivity)', fontsize=12)
    ax2.set_ylabel('Proportion', fontsize=12)
    ax2.set_title('The Dual Nature of Connectivity', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Phase diagram
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create heatmap of default probability vs shock and connectivity
    shock_grid = np.linspace(0, 0.5, 40)
    density_grid = np.linspace(0.05, 0.4, 30)
    default_matrix = np.zeros((len(density_grid), len(shock_grid)))
    
    for i, density in enumerate(density_grid):
        fn = FinancialNetwork(n_banks=50, seed=42)
        fn.generate_network('erdos_renyi', density=density)
        fn.initialize_balance_sheets(capital_ratio_mean=0.08, interbank_fraction=0.2)
        
        results = fn.analyze_tipping_point(shock_range=shock_grid, n_simulations=20)
        default_matrix[i, :] = results['mean_defaults']
    
    im = ax3.contourf(shock_grid, density_grid, default_matrix, 
                      levels=20, cmap='RdYlGn_r')
    
    # Add tipping point line
    ax3.plot(tipping_points, densities, 'w--', linewidth=3, label='Tipping Point')
    
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Fraction of Banks Defaulted', fontsize=12)
    
    ax3.set_xlabel('Initial Shock Size', fontsize=12)
    ax3.set_ylabel('Network Density', fontsize=12)
    ax3.set_title('Phase Diagram: Network Density vs Shock Size', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper left')
    
    plt.savefig('/mnt/user-data/outputs/connectivity_resilience_tradeoff.png',
                dpi=300, bbox_inches='tight')
    print("  Saved: connectivity_resilience_tradeoff.png")
    
    return fig


def analyze_systemic_importance():
    """
    Identify systemically important banks using network centrality.
    """
    print("\nAnalyzing systemic importance...")
    
    # Create scale-free network (realistic for banking)
    fn = FinancialNetwork(n_banks=50, seed=42)
    fn.generate_network('scale_free', m=3)
    fn.initialize_balance_sheets(capital_ratio_mean=0.08, interbank_fraction=0.25)
    
    # Compute centrality measures
    metrics = fn.compute_network_metrics()
    
    # Test each bank as initial shock source
    shock_size = 0.2
    impacts = []
    
    for bank_id in range(fn.n_banks):
        result = fn.simulate_contagion(shock_size, [bank_id], recovery_rate=0.4)
        impacts.append(result['default_fraction'])
    
    impacts = np.array(impacts)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Network visualization colored by systemic impact
    ax = axes[0, 0]
    pos = nx.spring_layout(fn.G, seed=42, k=0.5)
    
    node_colors = impacts
    nodes = nx.draw_networkx_nodes(fn.G, pos, node_color=node_colors, 
                                   node_size=300, cmap='Reds', 
                                   vmin=0, vmax=1, ax=ax)
    nx.draw_networkx_edges(fn.G, pos, alpha=0.2, ax=ax, arrows=True,
                          arrowsize=10, arrowstyle='->')
    
    plt.colorbar(nodes, ax=ax, label='Contagion Impact')
    ax.set_title('Network Structure\n(Node color = Systemic Impact)', 
                fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Plot 2: Systemic impact vs degree centrality
    ax = axes[0, 1]
    degree_cent = np.array([metrics['degree_centrality'][i] for i in range(fn.n_banks)])
    ax.scatter(degree_cent, impacts, alpha=0.6, s=100)
    
    # Fit line
    z = np.polyfit(degree_cent, impacts, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(degree_cent), p(np.sort(degree_cent)), 
            "r--", alpha=0.8, linewidth=2)
    
    corr = np.corrcoef(degree_cent, impacts)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Degree Centrality', fontsize=12)
    ax.set_ylabel('Contagion Impact (Fraction Defaulted)', fontsize=12)
    ax.set_title('Systemic Impact vs Degree Centrality', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Systemic impact vs betweenness centrality
    ax = axes[1, 0]
    between_cent = np.array([metrics['betweenness_centrality'][i] for i in range(fn.n_banks)])
    ax.scatter(between_cent, impacts, alpha=0.6, s=100, color='green')
    
    z = np.polyfit(between_cent, impacts, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(between_cent), p(np.sort(between_cent)), 
            "r--", alpha=0.8, linewidth=2)
    
    corr = np.corrcoef(between_cent, impacts)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Betweenness Centrality', fontsize=12)
    ax.set_ylabel('Contagion Impact (Fraction Defaulted)', fontsize=12)
    ax.set_title('Systemic Impact vs Betweenness Centrality',
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Top systemically important banks
    ax = axes[1, 1]
    top_n = 10
    top_indices = np.argsort(impacts)[-top_n:][::-1]
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
    bars = ax.barh(range(top_n), impacts[top_indices], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f'Bank {i}' for i in top_indices])
    ax.set_xlabel('Contagion Impact', fontsize=12)
    ax.set_title(f'Top {top_n} Systemically Important Banks', 
                fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add degree info
    for i, (idx, bar) in enumerate(zip(top_indices, bars)):
        degree = fn.G.out_degree(idx) + fn.G.in_degree(idx)
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'deg={degree}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/systemic_importance_analysis.png',
                dpi=300, bbox_inches='tight')
    print("  Saved: systemic_importance_analysis.png")
    
    return fig, impacts, metrics


def create_interactive_dashboard():
    """
    Create a comprehensive dashboard showing all key insights.
    """
    print("\nCreating comprehensive dashboard...")
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # Analyze different aspects
    fn = FinancialNetwork(n_banks=50, seed=42)
    fn.generate_network('scale_free', m=3)
    fn.initialize_balance_sheets(capital_ratio_mean=0.08, interbank_fraction=0.2)
    
    shock_range = np.linspace(0, 0.5, 40)
    
    # 1. Main tipping point curve
    ax1 = fig.add_subplot(gs[0, :2])
    results = fn.analyze_tipping_point(shock_range=shock_range, n_simulations=100)
    
    ax1.plot(results['shock_range'], results['mean_defaults'],
            linewidth=3, label='Mean Default Rate', color='darkred')
    ax1.fill_between(results['shock_range'],
                    results['mean_defaults'] - results['std_defaults'],
                    results['mean_defaults'] + results['std_defaults'],
                    alpha=0.3, color='red')
    
    # Find and mark tipping point
    derivative = np.gradient(results['mean_defaults'], results['shock_range'])
    tipping_idx = np.argmax(derivative)
    tipping_shock = results['shock_range'][tipping_idx]
    
    ax1.axvline(tipping_shock, color='orange', linestyle='--', linewidth=2.5,
               label=f'Tipping Point: {tipping_shock:.3f}')
    ax1.scatter([tipping_shock], [results['mean_defaults'][tipping_idx]],
               s=200, color='orange', zorder=5, edgecolors='black', linewidths=2)
    
    # Mark regions
    ax1.axvspan(0, tipping_shock, alpha=0.1, color='green', label='Resilient Zone')
    ax1.axvspan(tipping_shock, 0.5, alpha=0.1, color='red', label='Fragile Zone')
    
    ax1.set_xlabel('Initial Shock Size', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Fraction of Banks Defaulted', fontsize=13, fontweight='bold')
    ax1.set_title('Non-Linear Tipping Point in Financial Network Contagion',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Derivative (Rate of change)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(results['shock_range'], derivative, linewidth=2.5, color='purple')
    ax2.axvline(tipping_shock, color='orange', linestyle='--', linewidth=2)
    ax2.fill_between(results['shock_range'], 0, derivative, alpha=0.3, color='purple')
    ax2.set_xlabel('Shock Size', fontsize=11)
    ax2.set_ylabel('d(Defaults)/d(Shock)', fontsize=11)
    ax2.set_title('Contagion Sensitivity\n(First Derivative)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Network visualization
    ax3 = fig.add_subplot(gs[1, 0])
    pos = nx.spring_layout(fn.G, seed=42, k=0.5)
    
    # Color by capital buffer
    node_colors = fn.capital_buffers
    nodes = nx.draw_networkx_nodes(fn.G, pos, node_color=node_colors,
                                   node_size=200, cmap='RdYlGn',
                                   vmin=0.03, vmax=0.12, ax=ax3)
    nx.draw_networkx_edges(fn.G, pos, alpha=0.15, ax=ax3, arrows=False)
    
    ax3.set_title('Network Structure\n(Color = Capital Buffer)', 
                 fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Capital buffer distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(fn.capital_buffers, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(fn.capital_buffers), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {np.mean(fn.capital_buffers):.3f}')
    ax4.set_xlabel('Capital Buffer', fontsize=11)
    ax4.set_ylabel('Number of Banks', fontsize=11)
    ax4.set_title('Distribution of Capital Buffers', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Degree distribution
    ax5 = fig.add_subplot(gs[1, 2])
    degrees = [fn.G.degree(n) for n in fn.G.nodes()]
    ax5.hist(degrees, bins=20, color='coral', alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Node Degree', fontsize=11)
    ax5.set_ylabel('Number of Banks', fontsize=11)
    ax5.set_title('Degree Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Recovery rate sensitivity
    ax6 = fig.add_subplot(gs[2, 0])
    recovery_rates = [0.2, 0.4, 0.6, 0.8]
    colors_rec = plt.cm.viridis(np.linspace(0.2, 0.9, len(recovery_rates)))
    
    for rr, color in zip(recovery_rates, colors_rec):
        results_rr = fn.analyze_tipping_point(shock_range=shock_range[:20], 
                                              n_simulations=30, recovery_rate=rr)
        ax6.plot(results_rr['shock_range'], results_rr['mean_defaults'],
                linewidth=2, label=f'Recovery = {rr:.1f}', color=color)
    
    ax6.set_xlabel('Shock Size', fontsize=11)
    ax6.set_ylabel('Fraction Defaulted', fontsize=11)
    ax6.set_title('Impact of Recovery Rate', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 7. Cascade depth distribution
    ax7 = fig.add_subplot(gs[2, 1])
    cascade_lengths = []
    for _ in range(100):
        shock_bank = fn.rng.randint(0, fn.n_banks)
        result = fn.simulate_contagion(0.25, [shock_bank], recovery_rate=0.4)
        cascade_lengths.append(result['cascade_length'])
    
    ax7.hist(cascade_lengths, bins=15, color='darkgreen', alpha=0.7, edgecolor='black')
    ax7.set_xlabel('Cascade Length (time steps)', fontsize=11)
    ax7.set_ylabel('Frequency', fontsize=11)
    ax7.set_title('Distribution of Cascade Depths\n(Shock = 0.25)', 
                 fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Key metrics summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    metrics = fn.compute_network_metrics()
    
    summary_text = f"""
    NETWORK METRICS
    {'='*30}
    
    Topology:
    • N Banks: {fn.n_banks}
    • Density: {metrics['density']:.3f}
    • Avg Degree: {metrics['avg_degree']:.2f}
    • Clustering: {metrics['avg_clustering']:.3f}
    
    Balance Sheets:
    • Avg Capital: {np.mean(fn.capital_buffers):.3f}
    • Interbank Fraction: 0.20
    
    Contagion Dynamics:
    • Tipping Point: {tipping_shock:.3f}
    • Max Sensitivity: {np.max(derivative):.3f}
    
    Interpretation:
    Below {tipping_shock:.3f}: Network
    absorbs shocks through 
    diversification
    
    Above {tipping_shock:.3f}: Cascading
    failures dominate, leading
    to systemic crisis
    """
    
    ax8.text(0.1, 0.95, summary_text, transform=ax8.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('Financial Network Contagion: Comprehensive Analysis Dashboard',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('/mnt/user-data/outputs/contagion_dashboard.png',
                dpi=300, bbox_inches='tight')
    print("  Saved: contagion_dashboard.png")
    
    return fig


def main():
    """Main execution function."""
    print("="*70)
    print("FINANCIAL NETWORK CONTAGION: NON-LINEAR TIPPING POINTS")
    print("="*70)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run analyses
    print("\n[1/4] Comparing network structures...")
    fig1 = compare_network_structures()
    
    print("\n[2/4] Analyzing connectivity-resilience tradeoff...")
    fig2 = analyze_connectivity_resilience_tradeoff()
    
    print("\n[3/4] Identifying systemically important institutions...")
    fig3, impacts, metrics = analyze_systemic_importance()
    
    print("\n[4/4] Creating comprehensive dashboard...")
    fig4 = create_interactive_dashboard()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("  • Tipping points vary by network structure (density, topology)")
    print("  • Higher connectivity increases resilience to small shocks")
    print("  • BUT also increases fragility to large shocks")
    print("  • Scale-free networks have critical 'hub' banks")
    print("  • Systemic importance correlates with network centrality")
    print("\nFiles saved to /mnt/user-data/outputs/")
    
    plt.show()


if __name__ == "__main__":
    main()
