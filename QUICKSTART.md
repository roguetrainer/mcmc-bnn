# Quick Start Guide
## Financial Network Contagion Analysis

Get up and running in 5 minutes!

---

## 🚀 Installation (Choose One Method)

### Method 1: Automatic Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup.sh

# Run interactive setup
bash setup.sh

# Or install specific modes
bash setup.sh --basic    # Core dependencies only
bash setup.sh --full     # All dependencies including Bayesian
bash setup.sh --dev      # Development tools
```

### Method 2: Manual Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Using Make

```bash
# Install core dependencies
make install

# Or install everything
make install-full

# Check installation
make check-deps
```

---

## 📊 Run Your First Analysis

### Option 1: Command Line

```bash
# Run main analysis (generates 4 visualizations)
python financial_network_tipping_points.py

# Run Bayesian analysis (adds 2 more visualizations)
python bayesian_contagion_extension.py

# Or use Make shortcuts
make run       # Main analysis
make bayesian  # Bayesian analysis
make all       # Both analyses
```

### Option 2: Using Make Commands

```bash
# Quick analysis (faster, fewer simulations)
make quick-analysis

# Full analysis with report
make report

# Interactive Python session with imports loaded
make interactive
```

### Option 3: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook
# Or
make notebook

# Create a new notebook and run:
```

```python
import numpy as np
from financial_network_tipping_points import FinancialNetwork

# Create network
fn = FinancialNetwork(n_banks=50)
fn.generate_network('scale_free', m=3)
fn.initialize_balance_sheets()

# Analyze tipping point
shock_range = np.linspace(0, 0.5, 30)
results = fn.analyze_tipping_point(shock_range=shock_range, n_simulations=50)

# Find tipping point
import matplotlib.pyplot as plt
derivative = np.gradient(results['mean_defaults'])
tipping_point = results['shock_range'][np.argmax(derivative)]

print(f"Tipping point: {tipping_point:.3f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(results['shock_range'], results['mean_defaults'])
plt.axvline(tipping_point, color='red', linestyle='--', label='Tipping Point')
plt.xlabel('Shock Size')
plt.ylabel('Fraction Defaulted')
plt.legend()
plt.show()
```

---

## 🎯 What You'll Get

After running the analysis, you'll have:

### Visualizations (in `results/figures/` or current directory)

1. **network_tipping_points_comparison.png**
   - Compares 4 network types
   - Shows different tipping point behaviors

2. **connectivity_resilience_tradeoff.png**
   - Phase diagram of shock vs connectivity
   - Reveals optimal density

3. **systemic_importance_analysis.png**
   - Identifies critical banks
   - Network centrality correlations

4. **contagion_dashboard.png**
   - Comprehensive 8-panel overview
   - Ready for executive presentation

5. **bayesian_contagion_analysis.png** (if you ran Bayesian analysis)
   - Uncertainty quantification
   - Credible intervals

6. **bayesian_decision_framework.png**
   - Risk-based recommendations
   - Action items

---

## ⚙️ Customize Your Analysis

### Edit Configuration File

```bash
# Open config.yaml in your editor
nano config.yaml  # or vim, code, etc.
```

Key parameters to adjust:

```yaml
network:
  n_banks: 50              # Change network size
  topology: 'scale_free'   # Try different types

balance_sheet:
  capital_ratio_mean: 0.08 # Adjust capital levels
  interbank_fraction: 0.20 # Change interconnectedness

contagion:
  n_simulations: 100       # More = slower but more accurate
  recovery_rate: 0.40      # Adjust recovery assumptions
```

### Or Modify Directly in Code

```python
# In your Python script or notebook
from financial_network_tipping_points import FinancialNetwork

# Create larger network
fn = FinancialNetwork(n_banks=100)  # Default is 50

# Try different topology
fn.generate_network('small_world', k=6, p=0.1)  # Small-world instead of scale-free

# Adjust balance sheets
fn.initialize_balance_sheets(
    capital_ratio_mean=0.10,    # Higher capital
    interbank_fraction=0.25      # More interconnected
)

# Test with different recovery rate
results = fn.analyze_tipping_point(
    shock_range=np.linspace(0, 0.5, 50),
    n_simulations=100,
    recovery_rate=0.60  # Better recovery
)
```

---

## 📈 Common Use Cases

### 1. Stress Test Your Portfolio

```python
# Model your portfolio as a network
fn = FinancialNetwork(n_banks=30)  # Your 30 counterparties
fn.generate_network('core_periphery')  # You're the core
fn.initialize_balance_sheets(capital_ratio_mean=0.08)

# Run stress scenarios
for shock in [0.1, 0.2, 0.3]:
    result = fn.simulate_contagion(shock, shock_banks=[0])  # Shock yourself
    print(f"Shock {shock:.1%} → {result['default_fraction']:.1%} defaults")
```

### 2. Compare Regulatory Scenarios

```python
# Test impact of higher capital requirements
results_low = test_with_capital(0.08)   # Current Basel III
results_high = test_with_capital(0.12)  # Proposed increase

# Compare tipping points
print(f"Low capital TP: {find_tipping_point(results_low):.3f}")
print(f"High capital TP: {find_tipping_point(results_high):.3f}")
```

### 3. Identify Systemic Risks

```python
# Analyze your financial network
fn = FinancialNetwork(n_banks=50)
fn.generate_network('scale_free', m=3)
fn.initialize_balance_sheets()

# Test each bank as shock source
impacts = []
for bank_id in range(fn.n_banks):
    result = fn.simulate_contagion(0.2, [bank_id])
    impacts.append(result['default_fraction'])

# Find most systemic banks
critical_banks = np.argsort(impacts)[-5:]
print(f"Top 5 systemic banks: {critical_banks}")
```

---

## 🐛 Troubleshooting

### Problem: Import errors

```bash
# Solution: Make sure you installed dependencies
pip install -r requirements.txt
# or
make install
```

### Problem: Slow execution

```python
# Solution 1: Reduce simulation count
results = fn.analyze_tipping_point(n_simulations=20)  # Instead of 100

# Solution 2: Reduce shock points
shock_range = np.linspace(0, 0.5, 20)  # Instead of 50

# Solution 3: Smaller network
fn = FinancialNetwork(n_banks=30)  # Instead of 50
```

### Problem: Figures don't save

```bash
# Solution: Create output directories
mkdir -p results/figures
mkdir -p results/tables
# or
make setup
```

### Problem: Virtual environment issues

```bash
# Solution: Recreate environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 💡 Tips for Best Results

### 1. Start Small
- Begin with 30 banks and 20 simulations
- Increase once you understand the behavior
- Use `make quick-analysis` for testing

### 2. Use Version Control
```bash
git init
git add *.py config.yaml requirements.txt
git commit -m "Initial setup"
```

### 3. Document Your Changes
- Keep notes in a Jupyter notebook
- Save interesting parameter combinations
- Screenshot key results

### 4. Validate Results
- Compare with historical crises (2008, etc.)
- Check if tipping points make economic sense
- Test sensitivity to key parameters

---

## 📚 Next Steps

### Learn More
1. Read **README.md** for comprehensive documentation
2. Study **network_tipping_points_methodology.md** for theory
3. Explore the code in `financial_network_tipping_points.py`

### Extend the Analysis
1. Add your own network topology
2. Implement time-varying parameters
3. Connect to real data sources
4. Build a web dashboard

### Get Help
- Check the documentation
- Read academic papers (see README references)
- Review code comments
- Open issues on GitHub

---

## ⏱️ Time Estimates

| Task | Time | Command |
|------|------|---------|
| Setup | 5 min | `bash setup.sh` |
| Quick test | 1 min | `make quick-analysis` |
| Main analysis | 2-3 min | `make run` |
| Bayesian analysis | 1-2 min | `make bayesian` |
| Full analysis + report | 5 min | `make all && make report` |

---

## ✅ Success Checklist

After setup, you should have:

- [ ] All dependencies installed (check with `make check-deps`)
- [ ] Generated at least one visualization
- [ ] Understanding of tipping point concept
- [ ] Ability to customize parameters
- [ ] Saved results in output directory

If you have all checkmarks, you're ready to go! 🎉

---

## 🆘 Quick Help

```bash
# See all available commands
make help

# Get Python help
python -c "from financial_network_tipping_points import FinancialNetwork; help(FinancialNetwork)"

# Test your setup
make test

# Generate a sample report
make report
```

---

**You're all set! Start exploring financial network contagion dynamics!** 🚀

For detailed documentation, see README.md
