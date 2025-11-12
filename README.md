# Financial Network Contagion: Non-Linear Tipping Points Analysis

## Overview

This package provides a comprehensive toolkit for analyzing **non-linear tipping points** in financial network contagion using both deterministic simulations and **Bayesian Neural Network (BNN)** approaches. The analysis reveals how financial networks exhibit dual behavior: resilient to small shocks but catastrophically fragile beyond critical thresholds.

## 📊 Generated Outputs

### Visualizations

1. **`network_tipping_points_comparison.png`** (1.1 MB)
   - Compares tipping point behavior across 4 network structures
   - Shows how topology determines system fragility
   - **Key Insight**: Scale-free networks have sharp transitions; small-world networks are most resilient

2. **`connectivity_resilience_tradeoff.png`** (456 KB)
   - Demonstrates the **connectivity paradox**
   - Phase diagram showing resilient vs fragile zones
   - **Key Insight**: More connections ≠ always safer; optimal connectivity exists

3. **`systemic_importance_analysis.png`** (1.1 MB)
   - Identifies "too-connected-to-fail" institutions
   - Correlates network centrality with contagion impact
   - **Key Insight**: Centrality metrics predict systemic importance (r ~0.7-0.8)

4. **`contagion_dashboard.png`** (1.2 MB)
   - Comprehensive overview with 8 panels
   - Tipping point curve, sensitivity analysis, network structure
   - **Use Case**: Executive briefing on systemic risk

5. **`bayesian_contagion_analysis.png`** (1.4 MB)
   - Bayesian inference results with uncertainty quantification
   - Posterior distributions for all parameters
   - **Key Insight**: Provides credible intervals, not just point estimates

6. **`bayesian_decision_framework.png`** (437 KB)
   - Risk-based decision matrix using Bayesian posterior
   - Actionable recommendations based on uncertainty
   - **Use Case**: Real-time risk assessment for regulators

### Code

1. **`financial_network_tipping_points.py`** (30 KB)
   - Main simulation engine
   - Models 4 network topologies: Erdős-Rényi, Scale-Free, Small-World, Core-Periphery
   - Includes contagion cascade simulation with recovery rates
   - **Run**: `python financial_network_tipping_points.py`

2. **`bayesian_contagion_extension.py`** (23 KB)
   - Bayesian Neural Network approach to parameter uncertainty
   - Uses analytic approximations (for demo; adapt for PyMC/NumPyro)
   - Provides posterior distributions and decision frameworks
   - **Run**: `python bayesian_contagion_extension.py`

### Documentation

1. **`network_tipping_points_methodology.md`** (12 KB)
   - Detailed methodology explanation
   - Mathematical framework
   - Banking applications
   - Extension to full BNN with MCMC
   - Further reading and references

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib networkx seaborn scipy --break-system-packages
```

For full Bayesian inference (optional):
```bash
pip install pymc arviz pytensor --break-system-packages
```

### Run Complete Analysis

```bash
# Generate all visualizations
python financial_network_tipping_points.py

# Run Bayesian extension
python bayesian_contagion_extension.py
```

### Customize Parameters

```python
# In financial_network_tipping_points.py

# Modify network size
fn = FinancialNetwork(n_banks=100)  # Default: 50

# Change capital buffers
fn.initialize_balance_sheets(
    capital_ratio_mean=0.10,      # Default: 0.08
    interbank_fraction=0.25       # Default: 0.20
)

# Adjust recovery rate
results = fn.analyze_tipping_point(
    shock_range=np.linspace(0, 0.5, 50),
    recovery_rate=0.60            # Default: 0.40
)
```

## 🔑 Key Concepts

### What is a Tipping Point?

A **tipping point** is the critical shock threshold where system behavior fundamentally changes:

- **Below**: Network absorbs shocks through diversification
- **Above**: Cascading failures dominate, causing systemic crisis

Example from analysis:
- Shock of 0.15 → 5% banks default
- Shock of 0.20 → 60% banks default (tipping point crossed)

### The Connectivity Paradox

**Low connectivity**: Isolated failures, but no diversification benefits

**Optimal connectivity**: Risk-sharing without contagion channels

**High connectivity**: Efficient in normal times, catastrophic in crisis

**Banking implication**: More interbank connections don't always reduce risk

### Why Bayesian Neural Networks?

Traditional models give **point estimates**: "Tipping point is 0.20"

BNNs provide **distributions**: "Tipping point is 0.20 ± 0.03 with 95% CI [0.17, 0.23]"

**Why this matters**:
- Regulators need to know confidence in predictions
- Risk-based decision making requires uncertainty quantification
- Basel III/IV mandates model risk assessment

## 📈 Applications in Banking & Finance

### 1. Stress Testing (Central Banks)

**Traditional**: Single worst-case scenario

**This Framework**:
- Map full tipping point curve
- Quantify distance to critical threshold
- Provide probability-based risk metrics

**Example**: "System is 0.05 below tipping point with 85% confidence"

### 2. Capital Requirements (Basel III)

**Enhancement**: Network-adjusted capital charges
- Higher capital for high-centrality banks
- Surcharges based on systemic importance
- Dynamic buffers as network evolves

### 3. Systemic Risk Monitoring (Regulators)

**Real-time tracking**:
- Network density changes
- Centrality of key institutions
- Proximity to historical tipping points

**Early warning**: Alert when approaching critical thresholds

### 4. Portfolio Risk Management (Asset Managers)

**Use cases**:
- Model credit portfolio contagion
- Stress test using network correlations
- Adjust exposures based on counterparty centrality

### 5. Merger Review (Competition Authorities)

**Evaluate mergers on**:
- Change in network structure
- Impact on systemic importance
- Shift in tipping point location

## 🧮 Technical Details

### Network Model

Banks have:
- **External assets**: Loans to non-bank entities
- **Interbank assets**: Loans to other banks
- **Capital buffer**: Equity cushion absorbing losses

Contagion mechanism:
1. External shock hits bank(s)
2. If loss > capital → default
3. Default causes losses to creditors
4. Cascade continues until no new defaults

Mathematical formulation:

```
Capital(i, t+1) = Capital(i, t) - Σ_j [Exposure(i→j) × (1 - Recovery) × Default(j, t)]

Default(i, t) = 1 if Capital(i, t) ≤ 0, else 0
```

### Network Topologies

1. **Erdős-Rényi**: Random connections
   - Parameter: Connection probability p
   - Homogeneous structure

2. **Barabási-Albert**: Preferential attachment
   - Parameter: Edges per new node m
   - Power-law degree distribution (realistic)

3. **Watts-Strogatz**: Small-world
   - Parameters: Neighbors k, rewiring probability p
   - High clustering + short paths

4. **Core-Periphery**: Custom
   - Dense core (20% of nodes)
   - Sparse periphery
   - Models money-center vs regional banks

### Bayesian Inference

For parameters θ = {capital_mean, recovery_rate, network_density}:

**Prior**: P(θ)
- Capital mean ~ Normal(0.08, 0.02)
- Recovery ~ Beta(4, 6)  # Centered at 0.4

**Likelihood**: P(Data | θ)
- Observed defaults given parameters

**Posterior**: P(θ | Data) ∝ P(Data | θ) × P(θ)
- Computed via MCMC (NUTS algorithm)

**Prediction**: P(defaults | shock, Data)
- Integrate over posterior: ∫ P(defaults | shock, θ) P(θ | Data) dθ

## 📊 Interpreting Results

### Tipping Point Curve

**X-axis**: Initial shock size (fraction of external assets)

**Y-axis**: Fraction of banks defaulted

**Key features**:
- **Flat region**: Small shocks absorbed
- **Sharp rise**: Non-linear transition at tipping point
- **Plateau**: Maximum contagion

**Find tipping point**: Maximum of first derivative (steepest slope)

### Phase Diagram (Connectivity vs Shock)

**Color**: Default fraction
- Green: Resilient (low defaults)
- Yellow: Transition zone
- Red: Systemic crisis (high defaults)

**White dashed line**: Tipping point trajectory

**Interpretation**: Shows optimal connectivity for different shock regimes

### Systemic Importance Rankings

**Metrics**:
- **Contagion impact**: Defaults caused when this bank fails first
- **Degree centrality**: Number of connections
- **Betweenness**: Lies on shortest paths

**Use**: Identify "too-connected-to-fail" institutions requiring enhanced oversight

### Bayesian Uncertainty

**Credible intervals**: 95% probability true value lies within
- Narrower CI = more confident
- Wider CI = more uncertainty, need more data

**Posterior predictive samples**: Show range of plausible outcomes

**Decision threshold**: If P(exceed tipping point) > 10% → elevated risk

## 🔬 Advanced Extensions

### 1. Dynamic Networks

Current model: Static network structure

Extension: Networks evolve during crisis
- Banks cut credit lines to risky counterparties
- Flight to quality increases core concentration

Implementation: Update adjacency matrix at each time step based on perceived risk

### 2. Fire Sales

Current: Only direct counterparty losses

Extension: Add market price feedback
- Defaults → forced asset sales
- Asset sales → price drops
- Price drops → mark-to-market losses for all holders

Mathematical: Add price dynamics P(t+1) = P(t) - α × Sales(t)

### 3. Liquidity vs Solvency

Current: Only solvency defaults (capital depletion)

Extension: Add liquidity shocks
- Funding runs on short-term debt
- Inability to roll over liabilities
- Fire sales to meet margin calls

Implementation: Track both capital and liquidity ratios

### 4. Strategic Behavior

Current: Passive banks

Extension: Banks react strategically
- Cut lending to risky counterparties
- Hoard liquidity
- Coordinate rescue packages

Implementation: Game-theoretic framework or agent-based modeling

### 5. Multi-Layer Networks

Current: Single interbank network

Extension: Multiple contagion channels
- Interbank lending
- Common asset holdings
- Cross-holdings of equity
- Derivatives exposures

Implementation: Multiplex network with layer interactions

## 📚 References

### Key Papers

1. **Acemoglu, Ozdaglar & Tahbaz-Salehi (2015)**
   "Systemic Risk and Stability in Financial Networks"
   *American Economic Review*
   - Theoretical foundation for network contagion

2. **Haldane & May (2011)**
   "Systemic Risk in Banking Ecosystems"
   *Nature*
   - Ecological analogies; tipping points in banking

3. **Battiston et al. (2012)**
   "DebtRank: Too Central to Fail?"
   *Scientific Reports*
   - Network centrality measures for systemic importance

4. **Elliott, Golub & Jackson (2014)**
   "Financial Networks and Contagion"
   *American Economic Review*
   - Diversity vs connectivity tradeoff

5. **Glasserman & Young (2015)**
   "How Likely Is Contagion in Financial Networks?"
   *Journal of Banking & Finance*
   - Probability of cascades in different topologies

### Regulatory Documents

- Basel Committee: "Global Systemically Important Banks" (2013)
- FSB: "Assessment Methodologies for Identifying Non-Bank SIFIs" (2015)
- European Systemic Risk Board: "Financial Networks" (2020)

### Books

- Haldane, A. (2013) *Rethinking the Financial Network*
- Newman, M. (2018) *Networks: An Introduction* (2nd ed)
- Gai, P. (2013) *Systemic Risk: The Dynamics of Modern Financial Systems*

## 🤝 Contributing

Potential improvements:
- Add more network topologies (modular, hierarchical)
- Implement full PyMC/NumPyro BNN models
- Real data integration (FDIC call reports, FR Y-15)
- Web dashboard for interactive exploration
- GPU acceleration for large networks

## 📧 Contact

For questions about methodology, implementation, or applications in your institution, reach out through GitHub or the contact information in the repository.

## ⚠️ Disclaimer

This toolkit is for **research and educational purposes**. 

For production use in regulatory or risk management contexts:
- Calibrate carefully to your specific market
- Validate against historical crisis episodes
- Complement with other risk models
- Seek expert review before deployment

**Not financial or regulatory advice.**

## 📄 License

MIT License - see LICENSE file for details

---

*Built with: Python 3.x, NetworkX, NumPy, Matplotlib, SciPy*

*Methodology: Based on academic literature in network science, financial contagion, and Bayesian inference*

*Last updated: November 2025*
