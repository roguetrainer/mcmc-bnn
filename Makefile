# Financial Network Contagion Analysis - Makefile
# 
# Quick commands for running analysis, tests, and setup
#
# Usage:
#   make setup          - Run setup script
#   make install        - Install basic dependencies
#   make install-full   - Install all dependencies
#   make run            - Run main analysis
#   make bayesian       - Run Bayesian analysis
#   make all            - Run all analyses
#   make test           - Run tests
#   make clean          - Clean generated files
#   make notebook       - Start Jupyter notebook
#   make lint           - Run code quality checks
#   make format         - Format code with black

.PHONY: help setup install install-full run bayesian all test clean notebook lint format docs

# Default target
help:
	@echo "Financial Network Contagion Analysis - Available Commands"
	@echo "=========================================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Run interactive setup script"
	@echo "  make install        - Install core dependencies"
	@echo "  make install-full   - Install all dependencies (including Bayesian)"
	@echo "  make install-dev    - Install development dependencies"
	@echo ""
	@echo "Running Analysis:"
	@echo "  make run            - Run main network contagion analysis"
	@echo "  make bayesian       - Run Bayesian uncertainty analysis"
	@echo "  make all            - Run all analyses"
	@echo ""
	@echo "Development:"
	@echo "  make test           - Run test suite"
	@echo "  make lint           - Check code quality"
	@echo "  make format         - Auto-format code with black"
	@echo "  make notebook       - Start Jupyter notebook server"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean          - Remove generated files"
	@echo "  make clean-all      - Remove all generated files and cache"
	@echo "  make docs           - Generate documentation"
	@echo "  make package        - Create distribution package"
	@echo ""

# Setup
setup:
	@echo "Running setup script..."
	bash setup.sh

# Installation targets
install:
	@echo "Installing core dependencies..."
	pip3 install -r requirements.txt --break-system-packages || pip install -r requirements.txt

install-full:
	@echo "Installing all dependencies (including Bayesian frameworks)..."
	bash setup.sh --full

install-dev:
	@echo "Installing development dependencies..."
	bash setup.sh --dev

# Run analyses
run:
	@echo "Running main network contagion analysis..."
	python3 financial_network_tipping_points.py

bayesian:
	@echo "Running Bayesian uncertainty analysis..."
	python3 bayesian_contagion_extension.py

all: run bayesian
	@echo "All analyses complete!"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-quick:
	@echo "Running quick tests..."
	pytest tests/ -v -k "not slow"

# Code quality
lint:
	@echo "Running linters..."
	flake8 *.py --max-line-length=100 --ignore=E203,W503
	mypy *.py --ignore-missing-imports

format:
	@echo "Formatting code with black..."
	black *.py --line-length=100

# Jupyter
notebook:
	@echo "Starting Jupyter notebook..."
	jupyter notebook

# Documentation
docs:
	@echo "Generating documentation..."
	@mkdir -p docs/build
	sphinx-build -b html docs/source docs/build

# Cleaning
clean:
	@echo "Cleaning generated files..."
	rm -rf results/figures/*
	rm -rf results/tables/*
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf *.pyc
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "Clean complete!"

clean-all: clean
	@echo "Deep cleaning..."
	rm -rf venv/
	rm -rf data/processed/*
	rm -rf logs/*
	rm -rf results/models/*
	@echo "Deep clean complete!"

# Packaging
package:
	@echo "Creating distribution package..."
	@mkdir -p dist
	@tar -czf dist/financial-network-contagion-$(shell date +%Y%m%d).tar.gz \
		*.py *.md requirements.txt setup.sh Makefile config.yaml \
		--exclude=venv --exclude=__pycache__ --exclude=.git
	@echo "Package created in dist/"

# Quick analysis with custom parameters
quick-analysis:
	@echo "Running quick analysis (30 banks, 20 simulations)..."
	python3 << 'EOF'
from financial_network_tipping_points import FinancialNetwork
import numpy as np
fn = FinancialNetwork(n_banks=30)
fn.generate_network('scale_free', m=3)
fn.initialize_balance_sheets()
results = fn.analyze_tipping_point(shock_range=np.linspace(0, 0.5, 20), n_simulations=20)
print(f"Quick analysis complete: Mean tipping point ≈ {results['shock_range'][np.argmax(np.gradient(results['mean_defaults']))]:.3f}")
EOF

# Check dependencies
check-deps:
	@echo "Checking installed dependencies..."
	@python3 -c "import numpy, scipy, networkx, matplotlib, seaborn, pandas; print('✓ Core dependencies OK')"
	@python3 -c "import pymc; print('✓ PyMC installed')" 2>/dev/null || echo "✗ PyMC not installed"
	@python3 -c "import tensorflow_probability; print('✓ TensorFlow Probability installed')" 2>/dev/null || echo "✗ TFP not installed"

# Create sample data
sample-data:
	@echo "Generating sample synthetic data..."
	@mkdir -p data/raw
	python3 << 'EOF'
import numpy as np
import pandas as pd
np.random.seed(42)
data = {
    'bank_id': range(50),
    'capital_ratio': np.random.normal(0.08, 0.02, 50),
    'total_assets': np.random.lognormal(10, 1, 50),
    'interbank_assets': np.random.uniform(0.1, 0.3, 50)
}
df = pd.DataFrame(data)
df.to_csv('data/raw/sample_bank_data.csv', index=False)
print("Sample data created: data/raw/sample_bank_data.csv")
EOF

# Performance benchmark
benchmark:
	@echo "Running performance benchmark..."
	python3 << 'EOF'
import time
import numpy as np
from financial_network_tipping_points import FinancialNetwork

start = time.time()
fn = FinancialNetwork(n_banks=50)
fn.generate_network('scale_free', m=3)
fn.initialize_balance_sheets()
results = fn.analyze_tipping_point(shock_range=np.linspace(0, 0.5, 30), n_simulations=50)
elapsed = time.time() - start
print(f"Benchmark: {elapsed:.2f} seconds for 50 banks, 30 shock points, 50 simulations")
print(f"Performance: {(50*30*50)/elapsed:.0f} simulations/second")
EOF

# Generate report
report:
	@echo "Generating analysis report..."
	@mkdir -p reports
	python3 << 'EOF'
from datetime import datetime
import numpy as np
from financial_network_tipping_points import FinancialNetwork

# Run analysis
fn = FinancialNetwork(n_banks=50)
fn.generate_network('scale_free', m=3)
fn.initialize_balance_sheets()
results = fn.analyze_tipping_point(np.linspace(0, 0.5, 30), n_simulations=50)

# Calculate metrics
tp = results['shock_range'][np.argmax(np.gradient(results['mean_defaults']))]
metrics = fn.compute_network_metrics()

# Generate report
report = f"""
Financial Network Contagion Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

Network Configuration:
  - Number of banks: {fn.n_banks}
  - Network type: Scale-Free (Barabási-Albert)
  - Network density: {metrics['density']:.3f}
  - Average degree: {metrics['avg_degree']:.2f}

Key Findings:
  - Estimated tipping point: {tp:.3f}
  - Mean capital ratio: {np.mean(fn.capital_buffers):.3f}
  - Maximum contagion: {np.max(results['mean_defaults']):.1%}

Risk Assessment:
  - Current risk level: {'HIGH' if tp < 0.15 else 'MEDIUM' if tp < 0.25 else 'LOW'}
  - Systemic vulnerability: {'Elevated' if metrics['density'] > 0.15 else 'Moderate'}
  
Recommendations:
  - {'Increase capital buffers' if tp < 0.20 else 'Maintain monitoring'}
  - {'Enhance stress testing' if metrics['density'] > 0.15 else 'Continue current practices'}
"""

with open('reports/analysis_report.txt', 'w') as f:
    f.write(report)

print("Report generated: reports/analysis_report.txt")
print(report)
EOF

# Interactive mode
interactive:
	@echo "Starting interactive Python session with imports..."
	python3 -i << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from financial_network_tipping_points import FinancialNetwork

print("\nFinancial Network Contagion - Interactive Mode")
print("=" * 50)
print("\nAvailable objects:")
print("  - FinancialNetwork class")
print("  - numpy as np")
print("  - matplotlib.pyplot as plt")
print("  - networkx as nx")
print("\nExample:")
print("  fn = FinancialNetwork(n_banks=30)")
print("  fn.generate_network('scale_free', m=3)")
print("  fn.initialize_balance_sheets()")
print("  results = fn.analyze_tipping_point()")
print("")
EOF
