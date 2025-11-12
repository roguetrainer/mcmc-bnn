#!/bin/bash
#
# Financial Network Contagion Analysis - Setup Script
# 
# This script sets up the Python environment and installs all necessary dependencies
# for analyzing non-linear tipping points in financial networks.
#
# Usage:
#   bash setup.sh [--basic|--full|--dev]
#
# Options:
#   --basic    Install only core dependencies (default)
#   --full     Install all dependencies including Bayesian frameworks
#   --dev      Install development dependencies (testing, linting)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print formatted messages
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Parse command line arguments
INSTALL_MODE="basic"
if [ "$1" == "--full" ]; then
    INSTALL_MODE="full"
elif [ "$1" == "--dev" ]; then
    INSTALL_MODE="dev"
elif [ "$1" == "--basic" ]; then
    INSTALL_MODE="basic"
fi

print_header "Financial Network Contagion Analysis - Setup"
echo ""
print_info "Installation mode: $INSTALL_MODE"
echo ""

# Check Python version
print_info "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 8 ]; then
        print_success "Python $PYTHON_VERSION found"
        PYTHON_CMD="python3"
    else
        print_error "Python 3.8 or higher required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if pip is available
print_info "Checking pip..."
if ! command -v pip3 &> /dev/null; then
    print_warning "pip3 not found. Attempting to install..."
    $PYTHON_CMD -m ensurepip --upgrade
fi

if command -v pip3 &> /dev/null; then
    print_success "pip3 available"
    PIP_CMD="pip3"
else
    print_error "Could not install or find pip3"
    exit 1
fi

# Upgrade pip
print_info "Upgrading pip..."
$PIP_CMD install --upgrade pip

# Create virtual environment (optional but recommended)
read -p "Create a virtual environment? (recommended) [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    print_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Skipping creation."
    else
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi
    
    print_info "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
    
    # Use the venv pip
    PIP_CMD="pip"
fi

# Install basic dependencies
print_header "Installing Core Dependencies"

print_info "Installing numpy, scipy, networkx, matplotlib, seaborn..."
$PIP_CMD install numpy>=1.21.0 scipy>=1.7.0 networkx>=2.6.0 matplotlib>=3.4.0 seaborn>=0.11.0

print_info "Installing pandas and statsmodels..."
$PIP_CMD install pandas>=1.3.0 statsmodels>=0.13.0

print_info "Installing utility packages..."
$PIP_CMD install tqdm>=4.62.0 pyyaml>=5.4.0 python-dateutil>=2.8.0

print_success "Core dependencies installed"

# Install Jupyter if requested
read -p "Install Jupyter Notebook for interactive analysis? [Y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
    print_info "Installing Jupyter..."
    $PIP_CMD install jupyter>=1.0.0 ipykernel>=6.0.0 notebook>=6.4.0
    print_success "Jupyter installed"
fi

# Full installation mode
if [ "$INSTALL_MODE" == "full" ]; then
    print_header "Installing Bayesian Inference Frameworks"
    
    echo "Choose a Bayesian framework to install:"
    echo "  1) PyMC (recommended - NumPyro backend)"
    echo "  2) TensorFlow Probability"
    echo "  3) Pyro (PyTorch-based)"
    echo "  4) All of the above"
    echo "  5) Skip"
    read -p "Enter choice [1-5]: " BAYES_CHOICE
    
    case $BAYES_CHOICE in
        1)
            print_info "Installing PyMC with NumPyro backend..."
            $PIP_CMD install pymc>=5.0.0 arviz>=0.14.0 numpyro>=0.12.0
            print_success "PyMC installed"
            ;;
        2)
            print_info "Installing TensorFlow Probability..."
            $PIP_CMD install tensorflow-probability>=0.19.0 tensorflow>=2.10.0
            print_success "TensorFlow Probability installed"
            ;;
        3)
            print_info "Installing Pyro..."
            $PIP_CMD install pyro-ppl>=1.8.0 torch>=1.13.0
            print_success "Pyro installed"
            ;;
        4)
            print_info "Installing all Bayesian frameworks..."
            $PIP_CMD install pymc>=5.0.0 arviz>=0.14.0 numpyro>=0.12.0
            $PIP_CMD install tensorflow-probability>=0.19.0 tensorflow>=2.10.0
            $PIP_CMD install pyro-ppl>=1.8.0 torch>=1.13.0
            print_success "All frameworks installed"
            ;;
        *)
            print_info "Skipping Bayesian frameworks"
            ;;
    esac
    
    # Performance optimization
    print_info "Installing performance optimization packages..."
    $PIP_CMD install numba>=0.55.0 joblib>=1.1.0
    print_success "Performance packages installed"
    
    # Export capabilities
    print_info "Installing export capabilities (Excel, PowerPoint, PDF)..."
    $PIP_CMD install openpyxl>=3.0.0 xlsxwriter>=3.0.0 python-pptx>=0.6.21 reportlab>=3.6.0
    print_success "Export packages installed"
fi

# Development mode
if [ "$INSTALL_MODE" == "dev" ]; then
    print_header "Installing Development Dependencies"
    
    print_info "Installing testing frameworks..."
    $PIP_CMD install pytest>=7.0.0 pytest-cov>=3.0.0
    
    print_info "Installing code quality tools..."
    $PIP_CMD install black>=22.0.0 flake8>=4.0.0 mypy>=0.950
    
    print_info "Installing documentation tools..."
    $PIP_CMD install sphinx>=4.5.0 sphinx-rtd-theme>=1.0.0
    
    print_success "Development dependencies installed"
fi

# Verify installation
print_header "Verifying Installation"

print_info "Testing imports..."

$PYTHON_CMD << EOF
import sys
import numpy
import scipy
import networkx
import matplotlib
import seaborn
import pandas

print(f"✓ NumPy {numpy.__version__}")
print(f"✓ SciPy {scipy.__version__}")
print(f"✓ NetworkX {networkx.__version__}")
print(f"✓ Matplotlib {matplotlib.__version__}")
print(f"✓ Seaborn {seaborn.__version__}")
print(f"✓ Pandas {pandas.__version__}")

# Test Bayesian frameworks if full install
try:
    import pymc
    print(f"✓ PyMC {pymc.__version__}")
except ImportError:
    pass

try:
    import tensorflow_probability as tfp
    print(f"✓ TensorFlow Probability {tfp.__version__}")
except ImportError:
    pass

try:
    import pyro
    print(f"✓ Pyro {pyro.__version__}")
except ImportError:
    pass

print("\n✓ All core packages imported successfully!")
EOF

if [ $? -eq 0 ]; then
    print_success "Verification complete - all packages working"
else
    print_error "Verification failed - some packages may not be installed correctly"
    exit 1
fi

# Create directory structure
print_header "Setting Up Directory Structure"

print_info "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p results/figures
mkdir -p results/tables
mkdir -p results/models
mkdir -p logs
mkdir -p notebooks

print_success "Directory structure created"

# Create a sample config file
print_info "Creating sample configuration file..."

cat > config.yaml << 'EOF'
# Financial Network Contagion Analysis - Configuration

# Network parameters
network:
  n_banks: 50
  topology: "scale_free"  # Options: erdos_renyi, scale_free, small_world, core_periphery
  density: 0.1  # For Erdos-Renyi
  m: 3  # For Barabasi-Albert (scale-free)
  k: 6  # For Watts-Strogatz (small-world)
  p: 0.1  # Rewiring probability for Watts-Strogatz

# Balance sheet parameters
balance_sheet:
  capital_ratio_mean: 0.08
  capital_ratio_std: 0.02
  interbank_fraction: 0.20

# Contagion parameters
contagion:
  shock_range: [0.0, 0.5]
  n_shock_points: 50
  recovery_rate: 0.40
  n_simulations: 100

# Bayesian inference parameters
bayesian:
  n_samples: 2000
  n_tune: 1000
  n_chains: 4
  random_seed: 42

# Output parameters
output:
  figures_dir: "results/figures"
  tables_dir: "results/tables"
  models_dir: "results/models"
  figure_format: "png"
  figure_dpi: 300
EOF

print_success "Configuration file created: config.yaml"

# Test scripts
print_header "Testing Scripts"

if [ -f "financial_network_tipping_points.py" ]; then
    read -p "Run a quick test of the main script? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        print_info "Running test simulation (this may take 1-2 minutes)..."
        $PYTHON_CMD financial_network_tipping_points.py
        print_success "Test completed successfully!"
    fi
else
    print_warning "Main script not found in current directory"
fi

# Final instructions
print_header "Setup Complete!"

echo ""
print_success "Environment successfully configured!"
echo ""
print_info "Next steps:"
echo ""
echo "  1. If you created a virtual environment, activate it:"
echo "     ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "  2. Run the main analysis:"
echo "     ${GREEN}python financial_network_tipping_points.py${NC}"
echo ""
echo "  3. Run the Bayesian extension:"
echo "     ${GREEN}python bayesian_contagion_extension.py${NC}"
echo ""
echo "  4. Explore interactively with Jupyter:"
echo "     ${GREEN}jupyter notebook${NC}"
echo ""
echo "  5. Customize parameters in ${GREEN}config.yaml${NC}"
echo ""
print_info "For help and documentation:"
echo "  - See README.md for detailed usage instructions"
echo "  - See network_tipping_points_methodology.md for technical details"
echo ""
print_info "Directory structure:"
echo "  - ${GREEN}data/raw/${NC}          Place your input data here"
echo "  - ${GREEN}data/processed/${NC}    Processed data will be saved here"
echo "  - ${GREEN}results/figures/${NC}   Output visualizations"
echo "  - ${GREEN}results/tables/${NC}    Output tables and metrics"
echo "  - ${GREEN}results/models/${NC}    Saved models"
echo "  - ${GREEN}notebooks/${NC}         Jupyter notebooks"
echo "  - ${GREEN}logs/${NC}              Log files"
echo ""

if [ -d "venv" ]; then
    print_warning "Remember to activate your virtual environment before each session:"
    echo "  ${GREEN}source venv/bin/activate${NC}"
fi

echo ""
print_success "Happy analyzing! 📊"
echo ""
