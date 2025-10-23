# CS224W Graph Learning - Ethereum Transaction Analysis

This repository contains graph analysis of Ethereum transaction data, exploring network structures and transaction patterns using NetworkX and various graph learning techniques.

## Overview

The project analyzes Ethereum transaction data centered around a specific root address, examining:
- Transaction network topology
- Hub and spoke patterns in financial networks
- Value flow analysis
- Centrality measures and network properties

## Files

- `ethereum_exploration.ipynb` - Main analysis notebook with comprehensive graph analysis
- `ethereum_small.csv` - Sample Ethereum transaction data (149 transactions)
- `commuting_zone_characteristics.csv` - Additional dataset for analysis
- `graph_transformer_analysis.ipynb` - Advanced graph transformer analysis
- `requirements.txt` - Python dependencies

## Key Features

### Graph Construction
- Directed graph with Ethereum addresses as nodes
- Edge features: transaction count, total value, average value
- NetworkX-based implementation

### Analysis Components
- **Graph Statistics**: Node/edge counts, density, degree distributions
- **Hub Analysis**: Identification of high-degree and high-value nodes
- **Visualizations**: Full graph, local subgraphs, degree distributions, value flow
- **Interactive Tools**: Address analysis, subgraph extraction, path finding

### Visualizations
1. Full graph overview with node sizes by degree
2. Local subgraph around root node (2-hop neighborhood)
3. Degree distribution histograms
4. Value flow diagram highlighting high-value transactions

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Open and run `ethereum_exploration.ipynb` for the main analysis

3. Use helper functions for interactive exploration:
   - `get_address_info(address)` - Analyze specific addresses
   - `extract_subgraph_by_criteria()` - Filter graphs by various criteria
   - `find_paths_between_addresses()` - Find transaction paths
   - `analyze_centrality_measures()` - Calculate centrality metrics

## Data Source

The Ethereum transaction data was generated using BigQuery with a recursive CTE to find 2-hop neighbors around a root address, spanning approximately one month of transactions.

## Technologies Used

- Python
- NetworkX (graph analysis)
- Pandas (data manipulation)
- Matplotlib (visualization)
- Jupyter Notebooks

## Course Context

This project is part of CS224W (Graph Learning) coursework, demonstrating practical applications of graph theory and network analysis in blockchain transaction data.
