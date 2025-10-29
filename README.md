# Ethereum Transaction Graph Analysis Dashboard

A comprehensive web dashboard showcasing Ethereum transaction network analysis using graph theory and network analysis techniques.

## 🚀 Quick Deploy to Vercel

### Option 1: Deploy via Vercel CLI (Recommended)

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy the project**:
   ```bash
   vercel --prod
   ```

### Option 2: Deploy via Vercel Dashboard

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository or drag and drop the project folder
4. Vercel will automatically detect it as a static site
5. Click "Deploy"

## 📊 Project Overview

This dashboard presents a comprehensive analysis of Ethereum transaction data including:

- **149 transactions** analyzed over 30 days
- **91 unique addresses** in the network
- **Interactive visualizations** showing network evolution
- **Graph theory analysis** with centrality measures
- **Temporal analysis** of network growth

## 🔬 Methodology

The analysis employs advanced graph theory techniques:

- **NetworkX** for graph construction and analysis
- **Directed weighted graphs** with transaction metadata
- **Centrality measures**: Degree, Betweenness, PageRank
- **Temporal evolution** tracking network growth
- **Interactive visualizations** using Chart.js and D3.js

## 📈 Key Findings

- **Hub-and-spoke pattern**: One dominant address receives 75% of transactions
- **Low network density**: 0.0112, typical of financial networks
- **Centralized value flow**: Significant concentration in high-volume edges
- **Temporal growth**: Steady expansion from 11 to 91 nodes over 30 days

## 🛠️ Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Visualization**: Chart.js, D3.js
- **Styling**: Tailwind CSS
- **Deployment**: Vercel
- **Analysis**: Python (NetworkX, Pandas, Matplotlib)

## 📁 Project Structure

```
├── index.html          # Main dashboard page
├── package.json        # Project dependencies
├── vercel.json         # Vercel configuration
├── ethereum_small.csv  # Transaction data
├── ethereum_exploration.ipynb  # Analysis notebook
└── ethereum_graph_evolution.gif # Network animation
```

## 🎯 Features

- **Responsive Design**: Works on desktop and mobile
- **Interactive Charts**: Degree distribution, growth over time, value analysis
- **Network Animation**: Shows temporal evolution of the transaction network
- **Comprehensive Methodology**: Detailed explanation of analysis techniques
- **Professional Presentation**: Ready for stakeholder presentations

## 🔗 Live Demo

Once deployed, your dashboard will be available at:
`https://your-project-name.vercel.app`

## 📝 Customization

To customize the dashboard:

1. **Update data**: Replace `ethereum_small.csv` with your data
2. **Modify charts**: Edit the JavaScript section in `index.html`
3. **Change styling**: Update CSS classes and Tailwind utilities
4. **Add sections**: Extend the HTML structure as needed

## 🤝 Contributing

This project demonstrates blockchain transaction analysis techniques. Feel free to:
- Fork the repository
- Add new analysis methods
- Improve visualizations
- Extend to other blockchain networks

## 📄 License

MIT License - feel free to use for research and educational purposes.

---

**Built with ❤️ for blockchain network analysis**