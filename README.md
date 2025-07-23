# F1 Winner Predictor 🏎️

A machine learning project that predicts Formula 1 race winners using historical race data and performance analytics.

## 📋 Project Overview

This project aims to predict Formula 1 race winners by analyzing historical racing data from 2016 to 2025. Using various machine learning techniques and comprehensive Formula 1 datasets, the model considers multiple factors such as driver performance, constructor strength, circuit characteristics, and historical trends to make accurate predictions.

## 🎯 Features

- **Historical Data Analysis**: Comprehensive analysis of F1 race results from 2016-2025
- **Driver Performance Metrics**: Analysis of driver statistics including grid positions, final positions, points, and lap times
- **Constructor Analysis**: Team performance evaluation and trends
- **Circuit-Specific Insights**: Track-specific performance patterns and characteristics
- **Machine Learning Predictions**: Advanced ML models for race winner prediction
- **Interactive Visualizations**: Data visualization and exploratory data analysis

## 📊 Dataset

The project uses multiple datasets containing F1 race results:

- `f1_results_2016_2025.csv` - Complete dataset (2016-2025)
- `f1_results_2019_2025.csv` - Recent era dataset (2019-2025)
- `f1_results_2022_2025.csv` - Current regulation era (2022-2025)

### Data Features

Each dataset includes the following key features:
- **Season Information**: Year and race round number
- **Race Details**: Race name, date, and circuit information
- **Driver Data**: Name, nationality, and constructor
- **Performance Metrics**: Grid position, final position, points earned
- **Race Statistics**: Laps completed, race time, fastest lap, and race status

## 🛠️ Technology Stack

- **Python**: Primary programming language
- **FastF1**: Official F1 data API for real-time and historical data
- **PyTorch**: Deep learning framework for model development
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive development and analysis

## 📁 Project Structure

```
F1-Winner-Predictor/
├── README.md
├── LICENSE
├── requirements.txt
├── data/                          # Raw data storage
│   └── fastf1_cache/             # FastF1 API cache
├── notebooks/                     # Jupyter notebooks
│   ├── EDA.ipynb                 # Exploratory Data Analysis
│   ├── f1_results_*.csv          # Historical race datasets
│   └── data/                     # Notebook-specific data
├── src/                          # Source code (to be developed)
└── ff1_cache/                    # Additional FastF1 cache data
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/divyanshg03/F1-Winner-Predictor.git
   cd F1-Winner-Predictor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv f1_env
   source f1_env/bin/activate  # On Windows: f1_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Start with the EDA notebook**
   - Open `notebooks/EDA.ipynb` to begin exploring the data

## 📈 Usage

### Exploratory Data Analysis

Start by running the EDA notebook to understand the dataset:

```python
# Load and explore the data
import pandas as pd
import fastf1

# Load race results
df = pd.read_csv('notebooks/f1_results_2022_2025.csv')
print(df.head())
```

### Model Training (Coming Soon)

The project will include various machine learning models:
- Logistic Regression
- Random Forest
- Neural Networks
- Ensemble Methods

## 🎯 Model Features

The prediction model considers multiple factors:

- **Driver Performance**: Historical win rates, podium finishes, points per race
- **Constructor Strength**: Team performance, car reliability, development trends
- **Circuit Characteristics**: Track-specific performance, weather conditions
- **Current Season Form**: Recent race performance and momentum
- **Grid Position**: Qualifying performance and starting position advantage
- **Historical Patterns**: Seasonal trends and championship standings

## 📊 Results

*Results and model performance metrics will be updated as the project develops.*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastF1**: For providing comprehensive F1 data API
- **Formula 1**: For the exciting sport that inspired this project
- **FIA**: For maintaining detailed race records and statistics

## 📞 Contact

- **Author**: Divyansh Gupta
- **GitHub**: [@divyanshg03](https://github.com/divyanshg03)
- **Project Link**: [https://github.com/divyanshg03/F1-Winner-Predictor](https://github.com/divyanshg03/F1-Winner-Predictor)

## 🔮 Future Enhancements

- [ ] Real-time race prediction during live sessions
- [ ] Weather data integration
- [ ] Tire strategy analysis
- [ ] Driver market value prediction
- [ ] Web application for interactive predictions
- [ ] API development for external integrations

---

**Note**: This project is for educational and research purposes. Predictions are based on historical data and should not be used for gambling or commercial betting purposes.