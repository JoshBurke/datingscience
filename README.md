# Dating History Analysis

This project analyzes your dating history data to determine which factors contribute most to your relationship satisfaction, helping you make more informed dating decisions in the future.

## Features

- **Exploratory Data Analysis**: Examine correlations between various factors and relationship satisfaction
- **Regression Analysis**: Run multiple regression models to identify key predictors
- **Visualization**: Generate plots to better understand your dating patterns
- **Weighted Criteria**: Calculate optimal weights for different dating criteria based on your past experiences

## Setup

1. Make sure you have Python 3.7+ installed
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure your dating history CSV file is in the same directory as the script (default filename: `history.csv`)

## Usage

Run the analysis script:

```
python dating_analysis.py
```

The script will:
1. Load and analyze your dating history data
2. Generate visualizations saved as PNG files:
   - `correlation_matrix.png`: Correlation between all factors
   - `top_factors_scatter.png`: Scatter plot of top factors vs. satisfaction
   - `feature_importance.png`: Coefficient sizes for each factor
3. Print insights and recommendations based on the analysis

## Data Format

Your dating history data should be in CSV format with the following columns:
- `Relationship_ID`: Unique identifier for each relationship
- Various attribute columns (e.g., Communication, Intellectual_Connection, etc.)
- `Overall_Relationship_Satisfaction`: Target variable (0-1 scale)

## Customization

You can customize the analysis by:
- Modifying model parameters in the script
- Adding new analysis functions
- Adjusting thresholds for high/low satisfaction relationships

## Understanding the Results

The script will provide:
- The most important factors that correlate with your relationship satisfaction
- Recommended weights for different dating criteria
- Characteristics of your most and least satisfying relationships
- Statistical analysis of how different factors predict satisfaction

Use these insights to make more deliberate choices in your future dating life! 
