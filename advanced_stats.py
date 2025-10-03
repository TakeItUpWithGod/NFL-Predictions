import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedNFLStats:
    def __init__(self, data_path: str):
        """Initialize with advanced statistical analysis capabilities."""
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.scaler = StandardScaler()
        
    def calculate_advanced_metrics(self, team: str) -> Dict:
        """Calculate advanced team metrics."""
        team_data = self.df[self.df['team'] == team]
        
        # Points differential and volatility
        points_diff = team_data['points_scored'] - team_data['points_allowed']
        
        metrics = {
            'point_diff_mean': points_diff.mean(),
            'point_diff_std': points_diff.std(),
            'scoring_efficiency': team_data['points_scored'].sum() / team_data['total_yards'].sum(),
            'turnover_ratio': team_data['turnovers'].mean(),
            'win_probability': (team_data['result'] == 'W').mean(),
            'consistency_score': 1 - (points_diff.std() / abs(points_diff.mean())) if abs(points_diff.mean()) > 0 else 0
        }
        
        # Calculate rolling averages
        rolling_stats = team_data['points_scored'].rolling(window=3).agg(['mean', 'std']).dropna()
        metrics['recent_form'] = rolling_stats['mean'].iloc[-1] if not rolling_stats.empty else None
        
        return metrics
    
    def performance_trends(self, team: str) -> pd.DataFrame:
        """Analyze team performance trends using time series decomposition."""
        team_data = self.df[self.df['team'] == team].sort_values('date')
        
        # Create time series of points scored
        ts = pd.Series(team_data['points_scored'].values, index=team_data['date'])
        
        # Fit SARIMA model for trend analysis
        model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
        results = model.fit(disp=False)
        
        # Generate forecast
        forecast = results.get_forecast(steps=5)
        
        return pd.DataFrame({
            'forecast': forecast.predicted_mean,
            'lower_ci': forecast.conf_int()['lower'],
            'upper_ci': forecast.conf_int()['upper']
        })

    def calculate_efficiency_metrics(self, team: str) -> Dict:
        """Calculate advanced efficiency metrics."""
        team_data = self.df[self.df['team'] == team]
        
        metrics = {
            'yards_per_point': team_data['total_yards'].sum() / team_data['points_scored'].sum(),
            'points_per_drive': team_data['points_scored'].mean(),
            'turnover_rate': team_data['turnovers'].sum() / len(team_data),
            'win_rate_vs_spread': None  # Placeholder for point spread analysis
        }
        
        return metrics

    def analyze_game_context(self, team: str) -> Dict:
        """Analyze team performance in different game contexts."""
        team_data = self.df[self.df['team'] == team]
        
        # Home vs Away performance
        home_games = team_data[team_data['is_home'] == True]
        away_games = team_data[team_data['is_home'] == False]
        
        context_stats = {
            'home_win_rate': (home_games['result'] == 'W').mean(),
            'away_win_rate': (away_games['result'] == 'W').mean(),
            'home_points_avg': home_games['points_scored'].mean(),
            'away_points_avg': away_games['points_scored'].mean()
        }
        
        return context_stats

    def build_predictive_model(self, team: str) -> Tuple[float, float]:
        """Build and evaluate a predictive model for team performance."""
        team_data = self.df[self.df['team'] == team].copy()
        
        # Create features
        features = ['total_yards', 'turnovers', 'is_home']
        X = team_data[features]
        y = team_data['points_scored']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        model = xgb.XGBRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        return rmse, r2

    def generate_advanced_report(self, team: str) -> str:
        """Generate comprehensive statistical report."""
        basic_metrics = self.calculate_advanced_metrics(team)
        efficiency_metrics = self.calculate_efficiency_metrics(team)
        context_stats = self.analyze_game_context(team)
        rmse, r2 = self.build_predictive_model(team)
        
        report = f"""
Advanced Statistical Analysis for {team}
=====================================

Performance Metrics:
------------------
Point Differential: {basic_metrics['point_diff_mean']:.2f} (±{basic_metrics['point_diff_std']:.2f})
Scoring Efficiency: {basic_metrics['scoring_efficiency']:.3f} points/yard
Consistency Score: {basic_metrics['consistency_score']:.3f}
Recent Form: {basic_metrics['recent_form']:.2f} points/game

Efficiency Metrics:
-----------------
Yards per Point: {efficiency_metrics['yards_per_point']:.2f}
Points per Drive: {efficiency_metrics['points_per_drive']:.2f}
Turnover Rate: {efficiency_metrics['turnover_rate']:.3f}

Context Analysis:
---------------
Home Win Rate: {context_stats['home_win_rate']:.3f}
Away Win Rate: {context_stats['away_win_rate']:.3f}
Home Points Avg: {context_stats['home_points_avg']:.2f}
Away Points Avg: {context_stats['away_points_avg']:.2f}

Predictive Model Performance:
--------------------------
RMSE: {rmse:.2f}
R² Score: {r2:.3f}
"""
        return report

    def plot_advanced_visualizations(self, team: str) -> None:
        """Generate advanced statistical visualizations."""
        team_data = self.df[self.df['team'] == team].copy()
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Performance Distribution
        sns.kdeplot(data=team_data['points_scored'], ax=ax1, label='Points Scored')
        sns.kdeplot(data=team_data['points_allowed'], ax=ax1, label='Points Allowed')
        ax1.set_title('Score Distribution')
        ax1.legend()
        
        # 2. Rolling Performance
        window = 3
        rolling_mean = team_data['points_scored'].rolling(window=window).mean()
        rolling_std = team_data['points_scored'].rolling(window=window).std()
        ax2.plot(team_data['date'], rolling_mean, label=f'{window}-Game Average')
        ax2.fill_between(team_data['date'],
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.2)
        ax2.set_title('Rolling Performance')
        ax2.legend()
        
        # 3. Efficiency Metrics
        efficiency = team_data['points_scored'] / team_data['total_yards']
        ax3.scatter(team_data['total_yards'], team_data['points_scored'])
        ax3.set_title('Yards vs Points Correlation')
        ax3.set_xlabel('Total Yards')
        ax3.set_ylabel('Points Scored')
        
        # 4. Home vs Away Performance
        home_away = pd.DataFrame({
            'Location': ['Home', 'Away'],
            'Win Rate': [
                team_data[team_data['is_home']]['result'].value_counts(normalize=True).get('W', 0),
                team_data[~team_data['is_home']]['result'].value_counts(normalize=True).get('W', 0)
            ]
        })
        sns.barplot(data=home_away, x='Location', y='Win Rate', ax=ax4)
        ax4.set_title('Home vs Away Win Rate')
        
        plt.tight_layout()
        plt.savefig(f'advanced_stats_{team}.png')
        plt.close()

if __name__ == "__main__":
    analyzer = AdvancedNFLStats("C:\\Users\\Student-\\Desktop\\nfl_games.csv")
    
    # Get list of teams
    teams = sorted(analyzer.df['team'].unique())
    
    print("Advanced NFL Statistical Analysis")
    print("================================")
    
    # Print available teams
    print("\nAvailable teams:")
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team}")
    
    try:
        team_idx = int(input("\nEnter team number to analyze: ")) - 1
        selected_team = teams[team_idx]
        
        # Generate and display advanced report
        report = analyzer.generate_advanced_report(selected_team)
        print(report)
        
        # Generate advanced visualizations
        analyzer.plot_advanced_visualizations(selected_team)
        print(f"\nAdvanced statistical plots have been saved as 'advanced_stats_{selected_team}.png'")
        
    except (ValueError, IndexError):
        print("Invalid selection. Please run again and select a valid team number.")