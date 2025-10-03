import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NFLPredictiveAnalytics:
    def __init__(self, data_path: str):
        """Initialize with comprehensive predictive analytics capabilities."""
        self.df = pd.read_csv(data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.scaler = StandardScaler()
        self._prepare_data()
        
    def _prepare_data(self):
        """Prepare and enhance the dataset with advanced features."""
        # Add season information
        self.df['season'] = self.df['date'].dt.year
        
        # Determine playoff games (typically after December)
        self.df['is_playoff'] = (self.df['date'].dt.month >= 12) | (self.df['date'].dt.month == 1)
        
        # Calculate rolling averages for key metrics
        for team in self.df['team'].unique():
            team_mask = self.df['team'] == team
            for metric in ['points_scored', 'points_allowed', 'total_yards', 'turnovers']:
                self.df.loc[team_mask, f'{metric}_rolling_avg'] = (
                    self.df[team_mask][metric].rolling(window=5, min_periods=1).mean()
                )

    def calculate_opponent_adjusted_stats(self, team: str) -> Dict:
        """Calculate opponent-adjusted performance metrics."""
        team_data = self.df[self.df['team'] == team].copy()
        all_teams_avg = self.df.groupby('team').agg({
            'points_scored': 'mean',
            'points_allowed': 'mean',
            'total_yards': 'mean'
        }).mean()

        opponent_strength = {}
        for _, game in team_data.iterrows():
            opponent = game['opponent']
            opp_data = self.df[self.df['team'] == opponent]
            
            # Calculate opponent strength based on their performance vs league average
            opp_strength = (
                opp_data['points_scored'].mean() / all_teams_avg['points_scored'] +
                opp_data['total_yards'].mean() / all_teams_avg['total_yards']
            ) / 2
            
            opponent_strength[game['date']] = opp_strength

        # Adjust team's stats based on opponent strength
        adjusted_stats = {
            'adj_points_scored': team_data['points_scored'].mean() * np.mean(list(opponent_strength.values())),
            'adj_points_allowed': team_data['points_allowed'].mean() * np.mean(list(opponent_strength.values())),
            'opponent_strength_index': np.mean(list(opponent_strength.values()))
        }
        
        return adjusted_stats

    def analyze_playoff_performance(self, team: str) -> Dict:
        """Analyze team's performance in playoff vs regular season games."""
        team_data = self.df[self.df['team'] == team]
        
        playoff_stats = team_data[team_data['is_playoff']].agg({
            'points_scored': ['mean', 'std'],
            'points_allowed': ['mean', 'std'],
            'result': lambda x: (x == 'W').mean()
        })
        
        regular_stats = team_data[~team_data['is_playoff']].agg({
            'points_scored': ['mean', 'std'],
            'points_allowed': ['mean', 'std'],
            'result': lambda x: (x == 'W').mean()
        })
        
        return {
            'playoff': {
                'win_rate': playoff_stats['result'],
                'points_scored': playoff_stats['points_scored']['mean'],
                'points_allowed': playoff_stats['points_allowed']['mean']
            },
            'regular': {
                'win_rate': regular_stats['result'],
                'points_scored': regular_stats['points_scored']['mean'],
                'points_allowed': regular_stats['points_allowed']['mean']
            }
        }

    def build_ensemble_predictor(self, team: str) -> Dict:
        """Build an ensemble of advanced predictive models."""
        team_data = self.df[self.df['team'] == team].copy()
        
        # Feature engineering
        features = [
            'total_yards', 'turnovers', 'is_home',
            'points_scored_rolling_avg', 'points_allowed_rolling_avg',
            'total_yards_rolling_avg', 'turnovers_rolling_avg'
        ]
        
        X = team_data[features]
        y = team_data['points_scored']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train multiple models
        models = {
            'xgboost': xgb.XGBRegressor(random_state=42),
            'lightgbm': lgb.LGBMRegressor(random_state=42),
            'catboost': CatBoostRegressor(random_state=42, verbose=False),
            'rf': RandomForestRegressor(random_state=42)
        }
        
        predictions = {}
        scores = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores[name] = np.sqrt(((pred - y_test) ** 2).mean())
            predictions[name] = pred
        
        # Weighted ensemble prediction
        weights = {name: 1/score for name, score in scores.items()}
        weight_sum = sum(weights.values())
        weights = {name: w/weight_sum for name, w in weights.items()}
        
        ensemble_pred = sum(pred * weights[name] for name, pred in predictions.items())
        ensemble_score = np.sqrt(((ensemble_pred - y_test) ** 2).mean())
        
        return {
            'model_scores': scores,
            'ensemble_score': ensemble_score,
            'feature_importance': self._get_feature_importance(models['xgboost'], features)
        }

    def _get_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Calculate SHAP-based feature importance."""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.df[feature_names])
        
        importance_dict = {}
        for idx, feature in enumerate(feature_names):
            importance_dict[feature] = np.abs(shap_values[:, idx]).mean()
            
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def compare_teams(self, team1: str, team2: str) -> Dict:
        """Perform detailed comparative analysis between two teams."""
        team1_data = self.df[self.df['team'] == team1]
        team2_data = self.df[self.df['team'] == team2]
        
        metrics = ['points_scored', 'points_allowed', 'total_yards', 'turnovers']
        comparison = {}
        
        for metric in metrics:
            comparison[metric] = {
                team1: {
                    'mean': team1_data[metric].mean(),
                    'std': team1_data[metric].std(),
                    'trend': team1_data[f'{metric}_rolling_avg'].iloc[-1]
                },
                team2: {
                    'mean': team2_data[metric].mean(),
                    'std': team2_data[metric].std(),
                    'trend': team2_data[f'{metric}_rolling_avg'].iloc[-1]
                }
            }
            
        # Head-to-head analysis
        h2h = self.df[
            ((self.df['team'] == team1) & (self.df['opponent'] == team2)) |
            ((self.df['team'] == team2) & (self.df['opponent'] == team1))
        ]
        
        comparison['head_to_head'] = {
            team1: (h2h[h2h['team'] == team1]['result'] == 'W').mean(),
            team2: (h2h[h2h['team'] == team2]['result'] == 'W').mean()
        }
        
        return comparison

    def plot_comparative_analysis(self, team1: str, team2: str) -> None:
        """Generate comprehensive comparative visualizations."""
        comparison = self.compare_teams(team1, team2)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Scoring Comparison
        teams = [team1, team2]
        points_scored = [comparison['points_scored'][team]['mean'] for team in teams]
        points_allowed = [comparison['points_allowed'][team]['mean'] for team in teams]
        
        x = np.arange(len(teams))
        width = 0.35
        
        ax1.bar(x - width/2, points_scored, width, label='Points Scored')
        ax1.bar(x + width/2, points_allowed, width, label='Points Allowed')
        ax1.set_xticks(x)
        ax1.set_xticklabels(teams)
        ax1.legend()
        ax1.set_title('Scoring Comparison')
        
        # 2. Efficiency Metrics
        yards_per_point = [
            comparison['total_yards'][team]['mean'] / comparison['points_scored'][team]['mean']
            for team in teams
        ]
        
        ax2.bar(teams, yards_per_point)
        ax2.set_title('Yards per Point')
        
        # 3. Performance Consistency
        consistency = [
            comparison['points_scored'][team]['std'] / comparison['points_scored'][team]['mean']
            for team in teams
        ]
        
        ax3.bar(teams, consistency)
        ax3.set_title('Scoring Consistency (lower is better)')
        
        # 4. Head-to-Head Record
        h2h_record = [comparison['head_to_head'][team] for team in teams]
        ax4.bar(teams, h2h_record)
        ax4.set_title('Head-to-Head Win Rate')
        
        plt.tight_layout()
        plt.savefig(f'comparative_analysis_{team1}_vs_{team2}.png')
        plt.close()

    def generate_comprehensive_report(self, team: str) -> str:
        """Generate a comprehensive analytical report including all advanced metrics."""
        adj_stats = self.calculate_opponent_adjusted_stats(team)
        playoff_stats = self.analyze_playoff_performance(team)
        predictor_stats = self.build_ensemble_predictor(team)
        
        report = f"""
Comprehensive Analysis Report for {team}
=====================================

Opponent-Adjusted Statistics:
--------------------------
Adjusted Points Scored: {adj_stats['adj_points_scored']:.2f}
Adjusted Points Allowed: {adj_stats['adj_points_allowed']:.2f}
Opponent Strength Index: {adj_stats['opponent_strength_index']:.3f}

Playoff vs Regular Season Performance:
----------------------------------
Playoff Win Rate: {playoff_stats['playoff']['win_rate']:.3f}
Playoff Points Scored: {playoff_stats['playoff']['points_scored']:.2f}
Regular Season Win Rate: {playoff_stats['regular']['win_rate']:.3f}
Regular Season Points Scored: {playoff_stats['regular']['points_scored']:.2f}

Predictive Model Performance:
--------------------------
Ensemble Model RMSE: {predictor_stats['ensemble_score']:.2f}

Top Feature Importance:
--------------------"""
        
        for feature, importance in list(predictor_stats['feature_importance'].items())[:5]:
            report += f"\n{feature}: {importance:.3f}"
        
        return report

if __name__ == "__main__":
    analyzer = NFLPredictiveAnalytics("C:\\Users\\Student-\\Desktop\\nfl_games.csv")
    
    print("NFL Advanced Analytics Platform")
    print("==============================")
    
    teams = sorted(analyzer.df['team'].unique())
    
    print("\nAvailable teams:")
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team}")
    
    try:
        print("\nSelect analysis type:")
        print("1. Single Team Analysis")
        print("2. Team Comparison")
        
        analysis_type = int(input("Enter your choice (1 or 2): "))
        
        if analysis_type == 1:
            team_idx = int(input("\nEnter team number to analyze: ")) - 1
            selected_team = teams[team_idx]
            
            report = analyzer.generate_comprehensive_report(selected_team)
            print(report)
            
        elif analysis_type == 2:
            team1_idx = int(input("\nEnter first team number: ")) - 1
            team2_idx = int(input("Enter second team number: ")) - 1
            
            team1 = teams[team1_idx]
            team2 = teams[team2_idx]
            
            analyzer.plot_comparative_analysis(team1, team2)
            print(f"\nComparative analysis plot saved as 'comparative_analysis_{team1}_vs_{team2}.png'")
            
        else:
            print("Invalid choice. Please select 1 or 2.")
            
    except (ValueError, IndexError):
        print("Invalid selection. Please run again and select valid options.")