import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

class NFLAnalyzer:
    def __init__(self, data_path):
        """Initialize the NFL Analyzer with the path to the games data."""
        self.df = pd.read_csv(data_path)
        self.process_data()

    def process_data(self):
        """Clean and prepare the data for analysis."""
        # Convert date columns to datetime
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Calculate basic team statistics
        self.calculate_team_stats()

    def calculate_team_stats(self):
        """Calculate comprehensive team statistics."""
        # Group by team and calculate averages
        self.team_stats = self.df.groupby('team').agg({
            'points_scored': ['mean', 'std', 'min', 'max'],
            'points_allowed': ['mean', 'std', 'min', 'max'],
            'total_yards': ['mean', 'std'],
            'turnovers': ['mean', 'sum']
        }).round(2)

    def analyze_team_performance(self, team_name):
        """Analyze performance metrics for a specific team."""
        team_data = self.df[self.df['team'] == team_name]
        
        # Performance over time
        performance_metrics = {
            'Points Scored': team_data['points_scored'].mean(),
            'Points Allowed': team_data['points_allowed'].mean(),
            'Win Rate': (team_data['result'] == 'W').mean(),
            'Total Games': len(team_data)
        }
        
        return performance_metrics

    def plot_team_trends(self, team_name):
        """Create visualizations for team performance trends."""
        team_data = self.df[self.df['team'] == team_name].sort_values('date')
        
        # Create trend plot
        plt.figure(figsize=(12, 6))
        plt.plot(team_data['date'], team_data['points_scored'], label='Points Scored')
        plt.plot(team_data['date'], team_data['points_allowed'], label='Points Allowed')
        plt.title(f'{team_name} Performance Trends')
        plt.xlabel('Date')
        plt.ylabel('Points')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(f'team_trends_{team_name}.png')
        plt.close()

    def calculate_strength_of_schedule(self):
        """Calculate strength of schedule for each team."""
        # Calculate average opponent win percentage
        team_records = self.df.groupby('team')['result'].apply(
            lambda x: (x == 'W').mean()
        )
        
        strength_of_schedule = {}
        for team in self.df['team'].unique():
            opponent_wins = []
            team_games = self.df[self.df['team'] == team]
            
            for _, game in team_games.iterrows():
                opponent = game['opponent']
                if opponent in team_records:
                    opponent_wins.append(team_records[opponent])
            
            strength_of_schedule[team] = np.mean(opponent_wins) if opponent_wins else 0
            
        return pd.Series(strength_of_schedule).sort_values(ascending=False)

    def generate_performance_report(self, team_name):
        """Generate a comprehensive performance report for a team."""
        performance = self.analyze_team_performance(team_name)
        sos = self.calculate_strength_of_schedule()
        
        report = f"""
Performance Report for {team_name}
================================
Win Rate: {performance['Win Rate']:.3f}
Average Points Scored: {performance['Points Scored']:.2f}
Average Points Allowed: {performance['Points Allowed']:.2f}
Total Games Analyzed: {performance['Total Games']}
Strength of Schedule Rank: {sos.index.get_loc(team_name) + 1} of {len(sos)}
        """
        
        return report

if __name__ == "__main__":
    # Initialize the analyzer with your data
    analyzer = NFLAnalyzer("C:\\Users\\Student-\\Desktop\\nfl_games.csv")
    
    # Example usage
    print("NFL Team Analysis Tool")
    print("=====================")
    
    # Get list of teams
    teams = sorted(analyzer.df['team'].unique())
    
    # Print available teams
    print("\nAvailable teams:")
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team}")
    
    # Get user input for team selection
    try:
        team_idx = int(input("\nEnter team number to analyze: ")) - 1
        selected_team = teams[team_idx]
        
        # Generate and display report
        report = analyzer.generate_performance_report(selected_team)
        print(report)
        
        # Generate visualization
        analyzer.plot_team_trends(selected_team)
        print(f"\nTeam performance plot has been saved as 'team_trends_{selected_team}.png'")
        
    except (ValueError, IndexError):
        print("Invalid selection. Please run again and select a valid team number.")