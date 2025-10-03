import streamlit as st
import pandas as pd
from nfl_predictive_analytics import NFLPredictiveAnalytics
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NFL Analytics Dashboard", layout="wide")

@st.cache_data
def load_analyzer():
    return NFLPredictiveAnalytics("nfl_games.csv")

def main():
    st.title("NFL Advanced Analytics Dashboard")
    
    analyzer = load_analyzer()
    teams = sorted(analyzer.df['team'].unique())
    
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type",
        ["Single Team Analysis", "Team Comparison"]
    )
    
    if analysis_type == "Single Team Analysis":
        selected_team = st.sidebar.selectbox("Select Team", teams)
        
        # Generate comprehensive report
        report = analyzer.generate_comprehensive_report(selected_team)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Team Statistics")
            adj_stats = analyzer.calculate_opponent_adjusted_stats(selected_team)
            
            # Create metrics
            st.metric("Adjusted Points Scored", f"{adj_stats['adj_points_scored']:.2f}")
            st.metric("Adjusted Points Allowed", f"{adj_stats['adj_points_allowed']:.2f}")
            st.metric("Opponent Strength Index", f"{adj_stats['opponent_strength_index']:.3f}")
        
        with col2:
            st.subheader("Playoff vs Regular Season")
            playoff_stats = analyzer.analyze_playoff_performance(selected_team)
            
            # Create comparative metrics
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric("Playoff Win Rate", f"{playoff_stats['playoff']['win_rate']:.3f}")
                st.metric("Playoff Points Scored", f"{playoff_stats['playoff']['points_scored']:.2f}")
            with col2b:
                st.metric("Regular Season Win Rate", f"{playoff_stats['regular']['win_rate']:.3f}")
                st.metric("Regular Season Points Scored", f"{playoff_stats['regular']['points_scored']:.2f}")
        
        # Model Performance
        st.subheader("Predictive Model Insights")
        predictor_stats = analyzer.build_ensemble_predictor(selected_team)
        
        # Feature importance plot
        feature_importance = predictor_stats['feature_importance']
        fig = px.bar(
            x=list(feature_importance.keys()),
            y=list(feature_importance.values()),
            title="Feature Importance in Prediction Model"
        )
        st.plotly_chart(fig)
        
    else:  # Team Comparison
        col1, col2 = st.columns(2)
        with col1:
            team1 = st.selectbox("Select First Team", teams, key="team1")
        with col2:
            team2 = st.selectbox("Select Second Team", teams, key="team2")
        
        if team1 and team2:
            comparison = analyzer.compare_teams(team1, team2)
            
            # Create comparison visualizations
            st.subheader("Team Comparison")
            
            # Scoring Comparison
            fig1 = go.Figure(data=[
                go.Bar(name='Points Scored', 
                      x=[team1, team2],
                      y=[comparison['points_scored'][team1]['mean'],
                         comparison['points_scored'][team2]['mean']]),
                go.Bar(name='Points Allowed',
                      x=[team1, team2],
                      y=[comparison['points_allowed'][team1]['mean'],
                         comparison['points_allowed'][team2]['mean']])
            ])
            fig1.update_layout(title="Scoring Comparison")
            st.plotly_chart(fig1)
            
            # Head to Head Record
            st.subheader("Head-to-Head Record")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{team1} Win Rate", f"{comparison['head_to_head'][team1]:.3f}")
            with col2:
                st.metric(f"{team2} Win Rate", f"{comparison['head_to_head'][team2]:.3f}")

if __name__ == "__main__":
    main()
