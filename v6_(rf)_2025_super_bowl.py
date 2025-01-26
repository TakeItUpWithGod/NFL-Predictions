import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("nfl_games.csv")


df_v1 = df.drop(columns=['game_id', 'week', 'gameday', 'weekday', 'gametime', 'location', 'total',
                         'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn',
                         'ftn', 'away_rest', 'home_rest', 'away_moneyline', 'home_moneyline', 'spread_line',
                         'away_spread_odds', 'home_spread_odds', 'total_line', 'under_odds', 'over_odds',
                         'temp', 'wind', 'away_qb_name', 'home_qb_name', 'away_coach', 'home_coach', 'referee',
                         'stadium'])
df_v1 = df_v1.dropna()


categorical_columns = ['season', 'game_type', 'away_team', 'home_team', 'roof', 'surface', 'div_game']
numerical_columns = ['away_score', 'home_score']
target = 'result'


oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
label_encoder = LabelEncoder()

X_categorical = oh_encoder.fit_transform(df_v1[categorical_columns])

X = np.hstack([X_categorical, df_v1[numerical_columns]])
y = label_encoder.fit_transform(df_v1[target])  # Encode target as integers

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


def predict_winner_rf(away_team, home_team, season, game_type, roof, surface, div_game, model, oh_encoder):
    sample_data = {
        'season': [season],
        'game_type': [game_type],
        'away_team': [away_team],
        'home_team': [home_team],
        'roof': [roof],
        'surface': [surface],
        'div_game': [div_game],
        'away_score': [0],
        'home_score': [0]
    }
    input_df = pd.DataFrame(sample_data)

    encoded_input = oh_encoder.transform(input_df[categorical_columns])

    final_input = np.hstack([encoded_input, input_df[numerical_columns].values])

    prediction = model.predict(final_input)
    predicted_class = prediction[0]
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    if predicted_label > 0:
        return f"{home_team} wins"
    elif predicted_label < 0:
        return f"{away_team} wins"
    else:
        return "Tie game expected"


result = predict_winner_rf(
    away_team="WAS",
    home_team="PHI",
    season=2025,
    game_type="REG",
    roof="Open",
    surface="Grass",
    div_game=1,
    model=rf_model,
    oh_encoder=oh_encoder
)
print(result)
