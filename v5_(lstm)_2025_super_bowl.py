import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from keras.utils import to_categorical


df = pd.read_csv("nfl_games.csv")

# Will come back to these in future as these add complications
df_v1 = df.drop(columns=['game_id', 'week', 'gameday', 'weekday', 'gametime', 'location', 'total',
                         'overtime', 'old_game_id', 'gsis', 'nfl_detail_id', 'pfr', 'pff', 'espn',
                         'ftn', 'away_rest', 'home_rest', 'away_moneyline', 'home_moneyline', 'spread_line',
                         'away_spread_odds', 'home_spread_odds', 'total_line', 'under_odds', 'over_odds',
                         'temp', 'wind', 'away_qb_name', 'home_qb_name', 'away_coach', 'home_coach', 'referee',
                         'stadium'])
df_v1 = df_v1.dropna()


# Games 2020 onward should have double weight
df_v1['recency_weight'] = df_v1['season'].apply(lambda x: 1 if x < 2020 else 2)

target = 'result' # home score - away score
X = df_v1.drop(columns=[target])
y = df_v1[target]

categorical_columns = ['season', 'game_type', 'away_team', 'home_team', 'roof', 'surface', 'div_game']
numerical_columns = ['away_score', 'home_score']

# Trying both encoders
oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # For most categorical columns
label_encoder = LabelEncoder()

scaler = StandardScaler()

X_categorical = oh_encoder.fit_transform(X[categorical_columns])
X_numerical = scaler.fit_transform(X[numerical_columns])

X_prepared = np.hstack([X_numerical, X_categorical])

# Encoding target
y_encoded = LabelEncoder().fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Reshape data for LSTM
X_reshaped = X_prepared.reshape(X_prepared.shape[0], 1, X_prepared.shape[1])

# might try different splits in future- train on even years, test on odd
X_train, X_test, y_train, y_test, train_weights, test_weights = train_test_split(
    X_reshaped, y_categorical, df_v1['recency_weight'], test_size=0.2, random_state=42)


rnn = Sequential([
    LSTM(128, activation='relu', input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])),
    Dropout(0.2),
    Dense(64, activation='sigmoid'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_categorical.shape[1], activation='softmax')
])

rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall'])

# training
history = rnn.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test),
                  sample_weight=train_weights)

rnn_loss, rnn_accuracy, rnn_precision, rnn_recall = rnn.evaluate(X_test, y_test)
print(f"Test Accuracy: {rnn_accuracy:.2f}")
print(f"Test Loss: {rnn_loss:.2f}")
print(f"Test Precision: {rnn_precision:.2f}")
print(f"Test Recall: {rnn_recall:.2f}")


def predict_winner_rnn(away_team, home_team, roof, surface, div_game, trained_model, oh_encoder, scaler):

    # placeholder input with the required structure..?
    sample_data = {
        "season": 2025,
        "game_type": "REG",  # Regular season as default
        "away_team": away_team,
        "home_team": home_team,
        "roof": roof,
        "surface": surface,
        "away_score": 0,  # Placeholder
        "home_score": 0,  # Placeholder
        "div_game": div_game
    }

    # Convert input into a DataFrame
    input_df = pd.DataFrame([sample_data])

    categorical_input = input_df[categorical_columns]
    encoded_categorical = oh_encoder.transform(categorical_input)
    numerical_input = input_df[['away_score', 'home_score']]
    scaled_numerical = scaler.transform(numerical_input)

    final_input = np.hstack([scaled_numerical, encoded_categorical])

    # Reshape input to match RNN
    final_input = final_input.reshape(1, 1, final_input.shape[1])

    # Predict using trained model
    prediction = trained_model.predict(final_input)
    predicted_class = np.argmax(prediction, axis=1)

    label_encoder.fit(y)
    result_value = label_encoder.classes_[predicted_class]

    if result_value > 0:
        return f"{home_team} wins by {abs(result_value)}"
    elif result_value < 0:
        return f"{away_team} wins by {abs(result_value)}"
    else:
        return "Tie game expected"


result = predict_winner_rnn(
    away_team="BUF",
    home_team="KC",
    roof="Open",
    surface="Grass",
    div_game=1,
    trained_model=rnn,
    oh_encoder=oh_encoder,
    scaler=scaler
)
print(result)
