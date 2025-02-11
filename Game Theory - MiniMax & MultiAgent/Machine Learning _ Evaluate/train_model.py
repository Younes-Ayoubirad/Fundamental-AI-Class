import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    def machine_model():
        data = pd.read_csv("training_data.csv")

        print(data.info())

        features = data.iloc[1:, 0:-1]
        scores = data.iloc[1:, -1]

        features.fillna(0, inplace=True)

        MMS = MinMaxScaler()
        features = MMS.fit_transform(features)

        model = RandomForestRegressor()
        model.fit(features, scores)

        with open('model.pkl', 'wb') as file:
            pickle.dump(model, file)

    def neural_network_model():
        data = pd.read_csv("training_data.csv")

        features = data[["distance_hen", "distance_egg", "distance_shooter", "num_eggs", "num_pigs"]]
        scores = data["score"]

        X_train, X_test, y_train, y_test = train_test_split(features, scores, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        with open("scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(5,)),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

        model.save("heuristic_model_nn.h5")

        print("Trained model saved")