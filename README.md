F1 2026 Race Predictor:

This project is a machine learning pipeline designed to forecast Formula 1 race outcomes for the 2026 season. It leverages historical racing data and technical regulation changes to provide predictive insights into driver and team performance.

Model Functionality:

-The system utilizes three distinct machine learning models to provide a comprehensive outlook on race weekends and season standings:

-Podium Classifier: A classification model that calculates the probability of a driver finishing in the top three positions (P1, P2, or P3).

-Position Regressor: A regression model that predicts the specific numerical finishing position for every driver on the grid.

-Championship Projection: A model that estimates total season points based on predicted race performance and historical consistency.

-To account for the 2026 technical reset, the models incorporate engineered features including Power Unit reliability scores for new manufacturers like Audi and Ford, Active Aero efficiency ratings, and Team Tier classifications based on pre-season testing performance.

Tools and Technologies:

-Python 3.10+ serves as the primary programming language for the entire pipeline.

-FastF1 API is used to interface with official F1 timing, telemetry, and lap data.

-Pandas and NumPy are utilized for data cleaning, transformation, and complex numerical operations.

-Scikit-Learn provides the framework for the Random Forest algorithms and data preprocessing tools.

-Joblib is used for model persistence, allowing trained binaries to be saved and loaded for future predictions.

-Vanilla HTML, CSS, and JavaScript power the local dashboard used to render prediction results via CSV uploads.
