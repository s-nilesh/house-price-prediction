import mlflow
from src.utils import utils
from src.data.data_loader import load_data
from src.algorithms.linear_regression import LinearRegression
from src.eval.metrics import Metrics
import pandas as pd
from src.pipelines.base_pipeline import BasePipeline  # Inherit from BasePipeline

class TrainPipeline(BasePipeline):
    def __init__(self, config):
        self.conf = config
        self.lr_rate = config.get("model").get("learning_rate")

    def load_and_preprocess_data(self):
        """Load and preprocess data."""
        data = load_data(self.conf.get("data").get("train_file"))
        print(f"Loaded data with {len(data)} rows")

        data = data.dropna()
        data = data.apply(pd.to_numeric, errors='coerce')

        X = data.drop("y", axis=1).values
        y = data['y'].values

        return X, y

    def initialize_model(self):
        """Initialize the Linear Regression model."""
        return LinearRegression()

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, model):
        """Train and evaluate the model."""
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            # Train the model
            model.fit(X_train, y_train)

            # Log model with MLflow
            mlflow.sklearn.log_model(model, "lr_model", input_example=X_train[:2], registered_model_name="HousePricePred")

            # Make predictions
            y_pred = model.predict(X_test)

            # Log metrics
            mlflow.log_param("learning_rate", self.lr_rate)
            mlflow.log_metric("accuracy", Metrics.accuracy(y_test, y_pred))
            mlflow.log_metric("cost", model.cost_function(y_pred, y_test))

        return run_id, model

    def execute(self):
        """Run the training pipeline."""
        # Load and preprocess data
        X, y = self.load_and_preprocess_data()
        X_train, X_test, y_train, y_test = utils.train_test_split(X, y)

        # Setup MLflow experiment
        experiment_name = "exp_1"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            mlflow.create_experiment(name=experiment_name, artifact_location="/home/nilesh/projects/")
        mlflow.set_experiment("exp_1")

        
        lr = self.initialize_model()
        _, model = self.train_and_evaluate_model(X_train, y_train, X_test, y_test, lr)
        model.save_model("/home/nilesh/projects/house_price_prediction/models/lr.pkl")   # add to config