import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
	import kagglehub
except ImportError:
	kagglehub = None


def find_train_file(dataset_path: str) -> Path:
	root = Path(dataset_path)
	if root.is_file():
		return root
	if root.name == "train.csv" and root.exists():
		return root
	if (root / "train.csv").exists():
		return root / "train.csv"
	candidates = list(root.rglob("train.csv"))
	if not candidates:
		raise FileNotFoundError(f"Could not find train.csv in {root}")
	return candidates[0]


def resolve_train_file(data_path: str | None) -> Path:
	if data_path:
		return find_train_file(data_path)

	env_path = os.getenv("HOUSE_PRICES_TRAIN")
	if env_path:
		return find_train_file(env_path)

	local_candidates = [Path("train.csv"), Path("data") / "train.csv"]
	for candidate in local_candidates:
		if candidate.exists():
			return candidate

	if kagglehub is not None:
		try:
			dataset_path = kagglehub.competition_download("house-prices-advanced-regression-techniques")
			return find_train_file(dataset_path)
		except Exception:
			pass

	raise FileNotFoundError(
		"Could not find train.csv locally and kagglehub is unavailable. "
		"Place the dataset CSV in the project folder, pass --data PATH, or set HOUSE_PRICES_TRAIN."
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a linear regression model for house prices.")
	parser.add_argument("--data", help="Path to train.csv or its containing folder", default=None)
	args, _ = parser.parse_known_args()

	train_file = resolve_train_file(args.data)

	data = pd.read_csv(train_file)

	feature_columns = ["GrLivArea", "BedroomAbvGr", "FullBath", "HalfBath"]
	target_column = "SalePrice"

	missing_columns = [column for column in feature_columns + [target_column] if column not in data.columns]
	if missing_columns:
		raise ValueError(f"Missing expected columns: {missing_columns}")

	features = data[feature_columns].copy()
	features["Bathrooms"] = features["FullBath"] + 0.5 * features["HalfBath"]
	features = features[["GrLivArea", "BedroomAbvGr", "Bathrooms"]]
	target = data[target_column]

	x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

	model = LinearRegression()
	model.fit(x_train, y_train)

	predictions = model.predict(x_test)

	mse = mean_squared_error(y_test, predictions)
	rmse = mse ** 0.5
	mae = mean_absolute_error(y_test, predictions)
	r2 = r2_score(y_test, predictions)

	print(f"Training rows: {len(x_train)}")
	print(f"Testing rows: {len(x_test)}")
	print(f"RMSE: {rmse:.2f}")
	print(f"MAE: {mae:.2f}")
	print(f"R^2: {r2:.4f}")
	print("Coefficients:")
	for name, coefficient in zip(features.columns, model.coef_):
		print(f"  {name}: {coefficient:.4f}")
	print(f"Intercept: {model.intercept_:.2f}")


if __name__ == "__main__":
	main()

