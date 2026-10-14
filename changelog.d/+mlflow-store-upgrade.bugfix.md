The app's own MLflow tracking store now migrates to the installed MLflow's schema on
start, after the file is copied to a `.bak` beside it, so upgrading to a release with a
newer MLflow no longer needs a manual `mlflow db upgrade` before models work again. A
failed migration is logged with the backup's path and the app starts anyway. A tracking
server's database named by `MLFLOW_TRACKING_URI` is left alone.
