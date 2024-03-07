from pathlib import Path
import os
from datetime import datetime

MODULE_DIR = Path(__file__).parent
ROOT_DIR = MODULE_DIR.parent

CURRENT_DATE = '2024-03-01' #datetime.utcnow().replace(microsecond=0, second=0, minute=0).strftime('%Y-%m-%d') #'2023-05-29' #
RUN_ENV = "dev"
PIPELINE_TYPE = "train"
CSV_BUFFER = True
SAVE_TO_CSVS = True if RUN_ENV == "dev" else False

# Mounting
MODEL_DIR = "/mnt/efs-ds-{run_env}/models/mortality/cam/"
if RUN_ENV == "dev":
    MODEL_DIR = os.path.join("//wsl.localhost/Ubuntu-20.04/", MODEL_DIR)

# Write, read DB
if RUN_ENV == "dev":
    READ_PARAMS = dict(
        host=os.environ["REPLICA_DB_HOST"],
        dbname=os.environ["REPLICA_DB_NAME"],
        user=os.environ["REPLICA_DB_USERNAME"],
        password=os.environ["REPLICA_DB_PASSWORD"],
        port=os.environ["REPLICA_DB_PORT"],
    )
else:
    READ_PARAMS = dict(
        host=os.environ["DB_HOST"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USERNAME"],
        password=os.environ["DB_PASSWORD"],
        port=os.environ["DB_PORT"],
    )
WRITE_PARAMS = dict(
    host=os.environ["DB_HOST"],
    dbname=os.environ["DB_NAME"],
    user=os.environ["DB_USERNAME"],
    password=os.environ["DB_PASSWORD"],
    port=os.environ["DB_PORT"],
)

