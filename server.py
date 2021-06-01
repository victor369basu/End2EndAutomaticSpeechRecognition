import subprocess
import threading
import time

import mlflow


from config import get_parameters

config = get_parameters()

def run_shell_command(command, wait):
    while True:
      time.sleep(wait)
      subprocess.check_output(command, shell=True)

t = threading.Thread(target=run_shell_command, args=["mlflow server \
                                                    --backend-store-uri {storage_uri} \
                                                    --default-artifact-root {artifact_root} \
                                                    --host {host}".format(
                                                    	storage_uri=config.backend_store_uri,
                                                        artifact_root=config.artifact_root,
                                                        host=config.host
                                                    	)
                                                    , 3])

t.start()

time.sleep(10)
mlflow.create_experiment(config.experiment_name,config.artifact_root)
mlflow.set_experiment(config.experiment_name)
mlflow.set_tracking_uri("http://"+config.host+":"+config.port)