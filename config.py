import argparse

basedir = './data/LibriSpeech/test-clean'

def str2bool(v):
    return v.lower() in ('true')

def get_parameters():

    parser = argparse.ArgumentParser()
    # Base Directory
    parser.add_argument('-d', '--dataset', dest='dataset', default=basedir, 
                        help="Path to the dataset directory.")
    # mlflow server
    parser.add_argument('-ar', '--artifact_root', default='./mlruns', 
                        help="Path to the artifact-root directory.")   
    parser.add_argument('-h0', '--host', default="127.0.0.1", type=str,
                        help="Server Host")
    parser.add_argument('-p', '--port', default="5000", type=str,
                        help="Server Port")
    parser.add_argument('-bsu', '--backend_store_uri', default="sqlite:///mlflow.db", type=str,
                        help="Backend URI.")
    # experiment 
    parser.add_argument('-en', '--experiment_name', default="SpeechRecognition", type=str,
                        help="Backend URI.")

    parser.add_argument('-rm', '--registered_model', default="SpeechRecognitionModel", type=str,
                        help="Registered model name.")
    parser.add_argument('-stg', '--model_stage', default="None", 
    	                type=str,choices=["None", "Staging", "Production", "Archieve"],
                        help="Stage of Registered Model.")
    parser.add_argument('-ver', '--model_version', default=1,type=int,
                        help="Version of the Registered Model.")

    parser.add_argument('-rsm', '--registered_scripted_model', default="SpeechRecognitionModel_Scripted", 
    	                type=str,
                        help="Registered model name.")
    parser.add_argument('-stg_s', '--scripted_model_stage', default="None", 
    	                type=str,choices=["None", "Staging", "Production", "Archieve"],
                        help="Stage of Registered Scripted Model.")
    parser.add_argument('-ver_s', '--scripted_model_version', default=1,type=int,
                        help="Version of the Registered scripted Model.")


    # Audio pre-processing
    parser.add_argument('-sr', '--sampling_rate',type=int, default=16000)
    parser.add_argument('-nmfcc', '--n_mfcc',type=int, default=128)

    # Model Parameters
    parser.add_argument('-cnn', '--n_cnn_layers', type=int, default=2)
    parser.add_argument('-rnn', '--n_rnn_layers', type=int, default=3)
    parser.add_argument('-dim', '--rnn_dim', type=int, default=800)
    parser.add_argument('-nclass', '--n_class', type=int, default=29)
    parser.add_argument('-nfeats', '--n_feats', type=int, default=128)
    parser.add_argument('-st', '--stride', type=int, default=2)
    parser.add_argument('-dout', '--dropout', type=float, default=0.4)

    parser.add_argument('-b', '--batch', dest='batch',type=int, default=32)
    parser.add_argument('-a', '--alpha', type=float, default=0.1)
    parser.add_argument('-b1', '--beta1', type=float, default=0.0)
    parser.add_argument('-b2', '--beta2', type=float, default=0.9)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-9)
    parser.add_argument('-e', '--epoch', type=int, default=800)

    # Misc
    parser.add_argument('-t','--train', type=str2bool, default=True,help="True when train the model, else used for validation.")
    parser.add_argument('-fid','--file_id', default=None, type=str, help="File_id of LibriSpeech dataset sound file for validation through model prediction.")

    return parser.parse_args()