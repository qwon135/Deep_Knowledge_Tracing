import datetime

# ====================================================
# CFG
# ====================================================
class CFG:
    seed = 42

    use_cuda_if_available = True
    user_wandb = True
    wandb_kwargs = dict(project="boostcamp-dkt")

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    output_dir = "./output"
    time = datetime.datetime.now().strftime('%b-%d-%Y_%H-%M-%S')
    valid_file = 'validation-'+time+'.csv'
    pred_file = "submission-"+time+".csv"

    # build
    embedding_dim = 64  # int
    num_layers = 2 # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {}  # other arguments
    weight = "weight/last_model.pt" # best 안됨...?

    # train
    n_epoch = 50
    learning_rate = 0.05
    weight_basepath = "weight"


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
