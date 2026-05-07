import logging
import os
import sys
import gc

class MultilineFormatter(logging.Formatter):
    def format(self, record):
        # Get the original formatted message
        original_message = super().format(record)
        
        # Split the message into lines
        lines = original_message.split('\n')
        
        if len(lines) <= 1:
            return original_message
        
        # For multi-line messages, format each line with the prefix
        formatted_lines = []
        for i, line in enumerate(lines):
            if i == 0:
                # First line already has the full format
                formatted_lines.append(line)
            else:
                # Subsequent lines need to be formatted with the same prefix
                # Create a new record for consistent formatting
                new_record = logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg=line,
                    args=(),
                    exc_info=None
                )
                new_record.created = record.created
                formatted_line = super().format(new_record)
                formatted_lines.append(formatted_line)
        
        return '\n'.join(formatted_lines)

class CondaFilter(logging.Filter):
    def filter(self, record):
        if 'optuna' in record.name:
            return True
        #return True
        return 'envs' not in record.pathname

conda_filter = CondaFilter()

os.makedirs("log", exist_ok=True)

# Get job identification from environment variables, supporting both generic and SLURM-specific ones
job_id = os.getenv('JOB_ID') or os.getenv('SLURM_JOB_ID')
job_name = os.getenv('JOB_NAME') or os.getenv('SLURM_JOB_NAME', 'unnamed_job')

if job_id:
    log_filename = f"{job_name}_{job_id}.out"
else:
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{current_time}.log"

file_handler = logging.FileHandler(f"log/{log_filename}")
stream_handler = logging.StreamHandler(sys.stdout)

file_handler.addFilter(conda_filter)
stream_handler.addFilter(conda_filter)

formatter = MultilineFormatter(
    fmt='(%(asctime)s %(filename)s@%(lineno)d %(levelname)s) %(message)s',
    datefmt='%H:%M:%S'
)

file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, stream_handler]
)

optuna_logger = logging.getLogger("optuna")
optuna_logger.handlers = []
optuna_logger.addHandler(file_handler)
optuna_logger.addHandler(stream_handler)
optuna_logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception

def print_memory_summary(message, gpu=0, verbose=False):
    import torch
    free_memory, total_memory = torch.cuda.mem_get_info(gpu)
    used_memory = total_memory - free_memory
    logger.debug(f"{message}: {(used_memory / (1024**2)):.2f} / {(total_memory / (1024**2)):.2f} MB")
    if verbose:
        logger.debug(torch.cuda.memory_summary(device=gpu, abbreviated=False))
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    logger.debug(f"Type: {type(obj)} | Size: {obj.size()} | Usage: {obj.element_size() * obj.nelement() / 1024**2:.2f} MB")
            except:
                pass
