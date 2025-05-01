from pydantic import BaseModel


class VAEHistory(BaseModel):
    train_loss: list = []
    val_loss: list = []
    test_loss: list = []
    z: list = []
    z_mean: list = []
    z_log_var: list = []
    label: list = []
