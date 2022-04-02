import os
import itertools


def create_folders(epochs, missing_scope, look_back):
    for epoch, ms, lb in itertools.product(epochs, missing_scope, look_back):
        os.makedirs(f"models/e{epoch}/m{ms}/l{lb}", exist_ok=True)
        os.makedirs(f"npy/e{epoch}/m{ms}/l{lb}", exist_ok=True)
        os.makedirs(f"predict_sounds/e{epoch}/m{ms}/l{lb}", exist_ok=True)
        os.makedirs(f"scores/e{epoch}/m{ms}/l{lb}", exist_ok=True)
