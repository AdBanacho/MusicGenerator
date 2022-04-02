from musicGenerator import MusicGenerator
from folders import create_folders
import itertools


def run(epochs, missing_scope, look_back, missing_percent, length_of_signal, ann, rnnR, rnnL):
    create_folders(epochs, missing_scope, look_back)
    for epoch, ms, lb, mp in itertools.product(epochs, missing_scope, look_back, missing_percent):
        mg = MusicGenerator(epochs=epoch,
                            length_of_signal=length_of_signal,
                            missing_scope=ms,
                            look_back=lb,
                            missing_percent=mp
                            )
        mg.create_data()
        mg.load_data()
        mg.train(ann=ann, rnnR=rnnR, rnnL=rnnL)
        mg.predict(ann=ann, rnnR=rnnR, rnnL=rnnL)

        print("\nXXXXXXXXXXX                            XXXXXXXXXXX")
        print("XXXXXXXXX - MS: " + str(ms) + " MP: " + str(mp) + "% LB: " + str(lb) + " - XXXXXXXXX")
        print("XXXXXXXXXXX                            XXXXXXXXXXX\n")


if __name__ == '__main__':
    run(epochs=[20],                            # best 20
        missing_scope=[0, 1, 10, 100, 1000],    # 0 for random function
        missing_percent=range(10, 100, 10),     # 10% - 90%
        look_back=[10, 100, 1000],              #
        length_of_signal=200000,                # ~9sek 200000
        ann=True,
        rnnR=True,
        rnnL=True
        )
