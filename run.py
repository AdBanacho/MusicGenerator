from musicGenerator import MusicGenerator

if __name__ == '__main__':
    for epoch in [250]:
        for mp in range(40, 90, 30):
            for lb in range(900, 1001, 100):
                mg = MusicGenerator(epochs=epoch,               # best 200 - 250
                                    length_of_signal=200000,  # ~9sek 200000
                                    missing_scope=2000,       #
                                    look_back=lb,           # look_back <= missing_scope
                                    missing_percent=mp        # 10% - 90%
                                    )
                mg.create_data()
                mg.load_data()
                mg.train(ann=True, rnnR=False, rnnL=True)
                mg.predict(ann=True, rnnR=False, rnnL=True)
                print("\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print("XXXXXXXXXXX                            XXXXXXXXXXX")
                print("XXXXXXXXX - Epoka: " + str(epoch) + " MP: " + str(mp) + "% LB: " + str(lb) + " - XXXXXXXXX")
                print("XXXXXXXXXXX                            XXXXXXXXXXX")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n")


