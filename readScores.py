import pandas as pd

def read_scores():
    annMS, annMA, annR2 = [], [], []
    rnnRMS, rnnRMA, rnnRR2 = [], [], []
    rnnLMS, rnnLMA, rnnLR2 = [], [], []

    for i in range(1, 10):
        with open('scores/ann' + str(i) + '.txt') as f:
            ann = f.readlines()
        with open('scores/rnnR' + str(i) + '.txt') as f:
            rnnR = f.readlines()
        with open('scores/rnnL' + str(i) + '.txt') as f:
            rnnL = f.readlines()

        annMS.append(float(ann[2].split(":")[1].split("\n")[0][1:]))
        annMA.append(float(ann[3].split(":")[1].split("\n")[0][1:]))
        annR2.append(float(ann[4].split(":")[1].split("\n")[0][1:]))

        rnnRMS.append(float(rnnR[2].split(":")[1].split("\n")[0][1:]))
        rnnRMA.append(float(rnnR[3].split(":")[1].split("\n")[0][1:]))
        rnnRR2.append(float(rnnR[4].split(":")[1].split("\n")[0][1:]))

        rnnLMS.append(float(rnnL[2].split(":")[1].split("\n")[0][1:]))
        rnnLMA.append(float(rnnL[3].split(":")[1].split("\n")[0][1:]))
        rnnLR2.append(float(rnnL[4].split(":")[1].split("\n")[0][1:]))

    return [rnnRR2, rnnLR2, annR2, rnnRMS, rnnLMS, annMS, rnnRMA,  rnnLMA, annMA]


if __name__ == '__main__':
    index = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%']
    columnsR2 = ['rnnR-R2', 'rnnL-R2', 'ann-R2']
    columnsMS = ['rnnR-MS', 'rnnL-MS', 'ann-MS']
    columnsMA = ['rnnR-MA', 'rnnL-MA', 'ann-MA']
    R2 = pd.DataFrame({'brakujący %': index})
    MS = pd.DataFrame({'brakujący %': index})
    MA = pd.DataFrame({'brakujący %': index})
    for i in range(3):
        R2[columnsR2[i]] = read_scores()[i]
        MS[columnsMS[i]] = read_scores()[i+3]
        MA[columnsMA[i]] = read_scores()[i+6]
    R2 = R2.set_index('brakujący %')
    MS = MS.set_index('brakujący %')
    MA = MA.set_index('brakujący %')
    print("---------------X R2 X----------------")
    print(R2)
    print("---------------X MS X----------------")
    print(MS)
    print("---------------X MA X----------------")
    print(MA)
