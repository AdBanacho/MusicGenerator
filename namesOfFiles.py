
def name_missing_percent(train_data_set_up):
    return str(train_data_set_up.missing_percent) \
           + "0%_" + str(train_data_set_up.look_back) + "_look_back"\
           + "_" + str(train_data_set_up.epoch) + "_epoch"


def name_length_of_signal(train_data_set_up):
    return str(train_data_set_up.length_of_signal) \
           + "_" + str(train_data_set_up.look_back) + "_look_back"\
           + "_" + str(train_data_set_up.epoch) + "_epoch"
