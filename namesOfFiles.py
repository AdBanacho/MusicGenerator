
def name_missing_percent(train_data_set_up):
    return f"e{train_data_set_up.epoch}/m{train_data_set_up.missing_scope}/l{train_data_set_up.look_back}/" \
           + str(train_data_set_up.missing_percent) \
           + "0%_" + str(train_data_set_up.new_missing_percent)  \
           + "0%_" + str(train_data_set_up.look_back) + "_look_back"\
           + "_" + str(train_data_set_up.epoch) + "_epoch" \
           + "_" + str(train_data_set_up.missing_scope) + "_missing_scope"


def name_missing_percent_model(train_data_set_up):
    return str(train_data_set_up.missing_percent) \
           + "0%_" + str(train_data_set_up.new_missing_percent)  \
           + "0%_" + str(train_data_set_up.look_back) + "_look_back"\
           + "_" + str(train_data_set_up.epoch) + "_epoch" \
           + "_" + str(train_data_set_up.missing_scope) + "_missing_scope"


def name_length_of_signal(train_data_set_up):
    return str(train_data_set_up.length_of_signal) \
           + "_" + str(train_data_set_up.look_back) + "_look_back"


def name_of_folder(train_data_set_up):
    return f"e{train_data_set_up.epoch}/m{train_data_set_up.missing_scope}/l{train_data_set_up.look_back}"
