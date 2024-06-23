import tensorflow as tf
import pandas as pd
import numpy as np

from utils.df_loader import (
    load_electricity_mixed_df,
    load_eye_movements_mixed_df,
    load_covertype_mixed_df,
    load_albert_df,
    load_road_safety_df,
    load_california_df,
    load_credit_df,
    load_heloc_df,
    load_jannis_df,
    load_Diabetes130US_df,
    load_eye_movements_df,
    load_Higgs_df,
    load_default_of_credit_card_clients_df,
    load_MiniBooNE_df,
    load_bank_marketing_df,
    load_Bioresponse_df,
    load_bank_marketing_df,
    load_MagicTelescope_df,
    load_house_16H_df,
    load_pol_df,
    load_electricity_df,
    load_covertype_df,
    load_adult_df,
    load_compas_df,
    load_german_df,
    load_diabetes_df,
    load_breast_cancer_df,
)
from sklearn.model_selection import train_test_split
from utils.preprocessing import preprocess_df
from utils.models import (
    train_three_models,
    evaluation_test,
    save_three_models,
    load_three_models,
)
import utils.cf_proto as util_cf_proto
import utils.dice as util_dice
import utils.gs as util_gs
import utils.watcher as util_watcher
import utils.print as print_f


from utils.save import save_result_as_csv

numerical_datasets = [ "california", "credit", "heloc",  "jannis", "Diabetes130US", "eye_movements", "Higgs", "default-of-credit-card-clients", "MiniBooNE", "Bioresponse", "bank-marketing", "MagicTelescope", "house_16H", "pol","covertype", "electricity", "diabetes", "breast_cancer"]

mixed_datasets = ["electricity_mixed", "eye_movements_mixed", "covertype_mixed", "albert", "road-safety"]

datasets = [
    #"electricity_mixed", "eye_movements_mixed", "covertype_mixed", "albert", "road-safety",
    #"california",
    #"credit",
    #"heloc",
    #"jannis",
    #"Diabetes130US",
    "eye_movements",
    #"Higgs",
    #"default-of-credit-card-clients",
    #"MiniBooNE",
    #"bank-marketing",
    #"Bioresponse",
    #"bank-marketing",
    #"MagicTelescope",
    #"house_16H",
    #"pol",
    #"covertype",
    #"electricity",
    #"adult",
    #"german",
    #"compas",
    #"diabetes",
    #"breast_cancer",
]


seed = 123
# tf.random.set_seed(seed)
# np.random.seed(seed)


RUN_ALIBI = True # (1)True(2)False
TRAIN_MODEL = False # (1)True(2)False

RUN_WACHTER = False # (1)True(2)False
RUN_GS = False # (1)True(2)False
RUN_PROTO = True # (1)True(2)False
RUN_DICE = False # (1)True(2)False

MODELS_TO_RUN =  [  "dt",  "rfc"] # (1)dt, rfc, nn(2)nn

num_instances = 20 # (1)&(2) 20
num_cf_per_instance = 1 # (1)&(2)5


# github (tf, alibi).
if RUN_ALIBI:
    tf.get_logger().setLevel(40)
    tf.compat.v1.disable_v2_behavior()
    tf.keras.backend.clear_session()
    tf.compat.v1.disable_eager_execution()
    #############################################



#    pd.options.mode.chained_assignment = None

#    print("TF version: ", tf.__version__)
    print("Eager execution enabled: ", tf.executing_eagerly())  # False    


print("TF version: ", tf.__version__)
print("Eager execution enabled: ", tf.executing_eagerly())  # False    

import tensorflow as tf
print(tf.__version__)

#### Select dataset ####
for dataset_name in datasets:  # [adult, german, compas]
    print(f"LOADING Dataset: [{dataset_name}]\n")
    
    if dataset_name == "electricity_mixed":
        dataset_loading_fn = load_electricity_mixed_df
    elif dataset_name == "eye_movements_mixed":
        dataset_loading_fn = load_eye_movements_mixed_df
    elif dataset_name == "covertype_mixed":
        dataset_loading_fn = load_covertype_mixed_df
    elif dataset_name == "albert":
        dataset_loading_fn = load_albert_df
    elif dataset_name == "road-safety":
        dataset_loading_fn = load_road_safety_df
    elif dataset_name == "california":
        dataset_loading_fn = load_california_df
    elif dataset_name == "credit":
        dataset_loading_fn = load_credit_df
    elif dataset_name == "heloc":
        dataset_loading_fn = load_heloc_df
    elif dataset_name == "jannis":
        dataset_loading_fn = load_jannis_df
    elif dataset_name == "Diabetes130US":
        dataset_loading_fn = load_Diabetes130US_df
    elif dataset_name == "eye_movements":
        dataset_loading_fn = load_eye_movements_df
    elif dataset_name == "Higgs":
        dataset_loading_fn = load_Higgs_df
    elif dataset_name == "default-of-credit-card-clients":
        dataset_loading_fn = load_default_of_credit_card_clients_df
    elif dataset_name == "MiniBooNE":
        dataset_loading_fn = load_MiniBooNE_df
    elif dataset_name == "bank-marketing":
        dataset_loading_fn = load_bank_marketing_df
    elif dataset_name == "Bioresponse":
        dataset_loading_fn = load_Bioresponse_df
    elif dataset_name == "MagicTelescope":
        dataset_loading_fn = load_MagicTelescope_df
    elif dataset_name == "house_16H":
        dataset_loading_fn = load_house_16H_df
    elif dataset_name == "pol":
        dataset_loading_fn = load_pol_df
    elif dataset_name == "pol":
        dataset_loading_fn = load_pol_df
    elif dataset_name == "covertype":
        dataset_loading_fn = load_covertype_df
    elif dataset_name == "electricity":
        dataset_loading_fn = load_electricity_df
    elif dataset_name == "adult":
        dataset_loading_fn = load_adult_df
    elif dataset_name == "german":
        dataset_loading_fn = load_german_df
    elif dataset_name == "compas":
        dataset_loading_fn = load_compas_df
    elif dataset_name == "diabetes":
        dataset_loading_fn = load_diabetes_df
    elif dataset_name == "breast_cancer":
        dataset_loading_fn = load_breast_cancer_df
    else:
        raise Exception("Unsupported dataset")

    print("\tPreprocessing dataset...\n")
    df_info = preprocess_df(dataset_loading_fn)

    train_df, test_df = train_test_split(
        df_info.dummy_df, train_size=0.8, random_state=seed, shuffle=True
    )
    X_train = np.array(train_df[df_info.ohe_feature_names])
    y_train = np.array(train_df[df_info.target_name])
    X_test = np.array(test_df[df_info.ohe_feature_names])
    y_test = np.array(test_df[df_info.target_name])

    if TRAIN_MODEL:
        print("\tTraining models...\n")
        ## Train models.
        models = train_three_models(X_train, y_train)
        ## Save models.
        save_three_models(models, dataset_name)

    ### Load models
    print("\tLoading models...\n")
    models = load_three_models(X_train.shape[-1], dataset_name)

    ### Print out accuracy on testset.
    evaluation_test(models, X_test, y_test)

    if dataset_name in numerical_datasets:
        # run the cf algorithms supporting categorical data.

        # watcher and gs can only run for the datasets containing numerical data only.
        if RUN_ALIBI:
            
            if RUN_WACHTER:
                print_f.print_block(title="Counterfactual Algorithm", content="Watcher")
                results = util_watcher.generate_watcher_result(
                    df_info,
                    train_df,
                    models,
                    num_instances,
                    num_cf_per_instance,
                    X_train,
                    X_test,
                    y_test,
                    max_iters=1000,
                    models_to_run=MODELS_TO_RUN,
                    output_int=True,
                )
                result_dfs = util_watcher.process_result(results, df_info)
                save_result_as_csv("watcher", dataset_name, result_dfs)

        # else:
    
        if RUN_GS:
            print_f.print_block(title="Counterfactual Algorithm", content="GS")
            results, instance_cfs = util_gs.generate_gs_result(
                df_info, test_df, models, num_instances, num_cf_per_instance, 2000
            )
            result_dfs = util_gs.process_results(df_info, results)
            save_result_as_csv("GS", dataset_name, result_dfs)

        if RUN_ALIBI:
            print_f.print_block(title="Counterfactual Algorithm", content="Prototype")
            if RUN_PROTO:
                results = util_cf_proto.generate_cf_proto_result(
                    df_info,
                    train_df,
                    models,
                    num_instances,
                    num_cf_per_instance,
                    X_train,
                    X_test,
                    y_test,
                    max_iters=1000,
                    models_to_run=MODELS_TO_RUN,
                    output_int=True,
                )
                result_dfs = util_cf_proto.process_result(results, df_info)
                save_result_as_csv("proto", dataset_name, result_dfs)
            else:
                print("No prototype running")
        else:
            print("HERE1")
            print_f.print_block(title="Counterfactual Algorithm", content="DiCE")
            if RUN_DICE:
                print("HERE2")
                results = util_dice.generate_dice_result(
                    df_info,
                    test_df,
                    models,
                    num_instances,
                    num_cf_per_instance,
                    sample_size=50,
                    models_to_run=MODELS_TO_RUN,
                )
                print("HERE3")
                result_dfs = util_dice.process_results(df_info, results)
                print("SAVE")
                save_result_as_csv("dice", dataset_name, result_dfs)
                print("DONE")
        
        
