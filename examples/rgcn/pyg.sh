icdm_sesison1_dir="data/"
icdm_sesison2_dir="data/"
# pyg_data_session1="data/icdm"
pyg_data_session2="data/icdm2022_session2_pseudo_label2"
# test_ids_session1="data/icdm2022_session1_test_ids.txt"
test_ids_session2="data/icdm2022_session2_test_ids.txt"

# sesison1 data generator
#python utils/format_pyg.py --graph=$icdm_sesison1_dir"icdm2022_session1_edges.csv" \
#        --node=$icdm_sesison1_dir"icdm2022_session1_nodes.csv" \
#        --label=$icdm_sesison1_dir"icdm2022_session1_train_labels.csv" \
#        --storefile=$pyg_data_session1

# sesison2 data generator
# python utils/format_pyg.py --graph=$icdm_sesison2_dir"icdm2022_session2_edges.csv" \
#        --node=$icdm_sesison2_dir"icdm2022_session2_nodes.csv" \
#        --label=$icdm_sesison2_dir"icdm2022_session2_pseudo_labels2.csv" \
#        --storefile=$pyg_data_session2


# Training: session 1 (save model at best_models/$model_id.pth)
python main.py --dataset "data/session1_plus_sub.pt" \
         --config_file config.json

# Inference: session1 1. loading model $model_id 2. reading test_ids 3. generator .json file
# python main.py --dataset "data/icdm.pt" \
#         --test-file "data/icdm2022_session1_test_ids.txt" \
#         --config_file config.json \
#         --inference True