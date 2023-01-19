## generate form downstream tasks
#python unified_babel_convertor.py --task formlm_opt formlm_qa formlm_block_type --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_form_20230115_log/
## ==============================
## generate table/databases downstream tasks
#python unified_babel_convertor.py --task cosql dart tabfact feverous tabfact hybridqa spider totto sql2text logic2text sqa webqsp --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_20230113_log/
## ==============================
# generate totto sequencization downstream tasks
#python unified_babel_convertor.py --task totto --objective heur_0 heur_1 heur_2 heur_3 heur_3 heur_4 heur_5 heur_6 heur_7 --split train validation --unified --unified_file_output ./exps/prompt_selection_20230113_log/

# generate table-based downstream tasks (one-shot)
# 0_1_2_3
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --linear_func html --unified --unified_file_output ./exps/downstream_tasks_20230119_manual_log/0_1_2_3 --use_role_prompting --add_table_size --use_format_explanation --use_partition_mark
# 0_2_3
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --linear_func html --unified --unified_file_output ./exps/downstream_tasks_20230119_manual_log/0_2_3 --use_role_prompting --use_format_explanation --use_partition_mark
# 0_1_3
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --linear_func html --unified --unified_file_output ./exps/downstream_tasks_20230119_manual_log/0_1_3 --add_table_size --use_role_prompting --use_format_explanation
# 1_2_3
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --linear_func html --unified --unified_file_output ./exps/downstream_tasks_20230119_manual_log/1_2_3 --use_format_explanation --add_table_size --use_partition_mark

## generate self-augmented information
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective heur_8 heur_9 heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230119_self_augmented_log/main --linear_func html --use_format_explanation --add_table_size --use_partition_mark

python unified_babel_convertor.py --task tabfact feverous --objective heur_0 heur_1 heur_2 heur_3 heur_5 heur_7 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230119_self_augmented_log/minor --linear_func html --use_format_explanation --add_table_size --use_partition_mark

