# generate form downstream tasks
#python unified_babel_convertor.py --task formlm_opt formlm_qa formlm_block_type --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_form_20230115_log/

# generate table/databases downstream tasks
#python unified_babel_convertor.py --task cosql dart tabfact feverous tabfact hybridqa spider totto sql2text logic2text sqa webqsp --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_20230113_log/

# generate totto sequencization downstream tasks
#python unified_babel_convertor.py --task totto --objective heur_0 heur_1 heur_2 heur_3 heur_3 heur_4 heur_5 heur_6 heur_7 --split train validation --unified --unified_file_output ./exps/prompt_selection_20230113_log/

# generate self-augmented information (phase 2)
#python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_8  --linear_func html
#python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_9  --linear_func html
#python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_10 --linear_func html

# 2023-02-19 generate multiple choices for downstream tasks evaluation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_downstream_tasks_ablation/0_1_3 --linearize_list html markdown json nl_sep xml --use_partition_mark --use_role_prompting --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_downstream_tasks_ablation/0_1 --linearize_list html markdown json nl_sep xml --use_partition_mark --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_downstream_tasks_ablation/0_3 --linearize_list html markdown json nl_sep xml --use_role_prompting --use_partition_mark
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_downstream_tasks_ablation/1_3 --linearize_list html markdown json nl_sep xml --use_role_prompting --use_format_explanation

# 2023-02-19 generate multiple choices for p2
# generate self-augmented information (phase 2)
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_8_0_1_3  --linearize_list html markdown json nl_sep xml --use_partition_mark --use_role_prompting --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_8_0_1  --linearize_list html markdown json nl_sep xml --use_partition_mark --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_8_0_3  --linearize_list html markdown json nl_sep xml --use_role_prompting --use_partition_mark
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_8_1_3  --linearize_list html markdown json nl_sep xml --use_role_prompting --use_format_explanation

python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_9_0_1_3  --linearize_list html markdown json nl_sep xml --use_partition_mark --use_role_prompting --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_9_0_1  --linearize_list html markdown json nl_sep xml --use_partition_mark --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_9_0_3  --linearize_list html markdown json nl_sep xml --use_role_prompting --use_partition_mark
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_9_1_3  --linearize_list html markdown json nl_sep xml --use_role_prompting --use_format_explanation

python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_10_0_1_3 --linearize_list html markdown json nl_sep xml --use_partition_mark --use_role_prompting --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_10_0_1 --linearize_list html markdown json nl_sep xml --use_partition_mark --use_format_explanation
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_10_0_3 --linearize_list html markdown json nl_sep xml --use_role_prompting --use_partition_mark
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230219_self_augmented_p1_log/heur_10_1_3 --linearize_list html markdown json nl_sep xml --use_role_prompting --use_format_explanation