# generate form downstream tasks
python unified_babel_convertor.py --task formlm_opt formlm_qa formlm_block_type --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_form_20230115_log/

# generate table/databases downstream tasks
python unified_babel_convertor.py --task cosql dart tabfact feverous tabfact hybridqa spider totto sql2text logic2text sqa webqsp --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_20230113_log/

# generate totto sequencization downstream tasks
python unified_babel_convertor.py --task totto --objective heur_0 heur_1 heur_2 heur_3 heur_3 heur_4 heur_5 heur_6 heur_7 --split train validation --unified --unified_file_output ./exps/prompt_selection_20230113_log/

# generate self-augmented information
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_8  --linear_func html
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_9  --linear_func html
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_10 --linear_func html