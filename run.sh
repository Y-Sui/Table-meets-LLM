#python unified_babel_convertor.py --task cosql --objective zero --split validation
#python unified_babel_convertor.py --task dart --objective zero --split validation
#python unified_babel_convertor.py --task tabfact --objective zero --split validation
#python unified_babel_convertor.py --task feverous --objective zero --split validation
#python unified_babel_convertor.py --task tabfact --objective zero --split validation
#python unified_babel_convertor.py --task hybridqa --objective zero --split validation
#python unified_babel_convertor.py --task spider --objective zero --split validation
#python unified_babel_convertor.py --task totto --objective zero heur_0 heur_1 heur_2 heur_3 heur_3 heur_4 heur_5 heur_6 heur_7 --split validation
#python unified_babel_convertor.py --task sql2text --objective zero --split validation
#python unified_babel_convertor.py --task logic2text --objective zero --split validation
#python unified_babel_convertor.py --task sqa --objective zero --split validation
#python unified_babel_convertor.py --task webqsp --objective zero --split validation
#python unified_babel_convertor.py --task gittables --objective zero --split validation

#python unified_babel_convertor.py --task cosql dart tabfact feverous tabfact hybridqa spider totto sql2text logic2text sqa webqsp --objective zero --split validation --unified --unified_file_output ./exps/downstream_tasks_20230113_log/
python unified_babel_convertor.py --task totto --objective heur_0 heur_1 heur_2 heur_3 heur_3 heur_4 heur_5 heur_6 heur_7 --split validation --unified --unified_file_output ./exps/prompt_selection_20230113_log/