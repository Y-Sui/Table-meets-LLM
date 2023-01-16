#python unified_benchmark_generator.py --dataset feverous tabfact sqa hybridqa totto --objective zero --split validation
#python unified_benchmark_generator.py --dataset feverous tabfact sqa hybridqa totto --objective zero --split validation --use_structure_mark
#python unified_benchmark_generator.py --dataset feverous tabfact sqa hybridqa totto --objective zero --split validation --add_grammar
#python unified_benchmark_generator.py --dataset feverous tabfact sqa hybridqa totto --objective zero --split validation --use_structure_mark --add_grammar
#python unified_benchmark_generator.py --dataset feverous tabfact sqa hybridqa totto --objective zero --split validation --change_order

python unified_benchmark_generator.py --dataset sqa hybridqa totto --objective zero --split validation
python unified_benchmark_generator.py --dataset sqa hybridqa totto --objective zero --split validation --use_structure_mark
python unified_benchmark_generator.py --dataset sqa hybridqa totto --objective zero --split validation --add_grammar
python unified_benchmark_generator.py --dataset sqa hybridqa totto --objective zero --split validation --use_structure_mark --add_grammar
python unified_benchmark_generator.py --dataset sqa hybridqa totto --objective zero --split validation --change_order

#python generated/benchmark/jsonl_concat.py