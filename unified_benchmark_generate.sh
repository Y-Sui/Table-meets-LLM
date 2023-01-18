# zero-shot
python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective zero-shot --split validation --linearize_list html --use_partition_mark --use_format_explanation --use_role_prompting
#=============================
## few-shot
#python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective few-shot --split validation --linearize_list html --use_partition_mark --use_format_explanation --use_role_prompting
##=============================
## 1-shot
#python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_format_explanation --use_partition_mark --use_role_prompting
#
## w/o role_prompting
#python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_format_explanation --use_partition_mark
#
## w/o partition_mark
#python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_format_explanation --use_role_prompting
#
## w/o format_explanation
#python unified_benchmark_generator.py --dataset tabfact feverous sqa hybridqa totto --objective 1-shot --split validation --use_role_prompting --use_partition_mark