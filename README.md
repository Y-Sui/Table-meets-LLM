# GPT4Table:

SUC is a useful benchmark for detecting table structural understanding capabilities proposed in the paper " `GPT4Table: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study`".

### Benchmark Setting

In SUC, we design several specific tasks for each structural understanding capability. The tasks are designed with increasing difficulty. Please find the relevant codes in unified_benchmark_generator.py. The benchmark supports zero-shot, 1-shot and multiple input choices. More detailed commands can be found here.

```bash
# Multiple arguments can be set up when generating the benchmark
--dataset, choices=[tabfact, feverous, sqa, hybridqa, totto], help="choose which dataset you intend to use in your experiments"
--liearize_list, choices=[nl_sep, html, markdown, mark_down_grid, json, xml, latex], help="choose which linearization function you want to use, currently, SUC supports using nl_sep, html, markdown, mark_down_grid, json, xml, latex"
--use_partition_mark, help="whether to use partition mark in your experiments"
--use_format_explanation, help="whether to use format explanation in your experiments"
--use_role_prompting, help="whether to use use_role_prompting in your experiments"
--swap_input_order, help="whether to put external text (like questions, statement) ahead of tables."
--objective, choices=[oneshot, zero-shot], help="whether to add 1-shot example to the prompt"

# find examples of multiple arguments in unified_benchmark_generate.sh
```

## Downstream Tasks Setting

Based on our findings and insights over SUC comparisons, we find that several combination of input designs will highly affect the LLMs performance on SUC tasks. In this paper, we give some guidance on how to apply our benchmark insights to promote LLMs performance on downstream tasks, and examine using  self-augmented prompting to generate additional knowledge with LLMs self-knowledge. The code associated with downstream tasks can be found in unified_babel_convertor.py. The downstream tasks setting support both manual prompting engineering and self-augmented prompting. Multiple prompt choices can be found in config.py.

```bash
# generate table/databases downstream tasks
python unified_babel_convertor.py --task cosql dart tabfact feverous tabfact hybridqa spider totto sql2text logic2text sqa webqsp --objective zero --split train validation --unified --unified_file_output ./exps/downstream_tasks_20230113_log/

# generate self-augmented information
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_8 --split validation --unified --unified_file_output  ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_8  --linear_func html
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_9 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_9  --linear_func html
python unified_babel_convertor.py --task totto tabfact hybridqa sqa feverous --objective oneshot --heuristic heur_10 --split validation --unified --unified_file_output ./exps/downstream_tasks_20230120_self_augmented_p2_log/heur_10 --linear_func html


# more detailed information can be found in unified_bael_convertor.sh
```

## Additional Info

- Dataset Collection Code can be found in scripts/dataset_collection. We use huggingface dataset as our dataloader.
- Serialization Functions can be found in utils/structured_data_linearize.py.

## Reference

`@misc{sui2023gpt4table,
      title={GPT4Table: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study}, 
      author={Yuan Sui and Mengyu Zhou and Mingjie Zhou and Shi Han and Dongmei Zhang},
      year={2023},
      eprint={2305.13062},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}`
