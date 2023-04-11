import json
import os
import time
import pandas as pd
import py_stringmatching as sm
import py_stringsimjoin as ssj
from py_stringmatching.tokenizer.delimiter_tokenizer import DelimiterTokenizer
from py_stringmatching.tokenizer.qgram_tokenizer import QgramTokenizer
from itertools import product

TOKENIZERS = {'SPACE_DELIMITER': DelimiterTokenizer(delim_set={' '}, return_set=True),
              '2_GRAM': QgramTokenizer(qval=2, padding=False, return_set=True),
              '3_GRAM': QgramTokenizer(qval=3, padding=False, return_set=True),
              '4_GRAM': QgramTokenizer(qval=4, padding=False, return_set=True),
              '5_GRAM': QgramTokenizer(qval=5, padding=False, return_set=True),
              '2_GRAM_BAG': QgramTokenizer(qval=2),
              '3_GRAM_BAG': QgramTokenizer(qval=3)
              }
FILTERS = {
    "OVERLAP_FILTER": ssj.OverlapFilter,
    "SIZE_FILTER": ssj.SizeFilter,
    "PREFIX_FILTER": ssj.PrefixFilter,
    "POSITION_FILTER": ssj.PositionFilter,
    "SUFFIX_FILTER": ssj.SuffixFilter
}

SIM_FUNC = {
    'COSINE': sm.Cosine().get_sim_score,
    'DICE': sm.Dice().get_sim_score,
    'JACCARD': sm.Jaccard().get_sim_score,
    'OVERLAP_COEFFICIENT': sm.OverlapCoefficient().get_sim_score,
    'LEVENSHTEIN': sm.Levenshtein().get_sim_score,
    'TF-IDF': sm.TfIdf,
}

DATA_PATH = os.sep.join([os.getcwd(), 'dataset'])
SCRIPTS_FILE = 'scripts.json'
EXECUTION_TIMES = 1
BENCHMARK_DIRECTORY = 'benchmark_results'
APPLY_RESULTS_DIRECTORY = 'apply_results'

def load_scripts():
    with open(SCRIPTS_FILE, 'r') as js_file:
        scripts = json.load(js_file)['scripts']
    return scripts

def load_data_and_test():
    if not os.path.exists(BENCHMARK_DIRECTORY):
        os.makedirs(BENCHMARK_DIRECTORY)
    scripts = load_scripts()
    output_header = ','.join(['left join attr', 'right join attr',
                              'similarity measure type', 'tokenizer',
                              'threshold', 'n_jobs', 'cand_set_size', 'avg_time'])
    write_header = False
    for idx, script in enumerate(scripts):
        print(script['ltable'])
        l_path = os.sep.join([DATA_PATH, *script['ltable']])
        r_path = os.sep.join([DATA_PATH, *script['rtable']])
        out_file_path = os.sep.join([BENCHMARK_DIRECTORY, script['dataset_name']])
        out_path = os.sep.join([BENCHMARK_DIRECTORY, script['dataset_name']])
        output_file = open(out_path + "_benchmark_" + str(idx) + '.csv', 'a')
        add_header = not os.path.exists(out_file_path)
        if add_header:
            output_file.write('%s\n' % output_header)
        l_table = pd.read_csv(l_path, encoding=script['ltable_encoding'])
        r_table = pd.read_csv(r_path, encoding=script['rtable_encoding'])
        test(script, output_file, l_table, r_table, idx)
        output_file.close()


def test(script, output_file, l_table, r_table, idx_script):
    total_info_obj = product(
        script['sim_funcs'],
        script['sim_measure_types'],
        script['tokenizers'],
        script['scale_filters'],
        script['thresholds'],
        script['n_jobs']
    )
    for sim_funcs, sim_measure_type, tokenizer, scale_filter, threshold, n_jobs in total_info_obj:
        if tokenizer in ["SPACE_DELIMITER"] and sim_measure_type != 'EDIT_DISTANCE':
            continue
        sim_func = SIM_FUNC[sim_funcs]
        tok = TOKENIZERS[tokenizer]
        if scale_filter == "OVERLAP_FILTER":
            s_filter = FILTERS[scale_filter](tok, overlap_size=1, comp_op='>=', allow_missing=False)
        else:
            s_filter = FILTERS[scale_filter](tok, sim_measure_type, threshold,
                                             allow_empty=True, allow_missing=False)
        sum_time = 0
        for i in range(EXECUTION_TIMES):
            start_time = time.time()
            candidate_set = s_filter.filter_tables(
                l_table, r_table,
                script['l_id_attr'], script['r_id_attr'],
                script['l_join_attr'], script['r_join_attr'],
                l_out_attrs=None, r_out_attrs=None,
                l_out_prefix='l_', r_out_prefix='r_',
                n_jobs=n_jobs, show_progress=True)
            output_table = ssj.apply_matcher(candidate_set,
                                             'l_' + script['l_id_attr'], 'r_' + script['r_id_attr'], l_table, r_table,
                                             script['l_id_attr'], script['r_id_attr'],
                                             script['l_join_attr'], script['r_join_attr'],
                                             tokenizer=tok, sim_function=sim_func, threshold=threshold,
                                             comp_op='>=', allow_missing=False,
                                             l_out_attrs=[script['l_join_attr']], r_out_attrs=[script['r_join_attr']],
                                             l_out_prefix='l_', r_out_prefix='r_',
                                             out_sim_score=True, n_jobs=n_jobs, show_progress=True)
            
            sum_time += (time.time() - start_time)
            cand_set_size = len(output_table)
            avg_time_elapsed = sum_time / EXECUTION_TIMES
            output_record = ','.join([str(script['l_join_attr']), str(script['r_join_attr']),
                                      str(sim_measure_type), str(tokenizer),
                                      str(threshold), str(n_jobs),
                                      str(cand_set_size), str(avg_time_elapsed)])
            if not os.path.exists(BENCHMARK_DIRECTORY):
                os.makedirs(BENCHMARK_DIRECTORY)
            if not os.path.exists(os.sep.join([BENCHMARK_DIRECTORY,idx_script])):
                os.makedirs(os.sep.join([BENCHMARK_DIRECTORY, idx_script]))
            output_table.to_csv(os.sep.join([BENCHMARK_DIRECTORY, 
                                            str(sim_measure_type) + '_' + tokenizer + '_' + scale_filter + '_' + sim_funcs + '_' + str(idx_script) + 'csv']))

if __name__ == "__main__":
    load_data_and_test()
