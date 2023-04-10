from math import ceil
import sys, os, mpld3, matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

BENCHMARK_RESULTS = sys.argv[1]
PLOTS_OUT_DIR = sys.argv[2]
NUM_COLS_IN_PDF = int(sys.argv[3]) if len(sys.argv) > 3 else 2

def plot_benchmark(benchmark_result):
    matplotlib.rcParams.update({'font.size': 12})
    single_pdf = PdfPages(os.sep.join([PLOTS_OUT_DIR, 'singlepage.pdf']))
    single_html = ''
    for dataset_name in sorted(benchmark_result.keys()):
        for figure_id in benchmark_result[dataset_name].keys():
            current_pdf = PdfPages(os.sep.join([PLOTS_OUT_DIR, 'pdf', 
                dataset_name+ '_' + figure_id.replace("$","__").lower() + '.pdf']))
            num_plots = len(benchmark_result[dataset_name][figure_id].keys())
            num_cols = NUM_COLS_IN_PDF
            num_rows = ceil(float(num_plots)/float(num_cols))
            fig=plt.figure(figsize=(13,num_rows*5))
            plot_number = 1
            for plot_id in sorted(benchmark_result[dataset_name][figure_id].keys()):
                legend = []
                plt.subplot(num_rows, num_cols, plot_number)
                for n_jobs in benchmark_result[dataset_name][figure_id][plot_id].keys():
                    x_values = list(sorted(
                        benchmark_result[dataset_name][figure_id][plot_id][n_jobs].keys()))
                    y_values = list(map(
                        lambda x: benchmark_result[dataset_name][figure_id][plot_id][n_jobs][x][0],
                        x_values) )
                    plt.plot(x_values, y_values, marker='o')
                    legend.append('n_jobs = '+str(n_jobs))
                plt.xlabel('threshold')
                plt.ylabel('time taken (seconds)')
                plt.legend(legend, loc='upper right')
                plt.title(plot_id)
                plot_number += 1
            title = 'Dataset: ' + dataset_name + ', Left join attribute: ' + \
                    figure_id.split('$')[0] + ', Right join attribute: ' + \
                    figure_id.split('$')[1]
            fig.suptitle(title, fontsize=20)
            fig.tight_layout()
            fig.subplots_adjust(top=0.95)
            current_pdf.savefig()
            single_pdf.savefig()
            current_pdf.close()
            plt.close()
    single_pdf.close()
    # write single html file containing all plots

if __name__ == '__main__':
    benchmark_result = {}
    benchmark_files = sorted(os.listdir(BENCHMARK_RESULTS))

    for file_name in benchmark_files:
        file_handle = open(os.sep.join([BENCHMARK_RESULTS, file_name])) 
        header_flag = True
        dataset_name = file_name.split('_')[0]

        for line in file_handle:
            if header_flag:
                header_flag = False
                continue

            record = line.split(',')
            l_join_attr = record[0]
            r_join_attr = record[1]
            sim_measure_type = record[2]
            tokenizer = record[3]
            threshold = record[4]
            n_jobs = record[5]
            candset_size = record[6]
            elapsed_time = record[7]
            figure_id = '$'.join([l_join_attr, r_join_attr])
            plot_id = '_'.join([sim_measure_type, tokenizer])    

            if benchmark_result.get(dataset_name) == None:
                benchmark_result[dataset_name] = {}

            if benchmark_result[dataset_name].get(figure_id) == None:             
                benchmark_result[dataset_name][figure_id] = {}
   
            if benchmark_result[dataset_name][figure_id].get(plot_id) == None:
                benchmark_result[dataset_name][figure_id][plot_id] = {}

            if benchmark_result[dataset_name][figure_id][plot_id].get(n_jobs) == None:
                benchmark_result[dataset_name][figure_id][plot_id][n_jobs] = {}

            benchmark_result[dataset_name][figure_id][plot_id][n_jobs][threshold] = (elapsed_time, candset_size)

        file_handle.close()

    # create output directory, if needed
    if not os.path.exists(PLOTS_OUT_DIR):                                          
        os.makedirs(PLOTS_OUT_DIR)

    pdf_out_dir = os.sep.join([PLOTS_OUT_DIR, 'pdf']) 
    if not os.path.exists(pdf_out_dir):
        os.makedirs(pdf_out_dir)

    plot_benchmark(benchmark_result)