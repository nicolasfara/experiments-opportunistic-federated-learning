import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import matplotlib
import glob
import re 


out_dir = 'data-movement-cleaned/'


def clean_csv(file):
    lines = []
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    regex1 = re.compile(r'\bNaN\b')
    regex2 = re.compile(r'^#(?!.*\btime\b)')
    regex_union = re.compile(regex1.pattern + '|' + regex2.pattern)
    f = open(file, "r")
    for line in f:
        if not regex_union.search(line):
            if line[0] == '#':
                line = line[2:]
            lines.append(line)
    with open(out_dir + file[13:], "w") as file:
        for line in lines:
            file.write(line)


def avg_dataframes(dfs):
    mean_df = dfs[0]
    for d in dfs[1:]:
        mean_df += d 
    mean_df = mean_df / len(dfs)
    return mean_df

def plot_metric(data, metric):
    plt.figure(figsize=(10, 6))
    colors_v = sns.color_palette("colorblind", 10) 
    ax = sns.lineplot(data=data, x='time', y=metric, color=colors_v[0])
    plt.xlabel('time')
    plt.ylabel(metric)
    plt.grid(True)
    ax.yaxis.grid(True)
    plt.savefig(f'charts/movement/{metric}-mean.pdf', dpi=500)
    plt.close()

if __name__ == '__main__':

    matplotlib.rcParams.update({'axes.titlesize': 18})
    matplotlib.rcParams.update({'axes.labelsize': 18})

    data_dir = 'data-movement'
    file_names = glob.glob(f'{data_dir}/*.csv')
    for file in file_names:
        clean_csv(file)

    Path('charts/movement').mkdir(parents=True, exist_ok=True)
    cleaned_file_names = glob.glob(f'{out_dir}*.csv')

    dfs = []
    for file_name in cleaned_file_names:
        df = pd.read_csv(file_name, sep=' ')
        dfs.append(df)

    mean_df = avg_dataframes(dfs)

    plot_metric(mean_df, 'ValidationLoss')
    plot_metric(mean_df, 'SameLeader')

