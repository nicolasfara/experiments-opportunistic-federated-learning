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
    mean_df["SameLeader"] = 1 - mean_df["SameLeader"]
    return mean_df

def avg_std_dataframes(dfs):
    # Calculate the mean DataFrame
    mean_df = sum(dfs) / len(dfs)

    # Calculate the standard deviation DataFrame
    std_df = pd.concat(dfs).groupby(level=0).std()
    minus_deviation = mean_df - std_df
    plus_deviation = mean_df + std_df
    minus_deviation['time'] = mean_df['time']
    plus_deviation['time'] = mean_df['time']
    ## stack the dataframes with the mean in one dataframe
    return pd.concat([mean_df, minus_deviation, plus_deviation])

def plot_metric(data, metric, yname=None):
    plt.figure(figsize=(12, 8))
    colors_v = sns.color_palette("colorblind", 10) 
    ax = sns.lineplot(data=data, x='time', y=metric, color=colors_v[0])
    plt.xlabel('time')
    if yname:
        plt.ylabel(yname)
    else:
        plt.ylabel(metric)
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)
    plt.savefig(f'charts/movement/{metric}-mean.pdf', dpi=500)
    plt.close()

if __name__ == '__main__':

    matplotlib.rcParams.update({'axes.titlesize': 30})
    matplotlib.rcParams.update({'axes.labelsize': 30})
    matplotlib.rcParams.update({'xtick.labelsize': 25})
    matplotlib.rcParams.update({'ytick.labelsize': 25})
    plt.rcParams.update({"text.usetex": True})
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

    mean_and_std = avg_std_dataframes(dfs)
    mean_df = avg_dataframes(dfs)

    plot_metric(mean_and_std, 'ValidationLoss', "$NLL - Validation$")
    plot_metric(mean_df, 'SameLeader', "$DL$")
    plot_metric(mean_and_std, 'AreaCount', "$|A|$")


