import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from matplotlib import rc
import numpy as np
import pandas as pd
import seaborn as sns

from rliable import library as rly
from rliable import metrics
from rliable import plot_utils


params = {
    # 'mathtext.fontset' : 'stix',
    'mathtext.rm'      : 'serif',
    'font.family'      : 'serif',
    'font.serif'       : "Times New Roman", # or "Times"
    # 'text.usetex': True,
    # 'text.latex.preamble': r'\usepackage{amsmath}'

}
matplotlib.rcParams.update(params)


ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]
OUR_ALG_NAME = 'REM'


def score_normalization(res_dict, min_scores, max_scores):
    games = res_dict.keys()
    norm_scores = {}
    for game, scores in res_dict.items():
        if len(scores) == 0:
            continue
        norm_scores[game] = (scores - min_scores[game])/(max_scores[game] - min_scores[game])
    return norm_scores

def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


def load_json_scores(algorithm_name, normalize=True, random_scores=None, human_scores=None, data_path=Path('data'), subset=None):
    path = data_path / f'{algorithm_name}.json'
    with path.open('r') as f:
        raw_scores = json.load(f)
    if subset is None:
        subset = ATARI_100K_GAMES
    raw_scores = {game: np.array(val) for game, val in raw_scores.items() if game in subset}
    if normalize:
        hn_scores = score_normalization(raw_scores, random_scores, human_scores)
        hn_score_matrix = convert_to_matrix(hn_scores)
    else:
        hn_scores, hn_score_matrix = None, None
    return hn_scores, hn_score_matrix, raw_scores


def save_fig(fig, name):
    fig.savefig(f'figures/{name}.pdf', format='pdf', bbox_inches='tight')


def plot_atari_scores_results(plot_aggregates: bool = True, plot_introduction_iqm: bool = True, plot_performance_profiles: bool = True,
         plot_probability_of_improvement: bool = True, plot_latex_table: bool = True):
    rcParams['legend.loc'] = 'best'
    # rcParams['pdf.fonttype'] = 42
    # rcParams['ps.fonttype'] = 42
    # rc('text', usetex=False)
    RAND_STATE = np.random.RandomState(42)
    # sns.set_style("white")

    Path('figures').mkdir(exist_ok=True, parents=False)

    StratifiedBootstrap = rly.StratifiedBootstrap

    IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
    OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
    MEAN = lambda x: metrics.aggregate_mean(x)
    MEDIAN = lambda x: metrics.aggregate_median(x)

    _, _, raw_scores_random = load_json_scores('RANDOM', normalize=False)
    _, _, raw_scores_human = load_json_scores('HUMAN', normalize=False)
    RANDOM_SCORES = {k: v[0] for k, v in raw_scores_random.items()}
    HUMAN_SCORES = {k: v[0] for k, v in raw_scores_human.items()}

    # score_dict_efficientzero, score_efficientzero, raw_scores_efficientzero = load_json_scores('EfficientZero', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES)
    score_dict_simple, score_simple, _ = load_json_scores('SimPLe', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES)
    score_dict_iris, score_iris, _ = load_json_scores('IRIS', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES)
    score_dict_twm, score_twm, _ = load_json_scores('TWM', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES)
    score_dict_storm, score_storm, _ = load_json_scores('STORM', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES)
    score_dict_dreamerv3, score_dreamerv3, _ = load_json_scores('dreamerv3', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES)
    score_dict_rwm, score_rwm, _ = load_json_scores('REM', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES)

    score_data_dict_games = {
        # 'EfficientZero': score_dict_efficientzero,
        'SimPLe': score_dict_simple,
        'TWM': score_dict_twm,
        'STORM': score_dict_storm,
        'DreamerV3': score_dict_dreamerv3,
        'IRIS': score_dict_iris,
        f'{OUR_ALG_NAME} (ours)': score_dict_rwm,
    }

    all_score_dict = {
        # 'EfficientZero': score_efficientzero,
        'SimPLe': score_simple,
        'TWM': score_twm,
        'STORM': score_storm,
        'DreamerV3': score_dreamerv3,
        'IRIS': score_iris,
        f'{OUR_ALG_NAME} (ours)': score_rwm,
    }

    colors = sns.color_palette('colorblind')
    xlabels = ['SimPLe', 'DreamerV3', 'TWM', 'STORM', 'IRIS', f'{OUR_ALG_NAME} (ours)']
    color_idxs = [7, 4, 0, 6, 2, 1]
    ATARI_100K_COLOR_DICT = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))

    aggregate_func = lambda x: np.array([MEAN(x), MEDIAN(x), IQM(x), OG(x)])
    aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(all_score_dict, aggregate_func,
                                                                                reps=50000)

    for algo in aggregate_scores.keys():
        n_runs, n_games = all_score_dict[algo].shape
        assert n_games == len(ATARI_100K_GAMES)
        print(f"{algo.ljust(14)}: {n_runs:3d} runs")

    # Mean, Median, IQM and Optimality Gap:
    algorithms = xlabels

    if plot_aggregates:
        plt.rcParams['font.size'] = 14
        fig, axes = plot_utils.plot_interval_estimates(
            {k: v[:4] for k, v in aggregate_scores.items()},
            {k: v[:, :4] for k, v in aggregate_interval_estimates.items()},
            metric_names=['Mean (↑)', 'Median (↑)', 'Interquartile Mean (↑)', 'Optimality Gap (↓)'],
            algorithms=algorithms,
            colors=ATARI_100K_COLOR_DICT,
            xlabel_y_coordinate=-0.1,
            xlabel='Human Normalized Score',
            subfigure_width=5,
            row_height=0.7)
        for ax in axes:
            xmin, xmax = ax.get_xlim()
            ax.hlines(y=3.5, xmin=xmin, xmax=xmax, linewidth=1, color='gray')
        plt.show()
        save_fig(fig, 'aggregates')


    if plot_introduction_iqm:
        plt.rcParams['font.size'] = 14
        fig, axes = plot_utils.plot_interval_estimates(
            {k: v[2:3] for k, v in aggregate_scores.items()},
            {k: v[:, 2:3] for k, v in aggregate_interval_estimates.items()},
            metric_names=[''],
            algorithms=algorithms,
            colors=ATARI_100K_COLOR_DICT,
            xlabel_y_coordinate=-0.05,
            xlabel='Human Normalized Score (IQM, ↑)',
            subfigure_width=9,
            row_height=0.8)
        for ax in [axes]:
            xmin, xmax = ax.get_xlim()
            ax.hlines(y=3.5, xmin=xmin, xmax=xmax, linewidth=1, color='gray')
        plt.show()
        save_fig(fig, 'intro_IQM')

    # Performance profile:
    if plot_performance_profiles:
        plt.rcParams['font.size'] = 11
        short_plot = False
        score_dict = {key: all_score_dict[key] for key in algorithms}
        ATARI_100K_TAU = np.linspace(0.0, 8.0, 201)
        reps = 2000

        score_distributions, score_distributions_cis = rly.create_performance_profile(score_dict, ATARI_100K_TAU, reps=reps)
        height = 4.7 if not short_plot else 4.5
        fig, ax = plt.subplots(ncols=1, figsize=(8, height))

        plot_utils.plot_performance_profiles(
            score_distributions, ATARI_100K_TAU,
            performance_profile_cis=score_distributions_cis,
            colors=ATARI_100K_COLOR_DICT,
            xlabel=r'Human Normalized Score $(\tau)$',
            labelsize='xx-large',
            ax=ax)

        ax.axhline(0.5, ls='--', color='k', alpha=0.4)
        fake_patches = [mpatches.Patch(color=ATARI_100K_COLOR_DICT[alg],
                                       alpha=0.75) for alg in algorithms]
        legend = fig.legend(fake_patches, algorithms, loc='upper center',
                            fancybox=True, ncol=3,
                            fontsize='x-large',
                            bbox_to_anchor=(0.57, 0.93))
        suffix = '' if not short_plot else '_short'
        save_fig(fig, f'performance_profile{suffix}')


    # Probability of Improvement:
    if plot_probability_of_improvement:
        plt.rcParams['font.size'] = 11
        our_algorithm = algorithms[-1]
        all_pairs = {}
        for alg in (algorithms):
            if alg == our_algorithm:
                continue
            pair_name = f'{our_algorithm}_{alg}'
            all_pairs[pair_name] = (all_score_dict[our_algorithm], all_score_dict[alg])

        probabilities, probability_cis = {}, {}
        reps = 1000
        probabilities, probability_cis = rly.get_interval_estimates(all_pairs, metrics.probability_of_improvement,
                                                                    reps=reps)
        fig, ax = plt.subplots(figsize=(4, 3))
        h = 0.6
        algorithm_labels = []

        for i, (alg_pair, prob) in enumerate(probabilities.items()):
            _, alg1 = alg_pair.split('_')
            algorithm_labels.append(alg1)
            (l, u) = probability_cis[alg_pair]
            ax.barh(y=i, width=u - l, height=h, left=l, color=ATARI_100K_COLOR_DICT[alg1], alpha=0.75)
            ax.vlines(x=prob, ymin=i - 7.5 * h / 16, ymax=i + (6 * h / 16), color='k', alpha=0.85)

        ax.set_yticks(range(len(algorithm_labels)))
        ax.set_yticklabels(algorithm_labels)

        ax.set_xlim(0, 1)
        ax.axvline(0.5, ls='--', color='k', alpha=0.4)
        ax.set_title(fr'P({OUR_ALG_NAME} > $Y$)', size='xx-large')
        plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
        ax.set_ylabel(r'Algorithm $Y$', size='xx-large')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        fig.subplots_adjust(wspace=0.25, hspace=0.45)
        save_fig(fig, 'probability_of_improvement')


    # Print LaTeX Table:
    if plot_latex_table:
        first_row = ["Game", "Random", "Human", "SimPLe", "DreamerV3", "TWM", "STORM", "IRIS", r"\textsc{" + f"{OUR_ALG_NAME}" + "} (ours)"]
        rows = [first_row]

        baselines_first_idx = 6

        # Raw scores
        for game in ATARI_100K_GAMES:
            raw_scores = [RANDOM_SCORES[game], HUMAN_SCORES[game]]
            raw_scores.extend([np.mean(
                score_data_dict_games[algo][game] * (HUMAN_SCORES[game] - RANDOM_SCORES[game]) + RANDOM_SCORES[game]) for
                               algo in aggregate_scores.keys()])
            idx_max_baselines = baselines_first_idx + np.argmax(raw_scores[baselines_first_idx:])
            idx_max_all = 2 + np.argmax(raw_scores[2:])
            raw_scores = [f"{x:.1f}" for x in raw_scores]
            raw_scores[idx_max_baselines] = f"\\textbf{{{raw_scores[idx_max_baselines]}}}"
            raw_scores[idx_max_all] = f"\\underline{{{raw_scores[idx_max_all]}}}"
            row = [game, *raw_scores]
            rows.append(row)

        # Aggregates
        first_col = ["\\#Superhuman (↑)", "Mean (↑)", "Median (↑)", "IQM (↑)", "Optimality Gap (↓)"]
        cols = [
            [0, 0, 0, 0, 1],  # Random
            [float('-inf'), 1, 1, 1, 0],  # Human
        ]
        for algo in aggregate_scores.keys():
            n_runs, n_games = all_score_dict[algo].shape
            assert n_games == len(ATARI_100K_GAMES)
            score_dict = score_data_dict_games[algo]
            sh = np.sum(np.mean(all_score_dict[algo], axis=0) >= 1)
            col = [sh, *aggregate_scores[algo]]
            cols.append(col)

        rows_ = np.array(cols).T
        for i, row in enumerate(rows_):
            idx_best_baselines = baselines_first_idx + (np.argmin(row[baselines_first_idx:]) if i == len(rows_) - 1 else np.argmax(row[baselines_first_idx:]))
            idx_best_all = 2 + (np.argmin(row[2:]) if i == len(rows_) - 1 else np.argmax(row[2:]))
            row = [f"{x:.{0 if i == 0 else 3}f}" if not math.isinf(x) else 'N/A' for x in row]
            row[idx_best_baselines] = f"\\textbf{{{row[idx_best_baselines]}}}"
            row[idx_best_all] = f"\\underline{{{row[idx_best_all]}}}"
            rows.append([first_col[i]] + row)

        rows = np.array(rows)
        for i in range(rows.shape[1]):
            max_len = max(map(len, rows[:, i])) + 1
            rows[:, i] = list(map(lambda x: x.ljust(max_len), rows[:, i]))

        rows = ['  &  '.join(row) for row in rows]

        for i, row in enumerate(rows):
            print(row + r'  \\')
            if i in [0, 26]:
                print(r"\midrule")


def plot_run_times_results(short: bool = False, orig_iris: bool = False):
    components = ['Tokenizer\nLearning (↓)', 'World Model\nTraining (↓)', 'Imagination (↓)', 'Combined (↓)']
    iris_times_our_config = [34, 45, 213]
    iris_times_orig_config = [34, 55, 309]
    iris_times = iris_times_our_config if not orig_iris else iris_times_orig_config
    our_times = [12, 39, 20]

    for times in [iris_times, our_times]:
        times.pop(0)
        times.append(np.sum(times))

    components = components[1:]

    speed_ups = []
    for i in range(len(iris_times)):
        speed_ups.append(iris_times[i] / our_times[i])

    height = 6 if not short else 3
    fig, ax = plt.subplots(figsize=(8, height))
    fontsize = 15

    bar_width = 0.3
    xs = np.arange(len(iris_times))

    colors = sns.color_palette('colorblind')
    iris_color = colors[2]
    our_color = colors[1]
    iris_bars = ax.bar(xs - bar_width*0.75, iris_times, bar_width, color=iris_color, label="IRIS")
    ours_bars = ax.bar(xs + bar_width*0.75, our_times, bar_width, color=our_color, label=f"{OUR_ALG_NAME} (ours)")

    ax.set_ylabel('Epoch Time (sec)', fontsize=fontsize)
    ax.set_xticks(xs, components, fontsize=fontsize)

    ax.legend(loc='best', fontsize=fontsize)

    ax.bar_label(ours_bars, labels=[f"{v:.3}x" for v in speed_ups], padding=10, fontsize=18, fontweight='bold')
    suffix = '' if not short else '_short'
    iris_suffix = '' if orig_iris else '_fair'
    save_fig(fig, f'times_comparison{iris_suffix}{suffix}')


def main():
    plot_atari_scores_results(
        plot_aggregates=False,
        plot_introduction_iqm=True,
        plot_performance_profiles=False,
        plot_probability_of_improvement=False,
        plot_latex_table=False
    )
    # plot_run_times_results(short=True, orig_iris=True)


if __name__ == '__main__':
    main()
