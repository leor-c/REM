import dataclasses
import json
import math
from enum import Enum
from pathlib import Path

import math
from pathlib import Path

import wandb

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import rc
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from rliable import library as rly
from rliable import metrics
from rliable import plot_utils

from results import load_json_scores, save_fig


params = {
    # 'mathtext.fontset' : 'stix',
    'mathtext.rm'      : 'serif',
    'font.family'      : 'serif',
    'font.serif'       : "Times New Roman", # or "Times"
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}'

}
matplotlib.rcParams.update(params)



ATARI_100K_GAMES = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo',
    'Krull', 'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert',
    'RoadRunner', 'Seaquest', 'UpNDown'
]
ABLATION_SUBSET = [
    'Assault',
    'Asterix',
    'ChopperCommand',
    'CrazyClimber',
    'DemonAttack',
    'Gopher',
    'Krull',
    'RoadRunner'
]
OUR_ALG_NAME = "REM"


class AblationType(Enum):
    NoPOP = "No POP"
    LSAC = r"No latent space {\ttfamily $\bf{C}$}"
    WMEmb = r"Separate {\ttfamily $\bf{M}$} emb."
    Tok4x4 = r"$4 \times 4$ tokenizer"
    NoActionInput = r"{\ttfamily $\bf{C}$} w/o action inputs"




def main(plot_aggregates: bool = True, plot_performance_profiles: bool = True,
         plot_probability_of_improvement: bool = True, plot_summary: bool = True, plot_latex_table: bool = True):
    rcParams['legend.loc'] = 'best'
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
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

    score_dict_rwm, score_rwm, _ = load_json_scores('RWM', random_scores=RANDOM_SCORES, human_scores=HUMAN_SCORES, subset=ABLATION_SUBSET)

    score_dict_sep_wm_emb, score_sep_wm_emb, _ = load_json_scores(
        'ablation-separate-wm-embeddings',
        random_scores=RANDOM_SCORES,
        human_scores=HUMAN_SCORES,
        data_path=Path('data') / 'ablations',
        subset=ABLATION_SUBSET
    )
    score_dict_no_pop, score_no_pop, _ = load_json_scores(
        'ablation-no-pop',
        random_scores=RANDOM_SCORES,
        human_scores=HUMAN_SCORES,
        data_path=Path('data') / 'ablations',
        subset=ABLATION_SUBSET
    )

    score_dict_4x4, score_4x4, _ = load_json_scores(
        'ablation-4x4-tokenizer',
        random_scores=RANDOM_SCORES,
        human_scores=HUMAN_SCORES,
        data_path=Path('data') / 'ablations',
        subset=ABLATION_SUBSET
    )
    score_dict_ab3, score_ab3, _ = load_json_scores(
        'ablation-no-latent-ac',
        random_scores=RANDOM_SCORES,
        human_scores=HUMAN_SCORES,
        data_path=Path('data') / 'ablations',
        subset=ABLATION_SUBSET
    )
    score_dict_ab4, score_ab4, _ = load_json_scores(
        'ablation-no-action-input',
        random_scores=RANDOM_SCORES,
        human_scores=HUMAN_SCORES,
        data_path=Path('data') / 'ablations',
        subset=ABLATION_SUBSET
    )


    score_data_dict_games = {
        f'{OUR_ALG_NAME}': score_dict_rwm,
        AblationType.NoPOP.value: score_dict_no_pop,
        AblationType.WMEmb.value: score_dict_sep_wm_emb,
        AblationType.Tok4x4.value: score_dict_4x4,
        AblationType.LSAC.value: score_dict_ab3,
        AblationType.NoActionInput.value: score_dict_ab4
    }

    all_score_dict = {
        f'{OUR_ALG_NAME}': score_rwm,
        AblationType.NoPOP.value: score_no_pop,
        AblationType.WMEmb.value: score_sep_wm_emb,
        AblationType.Tok4x4.value: score_4x4,
        AblationType.LSAC.value: score_ab3,
        AblationType.NoActionInput.value: score_ab4
    }

    colors = sns.color_palette('colorblind')
    xlabels = [f'{OUR_ALG_NAME}', AblationType.NoPOP.value, AblationType.WMEmb.value, AblationType.Tok4x4.value,
               AblationType.LSAC.value, AblationType.NoActionInput.value]
    color_idxs = [1, 0, 2, 3, 4, 5]
    ATARI_100K_COLOR_DICT = dict(zip(xlabels, [colors[idx] for idx in color_idxs]))

    algorithms = xlabels

    # Mean, Median, IQM and Optimality Gap:
    if plot_aggregates:

        aggregate_func = lambda x: np.array([MEAN(x), MEDIAN(x), IQM(x), OG(x)])
        aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(all_score_dict, aggregate_func, reps=50000)

        for algo in aggregate_scores.keys():
            n_runs, n_games = all_score_dict[algo].shape
            assert n_games == len(ABLATION_SUBSET)
            print(f"{algo.ljust(14)}: {n_runs:3d} runs")

        plt.rcParams['font.size'] = 14
        fig, axes = plot_utils.plot_interval_estimates(
            {k: v[:4] for k, v in aggregate_scores.items()},
            {k: v[:, :4] for k, v in aggregate_interval_estimates.items()},
            metric_names=['Mean', 'Median', 'Interquartile Mean', 'Optimality Gap'],
            algorithms=algorithms,
            colors=ATARI_100K_COLOR_DICT,
            xlabel_y_coordinate=-0.1,
            xlabel='Human Normalized Score',
            subfigure_width=4.5,
            row_height=0.7
        )
        # plt.show()
        save_fig(fig, 'ablations_aggregates')

    # Performance profile:
    if plot_performance_profiles:
        plt.rcParams['font.size'] = 11
        score_dict = {key: all_score_dict[key] for key in algorithms}
        ATARI_100K_TAU = np.linspace(0.0, 8.0, 201)
        reps = 2000

        score_distributions, score_distributions_cis = rly.create_performance_profile(score_dict, ATARI_100K_TAU, reps=reps)
        fig, ax = plt.subplots(ncols=1, figsize=(7.25, 4.7))

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
        save_fig(fig, 'ablations_performance_profile')

    # Probability of Improvement:
    if plot_probability_of_improvement:
        plt.rcParams['font.size'] = 15
        our_algorithm = algorithms[0]
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
        fig, ax = plt.subplots(figsize=(6, 4.5))
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
        ax.set_title(fr'P({our_algorithm} $ > Y$)', size='xx-large')
        plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
        ax.set_ylabel(r'Algorithm $Y$', size='xx-large')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        fig.subplots_adjust(wspace=0.25, hspace=0.45)
        save_fig(fig, 'ablations_probability_of_improvement')

    # Summary:
    if plot_summary:
        from rliable.plot_utils import _decorate_axis
        plt.rcParams['font.size'] = 15
        our_algorithm = algorithms[0]
        all_pairs = {}
        for alg in (algorithms):
            if alg == our_algorithm:
                continue
            pair_name = f'{our_algorithm}_{alg}'
            all_pairs[pair_name] = (all_score_dict[our_algorithm], all_score_dict[alg])

        probabilities, probability_cis = {}, {}
        reps = 1000
        probabilities, probability_cis = rly.get_interval_estimates(all_pairs, metrics.probability_of_improvement, reps=reps)

        aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(all_score_dict, IQM, reps=50000)
        h = 0.6
        figsize = (4.5 * 2, h * len(algorithms))
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        algorithm_labels = [OUR_ALG_NAME]

        ax, ax2 = axs[0], axs[1]
        for i, (alg_pair, prob) in enumerate(probabilities.items()):
            _, alg1 = alg_pair.split('_')
            algorithm_labels.append(alg1)
            (l, u) = probability_cis[alg_pair]
            ax.barh(y=i+1, width=u - l, height=h, left=l, color=ATARI_100K_COLOR_DICT[alg1], alpha=0.75)
            ax.vlines(x=prob, ymin=i+1 - 7.5 * h / 16, ymax=i+1 + (6 * h / 16), color='k', alpha=0.85)

        ax.set_yticks(range(len(algorithm_labels)))
        ax.set_yticklabels(algorithm_labels, fontsize='x-large')

        ax.set_xlim(0.35, 0.95)
        ax.set_ylim(-1 + 6*h/16, len(algorithm_labels)-7.5 * h / 16)
        ax.axvline(0.5, ls='--', color='k', alpha=0.4)
        ax.set_title(fr'P({our_algorithm} $ > Y$)', size='xx-large')
        _decorate_axis(ax, ticklabelsize='xx-large', wrect=5)
        ax.set_xticks([0.5, 0.7, 0.9])
        ax.grid(True, alpha=0.25)
        ax.spines['left'].set_visible(False)
        # plot_utils._annotate_and_decorate_axis(ax, labelsize='xx-large', ticklabelsize='xx-large')
        ax.set_ylabel(r'Ablation $Y$', size='xx-large')
        # ax.xaxis.set_major_locator(MaxNLocator(5))

        for alg_idx, algorithm in enumerate(algorithms):
            # Plot interval estimates.
            lower, upper = aggregate_interval_estimates[algorithm][:, 0]
            ax2.barh(
                y=alg_idx,
                width=upper - lower,
                height=h,
                left=lower,
                color=ATARI_100K_COLOR_DICT[algorithm],
                alpha=0.75,
                label=algorithm)
            # Plot point estimates.
            ax2.vlines(
                x=aggregate_scores[algorithm],
                ymin=alg_idx - (7.5 * h / 16),
                ymax=alg_idx + (6 * h / 16),
                label=algorithm,
                color='k',
                alpha=0.5)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.set_yticks([])
        ax2.tick_params(axis='both', which='major')
        ax2.set_title("IQM (↑)", fontsize='xx-large')
        ax2.set_xlabel("Human Normalized Score", fontsize="x-large")
        _decorate_axis(ax2, ticklabelsize='xx-large', wrect=5)
        ax2.spines['left'].set_visible(False)
        ax2.grid(True, axis='x', alpha=0.25)

        fig.subplots_adjust(wspace=0.25, hspace=0.45)
        save_fig(fig, 'ablations_summary')

    # Print LaTeX Table:
    if plot_latex_table:
        first_row = ["Game", "Random", "Human"] + algorithms
        rows = [first_row]

        # Raw scores
        for game in ABLATION_SUBSET:
            raw_scores = [RANDOM_SCORES[game], HUMAN_SCORES[game]]
            raw_scores.extend([np.mean(
                score_data_dict_games[algo][game] * (HUMAN_SCORES[game] - RANDOM_SCORES[game]) + RANDOM_SCORES[
                    game]) for
                algo in aggregate_scores.keys()])
            idx_max_all = 2 + np.argmax(raw_scores[2:])
            raw_scores = [f"{x:.1f}" for x in raw_scores]
            raw_scores[idx_max_all] = f"\\textbf{{{raw_scores[idx_max_all]}}}"
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
            score_dict = score_data_dict_games[algo]
            sh = np.sum(np.mean(all_score_dict[algo], axis=0) >= 1)
            col = [sh, *aggregate_scores[algo]]
            cols.append(col)

        rows_ = np.array(cols).T
        for i, row in enumerate(rows_):
            idx_best_all = 2 + (np.argmin(row[2:]) if i == len(rows_) - 1 else np.argmax(row[2:]))
            row = [f"{x:.{0 if i == 0 else 3}f}" if not math.isinf(x) else 'N/A' for x in row]
            row[idx_best_all] = f"\\textbf{{{row[idx_best_all]}}}"
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


def plot_ablations_run_times_results(short: bool = False):
    components = ['World Model\nTraining (↓)', 'Imagination (↓)', 'Combined (↓)']
    our_times = [39, 20]
    ablation_pop_times = [25.8, 92]
    ablation_ac_times = [39, 32.7]
    all_times = {
        f"{OUR_ALG_NAME}": our_times,
        AblationType.NoPOP.value: ablation_pop_times,
        AblationType.LSAC.value: ablation_ac_times
    }

    for times in all_times.values():
        times.append(np.sum(times))

    height = 5 if not short else 3
    fig, ax = plt.subplots(figsize=(8, height))

    bar_width = 0.75
    n_bars = len(all_times)
    space = 2
    xs = np.arange(len(our_times)) * (n_bars + space)

    colors = sns.color_palette('colorblind')
    alg_colors = {
        f"{OUR_ALG_NAME}": colors[1],
        AblationType.NoPOP.value: colors[0],
        AblationType.LSAC.value: colors[4]
    }
    all_bars = []
    for i, (alg_name, times) in enumerate(all_times.items()):
        shift = i - 0.5*(n_bars - 1)
        bars = ax.bar(xs + shift, times, bar_width, color=alg_colors[alg_name], label=alg_name)
        all_bars.append(bars)

    ax.set_ylabel('Epoch Time (sec)', fontsize=16)
    ax.set_xticks(xs, components, fontsize=16)

    ax.legend(loc='best', fontsize=16)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=0.2)

    suffix = '' if not short else '_short'
    save_fig(fig, f'ablations_times_comparison{suffix}')


@dataclasses.dataclass
class WandBRunWMLossData:
    train_obs_loss: list
    eval_obs_loss: list


def _plot_mean_and_std(seeds_dicts, ax, color, label):
    line_style = {
        'train': '-',
        'eval': '--'
    }

    # Filter out eval steps that are not a multiple of 50...:
    for seed_dict in seeds_dicts:
        seed_dict['eval'] = [(v[0], v[1]) for v in seed_dict['eval'] if v[0] % 50 == 0]

    for split in ['train', 'eval']:
        split_values = np.array([
            [step[1] for step in seed_dict[split]]
            for seed_dict in seeds_dicts
        ])
        split_x = np.array([
            step[0] for step in seeds_dicts[0][split]
        ])
        split_mean = split_values.mean(axis=0)
        split_std = split_values.std(axis=0)

        ax.plot(split_x, split_mean, linestyle=line_style[split], color=color, label=f"{label} ({split})")
        ax.fill_between(split_x, split_mean - split_std, split_mean + split_std, color=color, alpha=0.1)


def plot_ablations_wm_loss():
    with open('ablations_wm_loss_data.json', 'r') as f:
        data = json.load(f)

    colors = sns.color_palette('colorblind')
    clrs = {
        "main-result": colors[1],
        "ablation-no-pop": colors[0],
        "ablation-separate-wm-emb": colors[2],
    }
    labels = {
        "main-result": f"{OUR_ALG_NAME}",
        "ablation-no-pop": AblationType.NoPOP.value,
        "ablation-separate-wm-emb": AblationType.WMEmb.value,
    }

    for game_idx in range(8):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        game = f"{ABLATION_SUBSET[game_idx]}NoFrameskip-v4"
        print(f"Plotting for {game}...")
        ax.grid()
        for alg, color in clrs.items():
            _plot_mean_and_std(data[alg][game], ax, color, labels[alg])
        ax.legend()
        ax.set_ylabel('World Model Obs. Loss')
        ax.set_xlabel('Epoch')
        ax.set_title(f"{game}")

        save_fig(fig, f"ablations_wm_obs_loss-{ABLATION_SUBSET[game_idx]}")
        plt.close()


if __name__ == '__main__':
    # import_ablations_wm_loss()
    # plot_ablations_wm_loss()
    plot_ablations_run_times_results(True)
    # main(False, False, False, True, False)
