import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

GRID_SCORE_POS = ['rank_test_score', 'std_test_score', 'delta_train_test', 'mean_score_time']


def grid_resultification(grid, light=False):
    res_lourd = pd.DataFrame(grid.cv_results_)
    if light:
        # On enleve les resultats des splits
        keepcols = [col for col in res_lourd.columns if "split" not in col]
        res_light = res_lourd[keepcols]
        res = res_light
    else:
        res = res_lourd
    return res


def get_split_scores_for_model(res_grid, nb_cv, by=GRID_SCORE_POS):
    try:
        grid = res_grid
        if 'delta_train_test' in by:
            grid['delta_train_test'] = np.abs(grid['mean_train_score'] - grid['mean_test_score'])
        df_to_plot = pd.DataFrame()
        lst_models = []
        # Test if possible
        for score in by:
            if score not in GRID_SCORE_POS:
                print(f"{score} not in {GRID_SCORE_POS}. We're popping it.")
                by = [elem for elem in by if elem != score]
        for score in by:
            # Get best model
            best_model = grid.sort_values(by=score, ascending=True).head(1)
            best_params = best_model['params'].values[0]
            wip_row = grid[grid['params'] == best_params]
            lst_models.append([score, best_params, wip_row['mean_test_score'].values[0], wip_row[score].values[0]])
            # Get its split values
            df_to_plot[f'TR_best_{score}'] = [wip_row[f'split{i}_train_score'].values[0] for i in range(nb_cv)]
            df_to_plot[f'TE_best_{score}'] = [wip_row[f'split{i}_test_score'].values[0] for i in range(nb_cv)]
        df_models = pd.DataFrame(lst_models, columns=['metrics', 'best_params', 'mean_test_score', 'metric_value'])
    except KeyError:
        print(f"The score you are looking ({by}) for is not in the grid.\n Check grid_resultification output.")
    return df_to_plot, df_models


def plot_split_scores(df_to_plot, df_model, save=False, path='./'):
    by = df_model['metrics'].values.tolist()
    cnt = 1
    if save:
        list_img = []
        path_to_models_csv = ""
    else:
        list_img = None
        path_to_models_csv = None
    for score in by:
        plt.figure(figsize=(10, 6))
        plt.plot([i for i in range(df_to_plot.shape[0])], df_to_plot[f'TR_best_{score}'], label='Training')
        plt.plot([i for i in range(df_to_plot.shape[0])], df_to_plot[f'TE_best_{score}'], label='Testing')
        if df_model is None:
            plt.title(f'Best {score}')
        else:
            plt.title(f"Best {score} \n {df_model[df_model['metrics'] == score]['best_params'].values[0]}")
        plt.xlabel("splits")
        plt.ylabel("grid_score_metric")
        plt.ylim(0, 1)
        plt.xticks([i for i in range(df_to_plot.shape[0])])
        plt.legend()
        if save:
            list_img.append(path + f"{score}.png")
            plt.savefig(path + f"{score}.png")
        cnt += 1
        plt.show()

    sns.barplot(x=df_model['metrics'], y=df_model['mean_test_score'])
    plt.title("Test score moyen des differents meilleurs modeles \n")
    plt.ylim(0, 1)
    if save:
        df_model.to_csv(path + "grid_results.csv")
        plt.savefig(path + "grid_results.png")
        path_to_models_csv = path + "grid_results.csv"
    plt.show()
    return list_img, path_to_models_csv
