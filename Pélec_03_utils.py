import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from yellowbrick.regressor import PredictionError, ResidualsPlot
from yellowbrick.model_selection import LearningCurve, ValidationCurve

dpi=100

def adj_r2(estimator, X, y_true):
    n, p = X.shape
    pred = estimator.predict(X)
    return 1 - ((1 - r2_score(y_true, pred)) * (n - 1))/(n-p-1)


def plot_contingency_table(data, col1, col2, title="Contingency Table", set_aspect=True, figsize=(20, 10), savefig=None):
    """
    Description: plot contingency table between 2 columns of a dataframe
    
    Args:
        - data (pd.DataFrame): dataframe
        - col2,col2 (str): names of the two columns to plot
        - title (str): plot title
        - set_aspect (bool): aspect (height/width), if True (default), set to width=height
        - figsize : size of the figure (default (20,10))
        - savefig : name of the saved plot
        
    Return :
        - Contingency table of the two variables (no annotation)
    """
    # Create contingency table
    cont_table = pd.crosstab(index=data.loc[:,col1], columns=data.loc[:,col2], margins=False, normalize='index')
    
    # Plot table
    fig = plt.figure(num=None, figsize=figsize,facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    if set_aspect:
        ax.set_aspect(1)
    sns.heatmap(cont_table, cmap="rocket_r", vmin=0.0, vmax=cont_table.max().max())
    plt.title(title,fontsize=20)
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=dpi)
    
    plt.show()    
    
    
def plot_numerical_var(data, plot='box', log=False, savefig=None) :
    """
    Description: plot distribution of all numerical variables of a dataframe
    
    Args:
        - data (pd.DataFrame): dataframe
        - plot (str) : type of plot; Can be 'box' for boxplot(default) or 'hist' for histograms
        - log (bool): precise if the variable have to be log_transformed (default: False)
        - savefig : name of the saved plot
        
    Return :
        - Distribution plots (histograms or boxplots) of all numerical variables of the dataframe
    """

    numerical_var = data.select_dtypes(include=['float64', 'int64'])
    data_for_facetgrid = pd.melt(numerical_var)
    with sns.plotting_context("notebook", font_scale=1):
        g = sns.FacetGrid(data_for_facetgrid, col='variable', col_wrap=4, margin_titles=True, sharey=False, sharex=False)
        if log == True:
            data_for_facetgrid.loc[:,'value_log'] = np.log1p(data_for_facetgrid.loc[:,'value'])
            if plot == 'box':
                g.map(sns.boxplot, 'value_log')
            elif plot =='hist':
                g.map(sns.histplot, 'value_log')
            else:
                 raise ErrorValue(f"plot should be 'box' (for boxplot) or 'hist' (for histogramm). You provided {plot}!")
        else:
            if plot == 'box':
                g.map(sns.boxplot, 'value')
            elif plot =='hist':
                g.map(sns.histplot, 'value')
            else:
                raise ErrorValue(f"plot should be 'box' (for boxplot) or 'hist' (for histogramm). You provided {plot}!")
                
        if savefig:
            plt.savefig(savefig, bbox_inches='tight', dpi=dpi)

            
            
def plot_correlation_matrix(corr_data, annot=True, savefig=None):
    """
    Description: plot correlation matrix
    
    Args:
        - corr_data (pd.DataFrame): correlation matrix
        - annot (bool) : whether to annotate (default) or not the plot with correlation values
        - savefig : name of the saved plot
        
    Return :
        - Visualisation of the correlation matrix. Can be annotated with correlation values.
    """

    f, ax = plt.subplots(figsize=(10, 10))

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr_data)
    mask[np.triu_indices_from(mask)] = True

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_data, mask=mask, vmax=1, center=1, annot=annot, 
                square=True, linewidths=.1, cbar_kws={"shrink": .5})
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=dpi)

    plt.show()
    
    
def plot_variable_on_Seattle_map(data, var, title=None, savefig=None):
    """
    Description: plot variable on Seattle map with colors according to variable classes
    
    Args:
        - data (pd.DataFrame): dataframe
        - var (str) : name of the variable to plot (should be classes)
        - title (str): plot title
        - savefig : name of the saved plot

        
    Return :
        - Visualisation of the values (classes) of the variable on Seattle map
    """
    
    seattle_long = -122.3300624
    seattle_lat = 47.6038321
    fig = px.scatter_mapbox(data.loc[data.loc[:, var].dropna().index], lat='Latitude', lon='Longitude', color=var, 
                            title=title, mapbox_style='open-street-map', 
                            size_max=15,  width=1200, height=800, zoom=10, 
                            center={'lat':seattle_lat, 'lon':seattle_long})
    fig.update_geos({'resolution':50, 'showsubunits':True, 'fitbounds':'locations', 'showsubunits':True})
    fig.show(config={'scrollZoom':True})
    
    if savefig:
        fig.write_image(savefig, bbox_inches='tight', dpi=dpi)

        
def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """
    Description: display circle of variable correlation on PCA plot
    
    Args:
        - pcs : componants of a PCA (i.e; pca.components_)
        - n_comp (int) : number of componants
        - pca (str): result of a PCA
        - axis_ranks : list of axis to plot (ex: [(0,1), (0,2)])
        - labels (str): labels to be written on the plot (default: None)
        - label_rotation (float): rotation angle of labels on the plot (default: 0)
        - lims: limit of the axis (default: None)

        
    Return :
        - Visualisation of the correlation plots of PCA, on several axis specified in axis_ranks
    """

    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

            
def plot_model_comparison(summary_table, model_name_var='Nom_modele', score_list=['r2', 'adj_r2'], savefig=None):
    """
    Description: plot barplot of metric scores for model comparison
    
    Args:
        - summary_table (dataframe): dataframe of scores for models
        - model_name_var (str) :name of the dataframe column containing model identification
        - score_list (list of str) : list of scores to plot (default: ['r2', 'adj_r2', 'explained_var'])
        - savefig (str) : name of the saved plot
        
    Return :
        - barplots of model comparison (one barplot per score)
    """
    
    # Organisation des données pour les plots
    resume_for_plot = pd.melt(summary_table, id_vars =[model_name_var])
    
    # Creation d'une colonne 'set' (précisant train ou cross_validation set)
    resume_for_plot.loc[resume_for_plot.loc[:, 'variable'].str.contains('train'), 'set'] = 'train'
    resume_for_plot.loc[resume_for_plot.loc[:, 'variable'].str.contains('test'), 'set'] = 'Cross-validation test'
    
    # Renommage des variables
    resume_for_plot = resume_for_plot.replace({'r2_train':'r2', 'r2_test':'r2',
                         'adj_r2_train':'adj_r2', 'adj_r2_test':'adj_r2',
                         'msle_train':'msle', 'msle_test':'msle',
                         'explained_var_train':'explained_var', 'explained_var_test':'explained_var',
                         'medAbsError_train':'medAbsError', 'medAbsError_test':'medAbsError', 
                         'mse_train':'mse', 'mse_test':'mse', 
                         'rmse_train':'rmse', 'rmse_test':'rmse'})
    
    # Liste des variables à plotter
    resume_for_plot = resume_for_plot.loc[resume_for_plot.loc[:, 'variable'].isin(score_list)]
    
    # Plot
    plot = sns.catplot(x="value", y=model_name_var, sharey=True, sharex=False, 
                hue="set", col="variable", palette='jet_r',
                data=resume_for_plot, kind="bar", col_wrap=4,
                height=4, aspect=1)
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=dpi)

        
        
def plot_validation_curves(model, param_list, param_list_ranges, X, y, cv=5, scoring='r2', nrows=1, sharey=True):
    """
    Description: plot validation curves following hyperparameter tuning
    
    Args:
        - model (dataframe): model to use
        - param_list (list of str): list of parameters for which plot validation curves
        - param_list_ranges (list of str): list of range associated to each parameter defined in param_list
        - X (dataframe): explanotory variable set
        - y (array): target variable set
        - cv (int):number of cross validation folds (default: 5)
        - scoring (str):scoring to use for validation curve (default : r²)
        - nrows (int): number of rows to use for ploting (default: 1 (i.e. all plot on the same row))
        - sharey (bool): graphical parameter to specify to share the y-axis with other (default: True)
        
    Return :
        - plot validation curves (from Yellowbrick library) of the model for each hyperparameter defined in param_list.
    """

    n_plot = len(param_list)
    ncols = int(np.ceil(n_plot/nrows))
    figsize = (7*ncols, 6*nrows)   
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)

    for param, param_range, ax in zip(param_list, param_list_ranges, axes.flat):
        viz_estim = ValidationCurve(
            model, param_name=param, 
            param_range=param_range, cv=cv, scoring=scoring, ax=ax)
        viz_estim.fit(X, y)
        viz_estim.finalize()
        
        
        
def plot_feature_importance(model, n_var=10, figsize=(10, 13), savefig=None):
    """
    Description: plot feature importance for model
    
    Args:
        - model (dataframe): model to use
        - n_var (int): number of the first features to plot (default:10)
        - figsize : size of the figure (default (10, 13))
        - savefig (str): name of the saved plot
        
    Return :
        - barplot showing the n_var most important features of the model
    """

    # Extraction de l'mportances des variables
    feature_imp = model.regressor_.named_steps['model'].feature_importances_
    
    # Noms des variables
    in_features = model.feature_names_in_
    features_out = model.regressor_[:-1].get_feature_names_out(input_features=in_features)

    # Renommage
    feature_names = []
    for feature in features_out:
        feature_names.append(feature.split('__')[1])
        
    # Plot des variables les plus importantes
    plt.figure(figsize=figsize)
    feature_imp_for_graph = pd.Series(feature_imp,index=feature_names).sort_values(ascending=False)
    sns.barplot(x=feature_imp_for_graph[:n_var], y=feature_imp_for_graph[:n_var].index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    
    if savefig: 
        plt.savefig(savefig, bbox_inches='tight', dpi=dpi)
        
    plt.show()
    

def plot_prediction_errors(model, X_train, y_train, X_test, y_test, y_test_pred, color='lightblue', title=None, savefig=None):
    """
    Description: plot prediction errors of the model, on log-transformed and raw data
    
    Args:
        - model (dataframe): model to use
        - X_train, X_test (dataframes) : training and validation set of explanatory variables
        - y_train, y_test (arrays) : arrays of training and validation set of target values
        - y_test_pred (array): array of predicted values (must be the same size as y_test)
        - color (str):colors of the markers for log-transformed data  (def%pprintlt :'lightblue')
        - title (str) : plot main title
        - savefig : name of the saved plot
        
    Return :
        - plot prediction error plot on raw and log-transformed data of the test set.
    """    
    # Initialisation de la figure
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    
    # Valeurs brutes
    visualizer = PredictionError(model,  ax=ax[0])
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.finalize()

    # Valeurs log_transformed
    sns.scatterplot(x=y_test, y=y_test_pred, color=color, ax=ax[1])
    plt.yscale('log')
    plt.xscale('log')

    p1 = max(max(y_test_pred), max(y_test))
    p2 = min(min(y_test_pred), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'r-')
    plt.annotate(text=str(f'R² ajusté: {adj_r2(model, X_test, y_test):.3f}'), xy=(0.05,0.90), xycoords='axes fraction')
    plt.annotate(text=str(f'R² : {r2_score(y_test_pred, y_test):.3f}'), xy=(0.05,0.83), xycoords='axes fraction')
    plt.xlabel('Valeurs observées', fontsize=15)
    plt.ylabel('Prédictions', fontsize=15)
    plt.axis('equal')
    
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=dpi)
    
    plt.show()
       

def plot_predictions(model, X_train, y_train, X_test, y_test, y_test_pred, hue=None, color='lightblue', title=None, savefig=None):
    """
    Description: plot prediction errors of the model, on log-transformed and raw data by grouping variable
    
    Args:
        - model (dataframe): model to use
        - X_train, X_test (dataframes) : training and validation set of explanatory variables
        - y_train, y_test (arrays) : arrays of training and validation set of target values
        - y_test_pred (array): array of predicted values (must be the same size as y_test)
        - hue (str); name of the grouping variable (must be a column name of X_train and X_test)
        - color (str):colors of the markers for log-transformed data  (default :'lightblue')
        - title (str) : plot main title
        - savefig : name of the saved plot
        
    Return :
        - plot prediction error plot on raw and log-transformed data of the test set, by levels of grouping variable.
    """    
    
    if hue:
        for label in X_train.loc[:,hue].unique():
            X_train_sub = X_train.loc[X_train.loc[:, hue]==label]
            idx = X_train_sub.index.tolist()
            y_train_sub = y_train.loc[y_train.index.isin(idx)]
     
            X_test_sub = X_test.loc[X_test.loc[:, hue]==label]
            idx_test = X_test_sub.index.tolist()
            y_test_sub = y_test.loc[y_test.index.isin(idx_test)]
    
            y_test_pred_sub = model.predict(X_test_sub)
    
            plot_prediction_errors(model, X_train_sub, y_train_sub, X_test_sub, y_test_sub, y_test_pred_sub, color=color, title=label, savefig=savefig)

    else:
        plot_prediction_errors(model, X_train, y_train, X_test, y_test, y_test_pred, color=color, title=title, savefig=savefig)  
        
        
def plot_feature_permutation_importance(model, X, y, scoring='r2', color="lightblue", n_repeats=100, random_state=0, savefig=None): 
    """
    Description: plot feature importance based on permutation importance
    
    Args:
        - model: model to use
        - X (dataframes) : set of explanatory variables
        - y(arrays) : arrays of target values
        - scoring (str): Scorer to use (default: 'r2')
        - color (str): colors of the bars  (default :'lightblue')
        - n_repeats (int) : mumber of times to permute a feature (default=100)
        - random_state (int): pseudo-random number generator to control the permutations of each feature. Pass an int to get reproducible results across function calls (default:0)
        - savefig : name of the saved plot
        
    Return :
        - plot prediction error barplot on train and test sets, by levels of grouping variable.
    """    
    # Compute permutation importance
    result = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state)    
    
    # Format results
    features_names = X.columns
    model_importances = pd.Series(result.importances_mean, index=features_names)
    sorted_idx = result.importances_mean.argsort()
   
    # Plot
    fig, ax = plt.subplots(figsize=(10,10))
    y_ticks = np.arange(0, len(features_names))
    ax.barh(y_ticks, model_importances[sorted_idx], xerr=result.importances_std[sorted_idx], color=color)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(features_names[sorted_idx])
    ax.set_title("Feature importances using permutation on full model")
    ax.set_xlabel("Mean accuracy decrease")
    ax.set_xlim(0,1)
    fig.tight_layout()
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=dpi)
 
    plt.show()

    
    
    
def barplot_prediction_Errors(model, X_train, y_train, X_test, y_test, hue, metric='adj_r2', color='lightblue', title=None, savefig=None):
    """
    Description: plot prediction errors of the model, on log-transformed and raw data by grouping variable
    
    Args:
        - model (dataframe): model to use
        - X_train, X_test (dataframes) : training and validation set of explanatory variables
        - y_train, y_test (arrays) : arrays of training and validation set of target values
        - y_test_pred (array): array of predicted values (must be the same size as y_test)
        - hue (str); name of the grouping variable (must be a column name of X_train and X_test)
        - metric (str): metric to plot; Can be 'adj_r2' (default) or 'r2'
        - color (str):colors of the markers for log-transformed data  (default :'lightblue')
        - title (str) : plot main title
        - savefig : name of the saved plot
        
    Return :
        - plot prediction error barplot on train and test sets, by levels of grouping variable.
    """    
    
    metric_train = str(metric + '_train')
    metric_test = str(metric + '_test')
    dtypes = np.dtype([("Type", str), (metric_train, float),(metric_test, float)])
    result = pd.DataFrame(np.empty(0, dtype=dtypes))    

            
    for label in X_train.loc[:,hue].unique():
        X_train_sub = X_train.loc[X_train.loc[:, hue]==label]
        idx = X_train_sub.index.tolist()
        y_train_sub = y_train.loc[y_train.index.isin(idx)]
     
        X_test_sub = X_test.loc[X_test.loc[:, hue]==label]
        idx_test = X_test_sub.index.tolist()
        y_test_sub = y_test.loc[y_test.index.isin(idx_test)]
    
        if metric == 'adj_r2':
            r2_test = adj_r2(model, X_test_sub, y_test_sub)
            r2_train = adj_r2(model, X_train_sub, y_train_sub)
        elif metric == 'r2':
            r2_test = model.score(X_test_sub, y_test_sub)
            r2_train = model.score(X_train_sub, y_train_sub)
        else: 
            raise ErrorValue(f"metric should be 'r2' (default) or 'adj_r2'. You provided {metric}!")
        
        result = result.append({'Type':label, metric_train:r2_train,  metric_test:r2_test}, ignore_index=True)
    
    plot_model_comparison(result, model_name_var='Type', score_list=[metric], savefig=savefig)
    

def evaluate_model(model, X_train, y_train, X_test, y_test, n_var=10, hue='PrimaryPropertyType_reduced', color='lightblue',  model_name_for_figsave="undefined"):
    """
    Description: draw and save plots for model evaluation
    
    Args:
        - model (dataframe): model to use
        - X_train, X_test (dataframes) : training and validation set of explanatory variables
        - y_train, y_test (arrays) : arrays of training and validation set of target values
        - n_var (int): number of the first features to plot (default:10)
        - hue (str); name of the grouping variable (must be a column name of X_train and X_test)
        - color (str):colors of the markers for log-transformed data  (default :'lightblue')
        - model_name_for_figsave : name of the model that will be used for all saved plots (default:'undefined')
        
    Return :
        - plot features importance, prediction error plots for all dataset and potentially by grouping variable, and residuals plot.
    """    
      
    # Importance des variables
    print("IMPORTANCE DES VARIABLES")
    plot_feature_importance(model, n_var, figsize=(10, 8), savefig=str('figures/' + model_name_for_figsave + '_feature_importance.png'))
    plot_feature_permutation_importance(model, X_train, y_train, scoring='r2', color="lightblue", n_repeats=100, random_state=0, 
                                        savefig=str('figures/' + model_name_for_figsave + '_featurePermuation_TrainSet.png'))
    plot_feature_permutation_importance(model, X_test, y_test, scoring='r2', color="lightblue", n_repeats=100, random_state=0, 
                                        savefig=str('figures/' + model_name_for_figsave + '_featurePermuation_TestSet.png'))
    
    # Prédictions
    print("PREDICTIONS")
    y_test_predmodel = model.predict(X_test)
    plot_predictions(model, X_train, y_train, X_test, y_test, y_test_predmodel, color=color, savefig=str('figures/' + model_name_for_figsave + '_predictions.png'))
    
    # Predictions par type de batiment
    if hue:
        print("PREDICTIONS PAR TYPE DE BATIMENT")
        barplot_prediction_Errors(model, X_train, y_train, X_test, y_test, hue=hue, metric='adj_r2', 
                                  color='lightblue', savefig=str('figures/' + model_name_for_figsave + '_predictions_By' + hue + '_Barplot_R2adj.png'))
        barplot_prediction_Errors(model, X_train, y_train, X_test, y_test, hue=hue, metric='r2', 
                                  color='lightblue', savefig=str('figures/' + model_name_for_figsave + '_predictions_By' + hue + '_Barplot_R2.png'))
        print("PREDICTIONS PAR TYPE DE BATIMENT-DETAILS")
        plot_predictions(model, X_train, y_train, X_test, y_test, y_test_predmodel,hue=hue, 
                         color=color, savefig=str('figures/' + model_name_for_figsave + '_predictions_By' + hue + '.png'))    
    
    # Residus
    print("DISTRIBUTION DES RESIDUS")
    fig = plt.figure(figsize=(10,5))
    visualizer = ResidualsPlot(model, hist=False, qqplot=True, is_fitted=True)
    visualizer.fit(X_train, y_train)  
    visualizer.score(X_test, y_test)
    visualizer.show() 