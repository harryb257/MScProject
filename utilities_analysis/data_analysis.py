import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


y = 'mcc'

# Import datasets
df = pd.read_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ESM2/merged_results/test_predictions.csv')
pfeature = pd.read_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_Pfeature/output/merged_results/test_predictions.csv')
prott5 = pd.read_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ProtT5/merged_results/test_predictions.csv')

df['model'] = 'ESM-2'

pfeature['model'] = 'Pfeature'
pfeature['mode'] = 'Pfeature'
pfeature['model size'] = 'Pfeature'

prott5['model'] = 'ProtT5-XL'
prott5['model size'] = '1.2B'

df['plot_label'] = df['model'] + ' ' + df['mode']

pfeature['plot_label'] = 'Pfeature'

prott5['plot_label'] = prott5['model'] + ' ' + prott5['mode']

df_prott5 = pd.concat([df, prott5], axis=0)

plot_order = [
    'ESM-2 base',
    'ESM-2 LoRA',
    'ESM-2 full',
    'ProtT5-XL base',
    'ProtT5-XL LoRA',
    'ProtT5-XL full',
    'Pfeature'
]

model_size_order = [
    '8M',
    '35M',
    '150M',
    '650M'
]

mode_order = [
    'base',
    'LoRA',
    'full'
]

df['plot_label'] = pd.Categorical(
    df['plot_label'],
    categories=plot_order,
    ordered=True
)

prott5['plot_label'] = pd.Categorical(
    prott5['plot_label'],
    categories=plot_order,
    ordered=True
)

pfeature['plot_label'] = pd.Categorical(
    pfeature['plot_label'],
    categories=plot_order,
    ordered=True
)


#---------------------------------- Calculations -----------------------------------------------


# --Blocking pathogen--
model = ols("mcc ~ C(mode) + C(Q('model size')) + C(pathogen)", data=df_prott5).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print('ESM-2 and ProtT5-XL, blocking Pathogen', anova_table)
anova_table.to_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ESM2_weighted_clf/merged_results/anova_blocking_pathogen.csv', float_format='%.3f')



# -- Pairwise comparisons between modes --
results = []
for mode1, mode2 in [
    ('base', 'LoRA'),
    ('base', 'full'),
    ('LoRA', 'full')
]:

    pairwise_data = df_prott5[df_prott5['mode'].isin([mode1, mode2])]
    model = ols("mcc ~ C(mode) + C(Q('model size')) + C(pathogen)",data=pairwise_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    results.append({'comparison': f'{mode1} vs {mode2}',
        'F': anova_table.loc['C(mode)', 'F'],
        'p': anova_table.loc['C(mode)', 'PR(>F)']
    })

pairwise_results_mode = pd.DataFrame(results)
print('mode level pairwise anova', pairwise_results_mode)

# -- HS correction across the three pairwise comparisons --
reject, p_hs, _, _ = multipletests(
    pairwise_results_mode['p'],
    alpha=0.05,
    method='hs'
)

pairwise_results_mode_hs = pd.DataFrame({
'mode_p_hs': p_hs,
'mode_reject': reject})

pairwise_results_mode_hs = pd.concat([pairwise_results_mode,pairwise_results_mode_hs[['mode_p_hs', 'mode_reject']]], axis=1)

pairwise_results_mode.to_csv(
    '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ESM2_weighted_clf/merged_results/overall_mode_pairwise_blocking_pathogen_hs.csv',
    index=False,
    float_format='%.3f'
)

print('Pairwise mode comparisons, blocking for pathogen with hs correction', pairwise_results_mode_hs)






# --Within pathogen effect of mode and model size--
results = []

for pathogen, pathogen_data in df_prott5.groupby('pathogen'):

    model = ols("mcc ~ C(mode) + C(Q('model size'))", data=pathogen_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    results.append({
        'pathogen': pathogen,
        'mode_F': anova_table.loc['C(mode)', 'F'],
        'mode_p': anova_table.loc['C(mode)', 'PR(>F)'],
        'model_size_F': anova_table.loc["C(Q('model size'))", 'F'],
        'model_size_p': anova_table.loc["C(Q('model size'))", 'PR(>F)'],
    })

within_pathogen = pd.DataFrame(results)

# --- False discovery rate across pathogen specific-mode and model size tests
# is it controlled for expected proportion of false discoveries
# p-values for C(mode)

mode_pvals = within_pathogen['mode_p']

# p-values for C(model_size)
size_pvals = within_pathogen['model_size_p']

# Generate FDR corrected p-values
mode_reject, mode_p_fdr, _, _ = multipletests(
    mode_pvals,
    alpha=0.05,
    method='fdr_bh'
)
size_reject, size_p_fdr, _, _ = multipletests(
    size_pvals,
    alpha=0.05,
    method='fdr_bh'
)

# Create dataframe of FDR corrected p values
fdr_df = pd.DataFrame({'mode_reject': mode_reject,
'mode_p_fdr': mode_p_fdr,
'size_reject': size_reject,
'size_p_fdr': size_p_fdr})

print(fdr_df)

# Concat to the raw p-values
within_pathogen_with_fdr = pd.concat([within_pathogen, fdr_df], axis=1)

# Reorder columns by mode or model size
within_pathogen_with_fdr = within_pathogen_with_fdr[[
    'pathogen', 'mode_F', 'mode_p', 'mode_p_fdr', 'mode_reject', 'model_size_F', 'model_size_p', 'size_p_fdr',
       'size_reject']]

# Save to csv
within_pathogen_with_fdr.to_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ESM2_weighted_clf/merged_results/within_pathogen_with_FDR.csv', float_format='%.3f')

print('Within pathogen with FDR', within_pathogen_with_fdr)


# -- Perform a pairwise comparison of modes within pathogens for those with FDR corrected significant p-values

# Create series of sig pathogens
pathogens_mode_sig = within_pathogen_with_fdr[within_pathogen_with_fdr['mode_reject'] == True]
pathogens_mode_sig = pathogens_mode_sig['pathogen'].reset_index(drop=True)

# Perform pairwise comparison
pairwise_results = []
for pathogen in pathogens_mode_sig:
    df_p = df_prott5[df_prott5['pathogen'] == pathogen]
    model = smf.ols("mcc ~ C(mode) + C(Q('model size'))", data=df_p).fit()
    pairwise = model.t_test_pairwise('C(mode)')
    pairwise_df = pairwise.result_frame.copy()
    pairwise_df['comparison'] = pairwise_df.index
    pairwise_df['pathogen'] = pathogen

    pairwise_results.append(pairwise_df)

pairwise_results = pd.concat(pairwise_results, ignore_index=True)
pairwise_results.insert(0, 'pathogen', pairwise_results.pop('pathogen'))
pairwise_results.insert(1, 'comparison', pairwise_results.pop('comparison'))
pairwise_results['comparison'] = pairwise_results['comparison'].str.replace('-',' vs ', regex=False)
pairwise_results.to_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/Results_ESM2_weighted_clf/merged_results/mode_pairwise_by_pathogen.csv', float_format='%.3f')

print(pairwise_results)
fig, ax = plt.subplots(1, 1)

hue_order = pairwise_results['comparison'].unique()

sns.barplot(
    data=pairwise_results,
    x='pathogen',
    y='coef',
    hue='comparison',
    hue_order=hue_order,
    order=pairwise_results['pathogen'].unique(),
    errorbar=None,
    ax=ax
)

ax.set_xlabel('Pathogen')
ax.set_ylabel('Difference in MCC (first mode − second mode)')
ax.set_ylim(-0.3, 0.5)
ax.axhline(0, color='black', linewidth=0.5)
ax.tick_params(axis='x', rotation=90)
ax.legend(title='Comparison')


def significance_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    return ''


# Add stars to the correct bars
for comparison, container in zip(hue_order, ax.containers):

    rows = (pairwise_results[pairwise_results['comparison'] == comparison]
        .set_index('pathogen')
        .loc[pairwise_results['pathogen'].unique()]
    )

    for bar, (_, row) in zip(container, rows.iterrows()):

        stars = significance_star(row['pvalue-hs'])

        if not stars:
            continue

        height = bar.get_height()

        # Put stars above positive bars and below negative bars
        if height >= 0:
            y = height + 0.01
            va = 'bottom'
        else:
            y = height - 0.01
            va = 'top'

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            stars,
            ha='center',
            va=va,
            fontsize=13,
            fontweight='bold'
        )

plt.show()





# ---- Pfeature vs pLM analysis

# Add a 'model' column to pfeature so that it can be compared to the pLMs
pfeature['model'] = 'Pfeature'

# Concat pfeature results to df_prott5
df_prott5_pfeature = pd.concat([df_prott5, pfeature], axis=0)

# --Blocking pathogen--
model = ols("mcc ~ C(model) + C(pathogen)", data=df_prott5_pfeature).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print('Pfeature vs ESM-2 vs ProtT5-XL, blocking Pathogen', anova_table)
anova_table.to_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/model_anova.csv',float_format='%.3f')

# --Within pathogen effect of model type (pLM and pfeature) --
results = []
pairwise_results = []

for pathogen, pathogen_data in df_prott5_pfeature.groupby('pathogen'):

    model = ols("mcc ~ C(model)", data=pathogen_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    results.append({
        'pathogen': pathogen,
        'model_F': anova_table.loc['C(model)', 'F'],
        'model_p': anova_table.loc['C(model)', 'PR(>F)'],
    })

    # Pairwise comparisons between model types
    pairwise = model.t_test_pairwise(
        'C(model)',
        method='fdr_bh'
    )

    pairwise_df = pairwise.result_frame.copy()

    # Add comparison as a column
    pairwise_df['comparison'] = pairwise_df.index
    pairwise_df['pathogen'] = pathogen
    pairwise_results.append(pairwise_df)

# Combine pathogen-level ANOVA results
within_pathogen_anova = pd.DataFrame(results)
print('\nWithin pathogen ANOVA', within_pathogen_anova)
within_pathogen_anova.to_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/within_pathogen_anova_model_comp.csv',float_format='%.3f')


# --- FDR: is it controlled for expected proportion of false discoveries
model_pvals = within_pathogen_anova['model_p']

model_reject, model_p_fdr, _, _ = multipletests(
    model_pvals,
    alpha=0.05,
    method='fdr_bh'
)

within_pathogen_anova['model_reject'] = model_reject
within_pathogen_anova['model_p_fdr'] = model_p_fdr

print('\nWithin pathogen ANOVA with FDR')
print(within_pathogen_anova)

within_pathogen_anova.to_csv(
    '/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/within_pathogen_anova_model_comp.csv',
    float_format='%.3f')

# Combine pairwise results
within_pathogen_pairwise = pd.concat(
    pairwise_results,
    ignore_index=True)

print('\nWithin pathogen pairwise comparisons', within_pathogen_pairwise)
within_pathogen_pairwise.to_csv('/Users/harry/Documents/Data Science MSc/PROJECT/MScProject/within_pathogen_pairwise_model_comp.csv',float_format='%.3f')
