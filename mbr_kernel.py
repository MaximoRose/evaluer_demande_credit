import pandas as pd
import outils_feature_engineering_810 as fe810
import numpy as np
import gc
# Eviter d'avoir les warning de caveheat
import warnings

import time
from contextlib import contextmanager

DEFAULT_CATEGORY_MEAN_FREQ = 0.03
DEFAULT_EMPTY_FEAT_TRESHOLD_FOR_ROW = 0.4


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(name, time.time() - t0))


warnings.simplefilter(action='ignore', category=FutureWarning)

list_of_feature_toRem = \
    ['CODE_GENDER',  # I dont want a model that stigmatize. No interest in doing so
     'EXT_SOURCE_2',  # Highly correlated to target, yet we have no info on what it is
     'EXT_SOURCE_3',  # Highly correlated to target, yet we have no info on what it is
     'EXT_SOURCE_1',  # Highly correlated to target, yet we have no info on what it is
     'NAME_TYPE_SUITE',
     # Judgemental : I will not explain to a customer, they cannot have a loan because they came with their child...
     'DAYS_LAST_PHONE_CHANGE',  # I don't see myself explaining it to a client
     'FLAG_DOCUMENT_2',  # Most of the client have given less than 2 documents.
     'FLAG_DOCUMENT_3',  # + How would you explain : "You've been penalized because you gave me such document"
     'FLAG_DOCUMENT_4',
     'FLAG_DOCUMENT_5',
     'FLAG_DOCUMENT_6',
     'FLAG_DOCUMENT_7',
     'FLAG_DOCUMENT_8',
     'FLAG_DOCUMENT_9',
     'FLAG_DOCUMENT_10',
     'FLAG_DOCUMENT_11',
     'FLAG_DOCUMENT_12',
     'FLAG_DOCUMENT_13',
     'FLAG_DOCUMENT_14',
     'FLAG_DOCUMENT_15',
     'FLAG_DOCUMENT_16',
     'FLAG_DOCUMENT_17',
     'FLAG_DOCUMENT_18',
     'FLAG_DOCUMENT_19',
     'FLAG_DOCUMENT_20',
     'FLAG_DOCUMENT_21',
     'FLAG_EMP_PHONE',
     # Impossible de dire a un client : "On vous refuse le pret parce qu'on vous n'avez pas de telephone pro
     'FLAG_WORK_PHONE',
     # Je trouverais ca super pourrave de dire a un client : "je ne vous fait pas de pret, parce que vous vivez actuellement dans un taudis"
     'FLAG_EMAIL',
     'APARTMENTS_AVG',
     'BASEMENTAREA_AVG',
     'YEARS_BEGINEXPLUATATION_AVG',
     'YEARS_BUILD_AVG',
     'COMMONAREA_AVG',
     'ELEVATORS_AVG',
     'ENTRANCES_AVG',
     'FLOORSMAX_AVG',
     'FLOORSMIN_AVG',
     'LANDAREA_AVG',
     'LIVINGAPARTMENTS_AVG',
     'LIVINGAREA_AVG',
     'NONLIVINGAPARTMENTS_AVG',
     'NONLIVINGAREA_AVG',
     'APARTMENTS_MODE',
     'BASEMENTAREA_MODE',
     'YEARS_BEGINEXPLUATATION_MODE',
     'YEARS_BUILD_MODE',
     'COMMONAREA_MODE',
     'ELEVATORS_MODE',
     'ENTRANCES_MODE',
     'FLOORSMAX_MODE',
     'FLOORSMIN_MODE',
     'LANDAREA_MODE',
     'LIVINGAPARTMENTS_MODE',
     'LIVINGAREA_MODE',
     'NONLIVINGAPARTMENTS_MODE',
     'NONLIVINGAREA_MODE',
     'APARTMENTS_MEDI',
     'BASEMENTAREA_MEDI',
     'YEARS_BEGINEXPLUATATION_MEDI',
     'YEARS_BUILD_MEDI',
     'COMMONAREA_MEDI',
     'ELEVATORS_MEDI',
     'ENTRANCES_MEDI',
     'FLOORSMAX_MEDI',
     'FLOORSMIN_MEDI',
     'LANDAREA_MEDI',
     'LIVINGAPARTMENTS_MEDI',
     'LIVINGAREA_MEDI',
     'NONLIVINGAPARTMENTS_MEDI',
     'NONLIVINGAREA_MEDI',
     'FONDKAPREMONT_MODE',
     'HOUSETYPE_MODE',
     'TOTALAREA_MODE',
     'WALLSMATERIAL_MODE',
     'EMERGENCYSTATE_MODE',
     'WEEKDAY_APPR_PROCESS_START',  # Je ne vois pas l'interet
     'REGION_POPULATION_RELATIVE',  # On a une note de region qui est plus interessante
     # 'DAYS_EMPLOYED' # On a fait des ratio a la place. (feat_imp plus grande dans logreg
     'LIVE_REGION_NOT_WORK_REGION',  # Redondant avec LIVE_CITY_NOT_WORK_CITY
     'CNT_FAM_MEMBERS',  # Highly correlated to NUMBER OF CHILDREN, though the latter is more correlated to the target
     'REGION_RATING_CLIENT',
     # # Highly correlated to REGION_RATING_CLIENT_W_CITY, though the latter is more correlated to the target
     'LIVE_CITY_NOT_WORK_CITY',
     'REG_CITY_NOT_WORK_CITY',
     'DEF_30_CNT_SOCIAL_CIRCLE',  # CNT_SOCIAL_CIRCLE ce sont des cas particuliers
     'OBS_30_CNT_SOCIAL_CIRCLE',
     'DEF_60_CNT_SOCIAL_CIRCLE',
     'OBS_60_CNT_SOCIAL_CIRCLE',
     'NAME_CONTRACT_TYPE',  # Creer des colonnes additionnelles tres correlees
     'ORGANIZATION_TYPE',
     # Tres correle a NAME_INCOME_TYPE_Pensioner et difficile d'expliquer a quelqu'un, on ne vous donne pas de pret, parce que vous travaillez la
     'AMT_GOODS_PRICE', # On a le montant du credit, c'est peut etre pas le peine de prendre aussi le montant du bien. (tres correles)
     'NAME_INCOME_TYPE', # Les categories principales donnent peu d'information
     'NAME_HOUSING_TYPE', # Pas utile de juger la personne en fonction de ou elle vit
     'OCCUPATION_TYPE', # On ne va pas penaliser quelqu'un pour son job
     'FLAG_MOBIL', # Ne devrait pas entrer en ligne de compte
     'AMT_REQ_CREDIT_BUREAU_HOUR', # information useless : n'a pas a voir avec la solvabilite du client
     'AMT_REQ_CREDIT_BUREAU_DAY',
     'AMT_REQ_CREDIT_BUREAU_WEEK',
     'AMT_REQ_CREDIT_BUREAU_MON',
     'AMT_REQ_CREDIT_BUREAU_QRT',
     'AMT_REQ_CREDIT_BUREAU_YEAR',
     'NAME_FAMILY_STATUS', # Ne devrait pas entrer en ligne de compte (vie perso, n'a pas a voir avec la solvabilite de la personne)
     'FLAG_CONT_MOBILE', # Pas de lien direct avec le probleme. Seulement des theories fumeuses (faible importance de la feature en rf)
     'REG_CITY_NOT_LIVE_CITY', # Ce genre de feature c'est bien pour detecter de la fraude, mais c'est autre chose (faible importance rf)
     'REG_REGION_NOT_LIVE_REGION', # N'apporte pas directement au probleme - peu correlee
     'REG_REGION_NOT_WORK_REGION',
     'CNT_CHILDREN',
     'FLAG_PHONE', # Je veux bien accorder un credit aux marginaux sans telephone
     'OWN_CAR_AGE',
     'DAYS_REGISTRATION' # PEU IMPORTE QU'ON AIT COMMENCE NOTRE JOB IL Y PEU
     ]


def col_categorisation(df, col, liste_ordonee):
    newcol = col + "_CATed"
    dfres = df
    for name in liste_ordonee:
        dfres.loc[dfres[col] == name, newcol] = liste_ordonee.index(name)
    dfres.drop(columns=[col], inplace=True)
    return dfres


def count_flag_documents(df):
    flag_columns = [col for col in df.columns if col.startswith('FLAG_DOCUMENT')]
    df['flag_count'] = df[flag_columns].sum(axis=1)
    return df


def get_age_label(days_birth):
    """ Return the age group label (int). """
    age_years = -days_birth / 365
    if age_years < 27:
        return 1
    elif age_years < 40:
        return 2
    elif age_years < 50:
        return 3
    elif age_years < 65:
        return 4
    elif age_years < 99:
        return 5
    else:
        return 0


def process_application(df, encoding_treshold=DEFAULT_CATEGORY_MEAN_FREQ,
                        nan_treshold=DEFAULT_EMPTY_FEAT_TRESHOLD_FOR_ROW):
    df = fe810.remove_too_nany_observations(df, treshold=nan_treshold)
    # Valeur aberrante
    df.loc[:, 'DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)

    # En fait en tant que personne qui n'a pas le permis, ca me ferait chier que les credit soient accordes en fonction de ca
    # df.loc[:, 'OWN_CAR'] = 0
    # df.loc[df['FLAG_OWN_CAR'] == 'Y', 'OWN_CAR'] = 1
    # df.loc[:, 'OWN_REALTY'] = 0
    # df.loc[df['FLAG_OWN_REALTY'] == 'Y', 'OWN_REALTY'] = 1
    df.drop(columns=['FLAG_OWN_CAR', 'FLAG_OWN_REALTY'], inplace=True)

    # Pas tres utile mais tres correle au type de pret realise
    # docs = [f for f in df.columns if 'FLAG_DOC' in f]
    # df.loc[:, 'DOCUMENT_COUNT'] = df[docs].sum(axis=1)
    # # Categorical age - based on target=1 plot
    df.loc[:, 'AGE_RANGE'] = df['DAYS_BIRTH'].apply(lambda x: get_age_label(x))

    # Credit ratios
    df.loc[:, 'CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df.loc[:, 'CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    # Income ratios
    df.loc[:, 'ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    # CREDIT et ANNUITY  sont tres lies. J'ai decide d'enlever ANNUITY apres, donc je vais garder ton taux et enlever celui de CREDIT
    # df.loc[:, 'CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df.loc[:, 'INCOME_TO_EMPLOYED_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df.loc[:, 'INCOME_TO_BIRTH_RATIO'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']

    # 5-6 MIDI
    # # Very correlated to DAYS_EMPLOYED yet less strong
    # df.loc[:, 'EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

    # Highly Correlated to our new ratio, yet less correlated to TARGET
    list_of_feature_toRem.append('AMT_INCOME_TOTAL')
    # # Highly Correlated to our new ratio + CREDIT montant + Good Price du coup (peut-etre pourrait on meme virer 'AMT_GOODS_PRICE')
    # list_of_feature_toRem.append('AMT_ANNUITY')
    # Highly Correlated to our new ratio
    list_of_feature_toRem.append('AMT_CREDIT')
    list_of_feature_toRem.append('DAYS_BIRTH')

    # Enlever les colonnes qu'on desire enlever
    col_to_keep = [col for col in df.columns.tolist() if col not in list_of_feature_toRem]
    df = df[col_to_keep]

    # Categoriser l'education puisqu'il y a un ordre.
    df = col_categorisation(df, 'NAME_EDUCATION_TYPE',
                            ['Lower secondary', 'Secondary / secondary special', 'Incomplete higher',
                             'Higher education', 'Academic degree'])
    print()

    df_ohe, new_cols = fe810.one_hot_encoder(df=df, nan_as_category=False, treshold=encoding_treshold)

    # Comme on a peut etre fait des divisions par 0 ...
    df_ohe = fe810.replace_infinite_by_nan(df=df_ohe, list_new_columns=df_ohe.columns.tolist())

    return df_ohe


def bureau_and_balance(num_rows=None, nan_as_category=True, path='./input_data/',
                       nan_treshold=DEFAULT_EMPTY_FEAT_TRESHOLD_FOR_ROW):
    colonnes_to_drop = ['CREDIT_CURRENCY',
                        'DAYS_CREDIT_UPDATE',
                        # 'AMT_CREDIT_SUM_DEBT', # correlation with more powerful variable AMT_CREDIT_SUM
                        'CNT_CREDIT_PROLONG',
                        # 'DAYS_CREDIT_ENDDATE',
                        'DAYS_ENDDATE_FACT' # Regular inquiries can prolonge a credit (peu correlee)
                        ]

    bureau = pd.read_csv(path + 'bureau.csv', nrows=num_rows)
    bureau = fe810.remove_too_nany_observations(bureau, treshold=nan_treshold)

    bureau.loc[:, 'HAS_CLOSED'] = 0
    bureau.loc[bureau['CREDIT_ACTIVE'] == 'Closed', 'HAS_CLOSED'] = 1
    bureau.loc[:, 'HAS_CREDIT'] = 0
    bureau.loc[bureau['CREDIT_ACTIVE'] == 'Active', 'HAS_CREDIT'] = 1
    bureau.loc[:, 'HAS_DELAYED_BADDEBT'] = 0
    bureau.loc[(bureau['CREDIT_ACTIVE'] == 'Sold') | (bureau['CREDIT_ACTIVE'] == 'Bad debt'),'HAS_DELAYED_BADDEBT'] = 1


    # Credit duration and credit/account end date difference
    bureau['CREDIT_ENDDATE'] = 0
    bureau.loc[bureau['HAS_CREDIT'] == 1, 'CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_ENDDATE']
    # Credit to debt ratio and difference
    bureau['DEBT_PERCENTAGE'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_CREDIT_SUM_DEBT']
    # bureau['DEBT_CREDIT_DIFF'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
    bureau['CREDIT_TO_ANNUITY_RATIO'] = bureau['AMT_CREDIT_SUM'] / bureau['AMT_ANNUITY']

    bureau.drop(columns=colonnes_to_drop, inplace=True)

    bb = pd.read_csv(path + 'bureau_balance.csv', nrows=num_rows)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    # bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    bb_aggregations = {'MONTHS_BALANCE': ['mean']}
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    cat_aggregations = {
        'HAS_CREDIT': ['sum'],
        'HAS_CLOSED': ['sum'],
        'HAS_DELAYED_BADDEBT': ['mean']
    }

    num_aggregations = {
        'DAYS_CREDIT': ['mean'], # OLD TOP 50
        'CREDIT_ENDDATE': ['max'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean'],
        # 'CREDIT_TO_ANNUITY_RATIO': ['mean'],
        'DEBT_PERCENTAGE':['mean']
    }

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # HIGHLY CORRELATED TO les donnees de Bureau en general
    # + BURO DATA is more correlated to TARGET





    # # Bureau: Active credits - using only numerical aggregations
    # active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    # bur_act_agg_params = {
    #     'DAYS_CREDIT': ['mean'],  # OLD TOP 50
    #     'AMT_CREDIT_MAX_OVERDUE': ['mean'],
    #     'AMT_CREDIT_SUM_LIMIT': ['mean'],
    #     'DEBT_PERCENTAGE': ['mean']
    # }
    # active_agg = active.groupby('SK_ID_CURR').agg(bur_act_agg_params)
    # active_agg.columns = pd.Index(['BUR_ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    # bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    # del active, active_agg
    # gc.collect()
    # # Bureau: Closed credits - using only numerical aggregations
    # closed_agg_params = {
    #     # 'DAYS_CREDIT': ['mean'],  # OLD TOP 50 - But highly correlated to regular DAYS_CREDIT
    #     'AMT_CREDIT_MAX_OVERDUE': ['mean']
    # }
    # closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']
    # closed_agg = closed.groupby('SK_ID_CURR').agg(closed_agg_params)
    # closed_agg.columns = pd.Index(['BUR_CLO_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    # bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    # del closed, closed_agg, bureau
    # gc.collect()
    # Comme on a peut etre fait des divisions par 0 ...
    bureau_agg, bureau_agg_cat = fe810.one_hot_encoder(df=bureau_agg, nan_as_category=nan_as_category)

    bureau_agg = fe810.replace_infinite_by_nan(df=bureau_agg, list_new_columns=bureau_agg.columns.tolist())
    return bureau_agg


def previous_applications(num_rows=None, nan_as_category=False, path='./input_data/',
                          encoding_treshold=DEFAULT_CATEGORY_MEAN_FREQ,
                          nan_treshold=DEFAULT_EMPTY_FEAT_TRESHOLD_FOR_ROW):
    features_torem = ['WEEKDAY_APPR_PROCESS_START',
                      'NAME_TYPE_SUITE',  # On ne doit pas prendre en compte cet element
                      'NAME_GOODS_CATEGORY',  # Peu importe ce que la personne a achete precedemment
                      'CODE_REJECT_REASON',  # Apporte peu d'info mais ajoute 9 features + aggregation
                      'DAYS_DECISION',
                      # I dont see the connection between a decision we made and a demand they make (logreg doest care (212) while RF does (35) + highly correlated to INSTALL_DAYS_ENTRY_PAYMENT ...
                      'CHANNEL_TYPE',
                      # Peu d'interet de connaitre le canal d'acquisition, alors que ce va nous rajouter des colonnes en dummyfiant
                      'NAME_CONTRACT_TYPE',  # Cree des colonnes additionnelles tres correlees
                      'NAME_PRODUCT_TYPE',  # Je ne vois pas l'interet et ca ajoute des colonnes
                      'NAME_SELLER_INDUSTRY',  # Peu d'interet. Patronizing
                      'FLAG_LAST_APPL_PER_CONTRACT', # Peu d'interet alors que ca cree beaucoup de colonnes
                      'NAME_CASH_LOAN_PURPOSE', # TMI
                      'NAME_PAYMENT_TYPE', # tmi
                      'NAME_CLIENT_TYPE', # Ne doit pas entrer en ligne de compte pour accorder un pret
                      'NAME_PORTFOLIO',
                      'PRODUCT_COMBINATION', # Beaucoup de categories et peu d'infos
                      'HOUR_APPR_PROCESS_START'
                      ]
    prev = pd.read_csv(path + 'previous_application.csv', nrows=num_rows)
    prev = fe810.remove_too_nany_observations(prev)
    install = pd.read_csv(path + 'installments_payments.csv', nrows=num_rows)
    install = fe810.remove_too_nany_observations(install)

    # Pas d'interet de conserver les demarches abandonnees, elles faussent nos indicateurs
    prev = prev[(prev['NAME_CONTRACT_STATUS']!='Canceled') & (prev['NAME_CONTRACT_STATUS']!='Unused offer')]

    prev.loc[:, 'YIELD_GROUP_HIGH'] = 0
    prev.loc[prev['NAME_YIELD_GROUP'] == 'high', 'YIELD_GROUP_HIGH'] = 1
    prev.drop(columns=['NAME_YIELD_GROUP'], inplace=True)

    # Feature engineering: ratios and difference
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['CREDIT_TO_ANNUITY_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_ANNUITY']
    prev['DOWN_PAYMENT_TO_CREDIT'] = prev['AMT_DOWN_PAYMENT'] / prev['AMT_CREDIT']

    # Interest ratio on previous application (simplified)
    total_payment = prev['AMT_ANNUITY'] * prev['CNT_PAYMENT']
    prev['SIMPLE_INTERESTS'] = (total_payment / prev['AMT_CREDIT'] - 1) / prev['CNT_PAYMENT']




    prev.loc[:, 'HAS_REFUSED'] = 0
    prev.loc[prev['NAME_CONTRACT_STATUS'] == 'Refused', 'HAS_REFUSED'] = 1
    prev.loc[:, 'HAS_ACTIVE'] = 0
    prev.loc[(prev['NAME_CONTRACT_STATUS'] == 'Approved') & (prev['DAYS_LAST_DUE'] == 365243), 'HAS_ACTIVE'] = 1



    # CREDITS ACTIFS
    # Active loans - approved and not complete yet (last_due 365243) - valeur speciale !
    approved = prev[prev['NAME_CONTRACT_STATUS'] == "Approved"]
    active_df = approved[approved['DAYS_LAST_DUE'] == 365243]
    # Find how much was already payed in active loans (using installments csv)
    active_pay = install[install['SK_ID_PREV'].isin(active_df['SK_ID_PREV'])]

    active_pay_agg = active_pay.groupby('SK_ID_PREV')[['AMT_INSTALMENT', 'AMT_PAYMENT']].sum()
    active_pay_agg.reset_index(inplace=True)
    active_pay_agg['INSTALMENT_PAYMENT_DIFF'] = active_pay_agg['AMT_INSTALMENT'] - active_pay_agg['AMT_PAYMENT']

    # Merge with active_df
    active_df = active_df.merge(active_pay_agg, on='SK_ID_PREV', how='left')

    active_df['REMAINING_DEBT'] = active_df['AMT_CREDIT'] - active_df['AMT_PAYMENT']

    bureau_active_agg = {
        'REMAINING_DEBT': ['sum'],
        'AMT_INSTALMENT': ['sum'],
    }
    active_agg_df = active_df.groupby('SK_ID_CURR').agg({**bureau_active_agg})
    active_agg_df.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg_df.columns.tolist()])

    # active_agg_df['TOTAL_REPAYMENT_RATIO'] = active_agg_df['ACTIVE_AMT_PAYMENT_SUM'] / active_agg_df[
    #     'ACTIVE_AMT_CREDIT_SUM']
    # active_agg_df.drop(columns=['ACTIVE_AMT_PAYMENT_SUM', 'ACTIVE_AMT_CREDIT_SUM'], inplace=True)
    del active_pay, active_pay_agg, active_df;
    gc.collect()


    # CREDITS REFUSED
    # Aggregations for approved and refused loans
    refused = prev[prev['NAME_CONTRACT_STATUS'] == "Refused"]
    bureau_refused_agg = {
        'APP_CREDIT_PERC': ['mean']
    }
    refused_agg_df = refused.groupby('SK_ID_CURR').agg({**bureau_refused_agg})
    refused_agg_df.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg_df.columns.tolist()])
    del refused; gc.collect()


    # CREDITS APPROVED
    bureau_approved_agg = {
        'AMT_DOWN_PAYMENT': ['sum']
    }
    approved_agg_df = approved.groupby('SK_ID_CURR').agg({**bureau_approved_agg})
    approved_agg_df.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg_df.columns.tolist()])
    del approved; gc.collect()


    # LATE PAYMENTS
    # Get the SK_ID_PREV for loans with late payments (days past due)
    install['LATE_PAYMENT'] = install['DAYS_ENTRY_PAYMENT'] - install['DAYS_INSTALMENT']
    install['LATE_PAYMENT'] = install['LATE_PAYMENT'].apply(lambda x: 1 if x > 0 else 0)
    dpd_id = install[install['LATE_PAYMENT'] > 0]['SK_ID_PREV'].unique()

    prev.loc[:, 'HAD_LATE_PAYMENTS'] = 0
    prev.loc[prev['SK_ID_PREV'].isin(dpd_id), 'HAD_LATE_PAYMENTS'] = 1


    # Change 365.243 values to nan (missing)
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    # Days last due difference (scheduled x done)
    prev['DAYS_LAST_DUE_DIFF'] = prev['DAYS_LAST_DUE_1ST_VERSION'] - prev['DAYS_LAST_DUE']

    prev = prev[[col for col in prev.columns if col not in features_torem]]

    prev_aggregation = {
        'CREDIT_TO_ANNUITY_RATIO': ['mean'],
        'SIMPLE_INTERESTS': ['mean'],
        'DOWN_PAYMENT_TO_CREDIT': ['mean'],
        'DAYS_LAST_DUE_DIFF': ['mean'],
        'YIELD_GROUP_HIGH': ['sum'],
        'HAS_REFUSED':['sum'],
        'HAS_ACTIVE':['sum'],
        'HAD_LATE_PAYMENTS':['mean']
    }

    prev = prev.groupby('SK_ID_CURR').agg({**prev_aggregation})  # , **cat_aggregations
    prev.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev.columns.tolist()])


    # MERGE ALL DF
    prev = prev.merge(active_agg_df, how='left', on='SK_ID_CURR')
    del active_agg_df;
    gc.collect()
    prev = prev.merge(refused_agg_df, how='left', on='SK_ID_CURR')
    del refused_agg_df;
    gc.collect()
    prev = prev.merge(approved_agg_df, how='left', on='SK_ID_CURR')
    del approved_agg_df;
    gc.collect()




    prev.loc[prev['PREV_HAS_ACTIVE_SUM'] == 0, 'ACTIVE_REMAINING_DEBT_SUM'] = 0
    prev.loc[prev['PREV_HAS_ACTIVE_SUM'] == 0, 'ACTIVE_AMT_INSTALMENT_SUM'] = 0

    prev.loc[prev['PREV_HAS_REFUSED_SUM'] == 0, 'REFUSED_APP_CREDIT_PERC_MEAN'] = 0
    prev.loc[prev['PREV_HAS_REFUSED_SUM'] == 0, 'REFUSED_DAYS_DECISION_MEAN'] = 0

    # prev.drop(columns = ['PREV_HAS_ACTIVE_SUM'], inplace=True)
    # Comme on a peut etre fait des divisions par 0 ...
    prev = fe810.replace_infinite_by_nan(df=prev, list_new_columns=prev.columns.tolist())
    prev.drop(columns=['PREV_HAS_ACTIVE_SUM'], inplace=True)
    return prev


def pos_cash(num_rows=None, nan_as_category=False, path='./input_data/'):
    col_to_remove = [
        'MONTHS_BALANCE', # Highly correlated to INSTAL_DAYS_ENTRY_PAYMENT_MEAN
        'NAME_CONTRACT_STATUS'
        # Classe tres mal repartie. Une categorie active sur-representee et des categories tierces tres peu frequentes.
    ]
    pos = pd.read_csv(path + 'POS_CASH_balance.csv', nrows=num_rows)
    pos.drop(columns=col_to_remove, inplace=True)
    pos, cat_cols = fe810.one_hot_encoder(pos, nan_as_category=nan_as_category)
    # Features
    aggregations = {
        # 'MONTHS_BALANCE': ['mean'],
        'SK_DPD': ['mean'],
        'SK_DPD_DEF': ['mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']

    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()

    # Comme on a peut etre fait des divisions par 0 ...
    pos_agg = fe810.replace_infinite_by_nan(df=pos_agg, list_new_columns=pos_agg.columns.tolist())
    return pos_agg


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=False, path='./input_data/'):
    ins = pd.read_csv(path + 'installments_payments.csv', nrows=num_rows)
    ins, cat_cols = fe810.one_hot_encoder(ins, nan_as_category=nan_as_category)

    # On suprrimer les variables les moins correlees a la TARGET entre deux variables tres correlees
    col_to_rem = ['AMT_INSTALMENT',
                  'NUM_INSTALMENT_VERSION'  # Pas tres interessant
                  ]

    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        # 'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['mean'],
        'DBD': ['mean'],
        'PAYMENT_PERC': ['mean'],
        'PAYMENT_DIFF': ['mean'],
        # 'AMT_INSTALMENT': ['mean'],
        'AMT_PAYMENT': ['mean'],
        'DAYS_ENTRY_PAYMENT': ['mean']
    }

    ins.drop(columns=col_to_rem, inplace=True)

    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()

    # Comme on a peut etre fait des divisions par 0 ...
    ins_agg = fe810.replace_infinite_by_nan(df=ins_agg, list_new_columns=ins_agg.columns.tolist())
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=False, path='./input_data/'):
    # Balance et receivable tres correlees, mais l'un est plus correle a la TARGET
    col_ro_rem = ['NAME_CONTRACT_STATUS',  # Colonne trop peu diversifiee. Risque d'overfit + augmente dimension
                  'AMT_INST_MIN_REGULARITY',  # The two following are highly correlated to AMT_BALANCE
                  'AMT_RECEIVABLE_PRINCIPAL',
                  'AMT_RECIVABLE',
                  'AMT_TOTAL_RECEIVABLE',  # J'ai un doute
                  'AMT_DRAWINGS_ATM_CURRENT',  # AMounT moins correle a la target que CouNT
                  'AMT_DRAWINGS_POS_CURRENT',  # C'est du flicage !
                  'AMT_PAYMENT_CURRENT',  # Pas tres correlee a la TARGET
                  'CNT_DRAWINGS_POS_CURRENT', # On a la variable plus generale CNT_DRAWINGS_CURRENT
                  'AMT_DRAWINGS_OTHER_CURRENT', # On garde le CNT
                  'AMT_PAYMENT_TOTAL_CURRENT', # Trop correle a CC_AMT_DRAWINGS_CURRENT_MEAN
                  'CNT_INSTALMENT_MATURE_CUM', # Correle a trop d'autres variables (e.g : INSTAL_COUNT)
                  'AMT_DRAWINGS_CURRENT', # Tres correle a la balance et je trouve que c'est un peu TMI si la banque utilise ca
                  'CNT_DRAWINGS_ATM_CURRENT', # Pas envie de penaliser quelqu'un qui utiliserait du Cash
                  'CNT_DRAWINGS_OTHER_CURRENT',
                  'SK_DPD',
                  'SK_DPD_DEF',
                  'AMT_CREDIT_LIMIT_ACTUAL'
                  ]
    cc = pd.read_csv(path + 'credit_card_balance.csv', nrows=num_rows)

    # Amount used from limit
    cc['LIMIT_USE'] = cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL']
    # Late payment
    cc['LATE_PAYMENT'] = cc['SK_DPD'].apply(lambda x: 1 if x > 0 else 0)
    # How much drawing of limit
    cc['DRAWING_LIMIT_RATIO'] = cc['AMT_DRAWINGS_ATM_CURRENT'] / cc['AMT_CREDIT_LIMIT_ACTUAL']

    cc.drop(columns=col_ro_rem, inplace=True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis=1, inplace=True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['mean'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # # Count credit card lines
    # cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()

    # Comme on a peut etre fait des divisions par 0 ...
    cc_agg = fe810.replace_infinite_by_nan(df=cc_agg, list_new_columns=cc_agg.columns.tolist())
    return cc_agg


##############################################################################################
# ------------------------------------------- CREATE PIPELINE FUNCTION
##############################################################################################

def full_feature_engineering(df_input, df_folder, encoding_treshold=DEFAULT_CATEGORY_MEAN_FREQ,
                             nan_treshold=DEFAULT_EMPTY_FEAT_TRESHOLD_FOR_ROW):
    with timer("application processing"):
        df = process_application(df=df_input, encoding_treshold=encoding_treshold, nan_treshold=nan_treshold)
        print("Application dataframe shape: ", df.shape)
    with timer("Bureau and bureau_balance processing"):
        bureau_df = bureau_and_balance(path=df_folder)
        df = pd.merge(df, bureau_df, on='SK_ID_CURR', how='left')
        print("Bureau dataframe shape: ", bureau_df.shape)
        del bureau_df
        gc.collect()
    with timer("Previous application processing"):
        prevapp = previous_applications(path=df_folder, encoding_treshold=0.1, nan_treshold=0.4)
        df = pd.merge(df, prevapp, on='SK_ID_CURR', how='left')
        print("Previous Application dataframe shape: ", prevapp.shape)
        del prevapp
        gc.collect()
    with timer("Pos-Cash processing"):
        poscash = pos_cash(path=df_folder)
        df = pd.merge(df, poscash, on='SK_ID_CURR', how='left')
        print("Pos-Cash dataframe shape: ", poscash.shape)
        del poscash
        gc.collect()
    with timer("Installment processing"):
        installmentspayments = installments_payments(path=df_folder)
        df = pd.merge(df, installmentspayments, on='SK_ID_CURR', how='left')
        print("Installement dataframe shape: ", installmentspayments.shape)
        del installmentspayments
        gc.collect()
    with timer("CC processing"):
        creditcardbalance = credit_card_balance(path=df_folder)
        df = pd.merge(df, creditcardbalance, on='SK_ID_CURR', how='left')
        print("CC dataframe shape: ", creditcardbalance.shape)
        del creditcardbalance
        gc.collect()
    # # MBR : On ne va pas prendre les colonnes trop vide, car ca implique qu'on impute gracement = donc biais
    # df = u810.remove_too_nany_columns(df=df, treshold=0.4)
    df = fe810.reduce_memory(df)
    return df