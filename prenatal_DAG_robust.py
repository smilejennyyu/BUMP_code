# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

from src.utils import data_load
import pandas as pd
from src.s3_utils import pandas_from_csv_s3
import re
import os

# # Data processing: Join PHQ9, GAD7 and ACE datasets together by record_id and redcap_event_name

data = data_load(data_keys={'phq9', 'generalized_anxiety_disorder_scale_gad7', 'ace', 'surveys', 'study_ids', 'check_in_adherence_log'})


outcomes = pd.merge(data['phq9'], data['generalized_anxiety_disorder_scale_gad7'],  how='outer', left_on=['record_id','redcap_event_name'], right_on = ['record_id','redcap_event_name'])


#
overall_df = pd.merge(data['ace'].drop(columns=['redcap_event_name']).dropna(), outcomes, how='left', on='record_id')

# [markdown]
# # Convert redcap_event_name to date for PHQ9, GAD7 and ACE datasets

# read study ids
id_df = data['study_ids'][['record_id', 'evidation_id']]
id_df.rename(columns={'evidation_id': 'user_id'}, inplace=True)

# add ids to survey
overall_df = overall_df.merge(id_df, on=['record_id'])
overall_df.user_id = overall_df.user_id.fillna(-1).astype(int)

# standarize naming convention for easier processing later on
overall_df.redcap_event_name = overall_df.redcap_event_name.replace('postnatal_checkin_arm_1','postnatal_ci_1_arm_1')


# read check-in dates
ci_df = data['check_in_adherence_log']
cols = ['record_id'] + [col for col in ci_df.columns if '_date' in col]
ci_df = ci_df[cols]

# standarize naming convention for easier processing later on
ci_df = ci_df.rename(columns={'checkin_postnatal_date': 'checkin_postnatal_date_1'})

# add dates to survey, need to map it using the check_in_adherence_log
def conver_checkin_string(x):
    x = x.split('_arm')[0] #delete all characters after the word 'arm'
    num = int(re.search(r'\d+', x).group())
    if 'postnatal' in x:
        return f'checkin_postnatal_date_{num}'
    else:
        return f'checkin_{num}_date'

# map checkin_postnature_date_{num} OR checkin_{num}_date to the actual date
def map_date(x):
    checkin_string_col = x['checkin_string']
    return x[checkin_string_col]

overall_df = overall_df.merge(ci_df, on=['record_id'])
overall_df['checkin_string'] = overall_df.redcap_event_name.apply(conver_checkin_string)
overall_df['date'] = overall_df.apply(map_date, axis=1)
overall_df = overall_df[overall_df.columns.drop(list(overall_df.filter(regex='checkin_')))]
overall_df['date'] = pd.to_datetime(overall_df['date'])


#
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 1000)
pd.set_option('max_colwidth', None)
overall_df.loc[overall_df['record_id'] == 28][['date', 'redcap_event_name']] #check if dates are correctly processed


def separate_by_delivery(overall_df, same_set_users=True):
    
    with_postnatal_users = overall_df[overall_df['redcap_event_name'].str.contains('postnatal')]['user_id'].unique()
    with_postnatal_df = overall_df[overall_df['user_id'].isin(with_postnatal_users)]

    if same_set_users:
        df = with_postnatal_df
    else:
        df = overall_df
    postnatal_df = df[df['redcap_event_name'].str.contains('postnatal')]
    prenatal_df = df[~df['redcap_event_name'].str.contains('postnatal')]
    return prenatal_df, postnatal_df
prenatal_df, postnatal_df = separate_by_delivery(overall_df, same_set_users=False)

# [markdown]
# # Add Global survey data - PROMIS quality of life

#
answer_dict0 = {
    0: 4,
    1: 3,
    2: 3,
    4: 2,
    5: 2,
    6: 2,
    7: 1,
    8: 1,
    9: 1,
    10: 0
}
answer_dict1 = {
    'Excellent': 4,
    'Very good': 3,
    'Good': 2,
    'Fair': 1,
    'Poor': 0
}
answer_dict2 = {
    'Completely': 4,
    'Mostly': 3,
    'Moderately': 2,
    'A little': 1,
    'Not at all': 0
}
answer_dict3 = {
    'Never': 4,
    'Rarely': 3,
    'Sometimes': 2,
    'Often': 2,
    'Always': 0
}
answer_dict4 = {
    'None': 4,
    'Mild': 3,
    'Moderate': 2,
    'Severe': 1,
    'Very severe': 0
}
# question 119 and 114 are not used in Global Physical Health and Global Mental Health. Refer to the BUMP data dictionary excel sheet
drop_question_ids = [114, 119]
promis_survey = data['surveys'].loc[data['surveys']['title']=='Global survey']
promis_survey.drop(promis_survey[promis_survey['question_id'].isin(drop_question_ids)].index, inplace=True)
for qid in promis_survey['question_id'].unique():
    question = promis_survey.loc[promis_survey['question_id'] == qid]
    if qid == 123:
        question['answer_text'] = question['answer_text'].astype(int)
        question.replace({"answer_text": answer_dict0}, inplace=True)
    elif qid in {115, 116, 117, 118}:
        question.replace({"answer_text": answer_dict1}, inplace=True)
    elif qid == 120:
        question.replace({"answer_text": answer_dict2}, inplace=True)
    elif qid == 121:
        question.replace({"answer_text": answer_dict3}, inplace=True)
    elif qid == 122:
        question.replace({"answer_text": answer_dict4}, inplace=True)
    promis_survey.loc[promis_survey['question_id'] == qid] = question
promis_survey['date'] = pd.to_datetime(promis_survey['date'])


#
promis_survey.head()


#
promis_mental_lst = [f'promis_global10_mental_{x}' for x in range(1,5)]
promis_physical_lst = [f'promis_global10_physical_{x}' for x in range(1,5)]
mental_health_question_ids = [115, 117, 118, 121]
physical_health_question_ids = [116, 120, 122, 123]
promis_survey_processed = pd.DataFrame(columns=['user_id', 'date'] + promis_mental_lst + promis_physical_lst)

for uid in promis_survey['user_id'].unique():
    each_user_survey = promis_survey.loc[promis_survey['user_id'] == uid]
    for date in each_user_survey['date'].unique():
        # answer_text = each_user_survey.loc[each_user_survey['date'] == date].sort_values(by='question_id')['answer_text'].to_numpy()
        each_date_survey = each_user_survey.loc[each_user_survey['date'] == date]
        mental_health_answers = each_date_survey[each_date_survey['question_id'].isin(mental_health_question_ids)].sort_values(by='question_id')['answer_text']
        physical_health_answers = each_date_survey[each_date_survey['question_id'].isin(physical_health_question_ids)].sort_values(by='question_id')['answer_text']
        if len(mental_health_answers) == 4 and len(physical_health_answers) == 4:
            promis_survey_processed = promis_survey_processed.append(pd.DataFrame([[uid, date] + mental_health_answers.tolist() + physical_health_answers.tolist()], columns=['user_id', 'date'] + promis_mental_lst + promis_physical_lst), ignore_index = True)

promis_survey_processed


#
import numpy as np
def separate_promis(prenatal_df, postnatal_df, promis_df):
    all_data_available_user_id = []
    promis_postnatal_df_list = []
    promis_prenatal_df_list = []
    for user in postnatal_df.user_id.unique():
        curr_user_df = promis_df[promis_df['user_id'] == user]
        promis_postnatal_exist = np.max(curr_user_df.date) >= np.min(postnatal_df[postnatal_df['user_id'] == user].date)
        if promis_postnatal_exist:
            # all_data_available_user_id.append(user)
            promis_postnatal_df_list.append(curr_user_df[curr_user_df['date'] >= np.min(postnatal_df[postnatal_df['user_id'] == user].date)])
            
    for user in prenatal_df.user_id.unique():
        curr_user_df = promis_df[promis_df['user_id'] == user]

        promis_prenatal_exist = np.min(curr_user_df.date) <= np.max(prenatal_df[prenatal_df['user_id'] == user].date)

        if promis_prenatal_exist:
            promis_prenatal_df_list.append(curr_user_df[curr_user_df['date'] <= np.max(prenatal_df[prenatal_df['user_id'] == user].date)])
    promis_postnatal_df = pd.concat(promis_postnatal_df_list, ignore_index=True)
    promis_prenatal_df = pd.concat(promis_prenatal_df_list, ignore_index=True)
    return promis_prenatal_df, promis_postnatal_df
promis_prenatal_df, promis_postnatal_df = separate_promis(prenatal_df, postnatal_df, promis_survey_processed)


#
# promis_prenatal_df.head()


# #
# promis_postnatal_df.head()

# [markdown]
# # Process PHQ9, GAD and PROMIS data by taking the average over time for each individual

#
# # more balanced set up
# ace_levels = {
#     (0, 1) : 0,
#     (2, 10): 1
# }
# phq9_levels = {
#     (0, 4) : 0,
#     (5, 9): 1,
#     (10, 14): 1,
#     (15, 19): 1,
#     (20, 27): 1
# }
# gad_levels = {
#     (0, 4) : 0,
#     (5, 9): 1,
#     (10, 14): 1,
#     (15, 21): 1
# }
# promis_levels = {
#     "Always" : 1,
#     "Often": 1,
#     "Sometimes": 1,
#     "Rarely": 0,
#     "Never": 0
# }

# from literature
ace_levels = {
    (0, 4) : 0,
    (5, 10): 1
}
phq9_levels = {
    (0, 4) : 0,
    (5, 9): 1,
    (10, 14): 2,
    (15, 19): 3,
    (20, 27): 4
}
gad_levels = {
    (0, 4) : 0,
    (5, 9): 1,
    (10, 14): 2,
    (15, 21): 3
}
promis_mental_levels = {
    (0, 11) : 0,
    (12, 16): 1,
}
promis_physical_levels = {
    (0, 10) : 0,
    (11, 16): 1,
}
def map_levels(x, map_dict):
    for key in map_dict:
        if isinstance(x, str):
            if x == key:
                return map_dict[key]
        else:
            if x >= key[0] and x <= key[1]:
                return map_dict[key]


#
# promis_mental_sum = promis_prenatal_df.loc[promis_prenatal_df['user_id']==uid][promis_mental_lst].sum(axis=1)


#
def process_all_df(overall_df, promis_survey_processed):
    processed_overall_df = pd.DataFrame(columns=['user_id', 'ace_sum', 'phq9_sum', 'gad_sum', 'promis_mental_mean', 'promis_physical_mean'])
    ace_lst = [f'ace_{x}' for x in range(1,11)]
    phq9_lst = [f'phq9_{x}' for x in range(1,11)]
    gad_lst = [f'gad_{x}' for x in range(1,9)]
    for uid in overall_df['user_id'].unique():
        each_df = overall_df.loc[overall_df['user_id']==uid]
        ace_sum = each_df[ace_lst].sum(axis=1)
        ace_sum_mean = ace_sum.apply(map_levels, map_dict=ace_levels).mean()

        phq9_sum = each_df[phq9_lst].sum(axis=1)
        phq9_sum_mean = phq9_sum.apply(map_levels, map_dict=phq9_levels).mean()

        gad_sum = each_df[gad_lst].sum(axis=1)
        gad_sum_mean = gad_sum.apply(map_levels, map_dict=gad_levels).mean()

        promis_mental_sum = promis_survey_processed.loc[promis_survey_processed['user_id']==uid][promis_mental_lst].sum(axis=1)
        promis_mental_mean = promis_mental_sum.apply(map_levels, map_dict=promis_mental_levels).mean()

        promis_physical_sum = promis_survey_processed.loc[promis_survey_processed['user_id']==uid][promis_physical_lst].sum(axis=1)
        promis_physical_mean = promis_physical_sum.apply(map_levels, map_dict=promis_physical_levels).mean()
        
        processed_overall_df = processed_overall_df.append({'user_id': uid, 'ace_sum': ace_sum_mean, 'promis_physical_mean': promis_physical_mean, 'promis_mental_mean': promis_mental_mean, 'phq9_sum': phq9_sum_mean, 'gad_sum': gad_sum_mean}, ignore_index=True)
    return processed_overall_df


#
# processed_overall_df = process_all_df(prenatal_df, promis_prenatal_df)
# # processed_overall_df = process_all_df(postnatal_df, promis_postnatal_df)

# processed_overall_df = processed_overall_df.dropna()

# processed_overall_df['ace_sum'].hist()
# processed_overall_df['phq9_sum'].round().hist()
# processed_overall_df['gad_sum'].round().hist()
# processed_overall_df['promis_mental_mean'].round().hist()
# processed_overall_df['promis_physical_mean'].round().hist()

# fig, axs = plt.subplots(3, 2)
# axs[0, 0].hist(processed_overall_df['ace_sum'])
# axs[0, 0].set_title('Ace sum')
# axs[0, 1].hist(processed_overall_df['phq9_sum'].round())
# axs[0, 1].set_title('PHQ9 sum')
# axs[1, 0].hist(processed_overall_df['gad_sum'].round())
# axs[1, 0].set_title('GAD sum')
# axs[1, 1].hist(processed_overall_df['promis_mental_mean'].round())
# axs[1, 1].set_title('promis mental mean')
# axs[2, 0].hist(processed_overall_df['promis_physical_mean'].round())
# axs[2, 0].set_title('promis physical mean')
# fig.tight_layout()


#
import networkx
import notears.notears as notears
import matplotlib.pyplot as plt

def get_adjacency_mat(processed_overall_df):
    data = processed_overall_df[['ace_sum', 'phq9_sum', 'gad_sum', 'promis_mental_mean', 'promis_physical_mean']].to_numpy().tolist()
    output_dict = notears.run(notears.notears_standard, data, notears.loss.least_squares_loss, notears.loss.least_squares_loss_grad, e=1e-8, verbose=False)


    #
    print('Acyclicity loss: {}'.format(output_dict['h']))
    print('Least squares loss: {}'.format(output_dict['loss']))

    plt.matshow(output_dict['W'])
    plt.title("Learned adjacency matrix")
    plt.colorbar()

    acyclic_W = notears.utils.threshold_output(output_dict['W'])

    plt.matshow(acyclic_W)
    plt.title("Learned adjacency matrix (thresholded)")
    plt.colorbar()

    G = networkx.DiGraph(acyclic_W)
    networkx.draw(G, with_labels=True)

    weighted_G = networkx.DiGraph((output_dict['W'] * acyclic_W).round(1))
    layout = networkx.shell_layout(weighted_G)
    networkx.draw(weighted_G, layout, node_size=1000, with_labels=True, font_weight='bold', font_size=15)
    labels = networkx.get_edge_attributes(weighted_G,'weight')
    networkx.draw_networkx_edge_labels(weighted_G,pos=layout,edge_labels=labels)
    plt.show()

    return output_dict['W']

path = "/mnt/results/adj_mat"
# os.mkdir(path)
from random import sample
unique_user_lst = list(promis_prenatal_df['user_id'].unique())
for i in range(10):
    
    sampled_user_lst = sample(unique_user_lst, 40)
    
    promis_prenatal_df_sampled = promis_prenatal_df[promis_prenatal_df['user_id'].isin(sampled_user_lst)]
    prenatal_df_sampled = prenatal_df[prenatal_df['user_id'].isin(sampled_user_lst)]
    processed_overall_df = process_all_df(prenatal_df_sampled, promis_prenatal_df_sampled)

    W = get_adjacency_mat(processed_overall_df)
    np.save(os.path.join(path, f"W_{i}.npy"), W)

# # [markdown]
# # # Individual similarity analysis

# #
# ace_lst = [f'ace_{x}' for x in range(1,11)]
# phq9_lst = [f'phq9_{x}' for x in range(1,11)]
# gad_lst = [f'gad_{x}' for x in range(1,9)]
# processed_individual_questions_df = pd.DataFrame(columns=['user_id', 'ace_sum'] + phq9_lst + gad_lst + promis_mental_lst + promis_physical_lst)

# for uid in overall_df['user_id'].unique():
#     each_df = overall_df.loc[overall_df['user_id']==uid]
#     ace_sum = each_df[ace_lst].sum(axis=1)
#     ace_sum_mean = ace_sum.apply(map_levels, map_dict=ace_levels).mean()

#     phq9_sum = each_df[phq9_lst].mean(axis=0)

#     gad_sum = each_df[gad_lst].mean(axis=0)

#     promis_mental_sum = promis_survey_processed.loc[promis_survey_processed['user_id']==uid][promis_mental_lst].mean(axis=0)

#     promis_physical_sum = promis_survey_processed.loc[promis_survey_processed['user_id']==uid][promis_physical_lst].mean(axis=0)
    
#     processed_individual_questions_df = processed_individual_questions_df.append(pd.DataFrame([[uid, ace_sum_mean] + phq9_sum.tolist() + gad_sum.tolist() + promis_mental_sum.tolist() + promis_physical_sum.tolist()], columns=['user_id', 'ace_sum'] + phq9_lst + gad_lst + promis_mental_lst + promis_physical_lst), ignore_index = True)


# #
# processed_individual_questions_df = processed_individual_questions_df.dropna()


# #
# len(processed_individual_questions_df.columns)


# #
# data = processed_individual_questions_df[phq9_lst + gad_lst + promis_mental_lst + promis_physical_lst].to_numpy().tolist()
# output_dict = notears.run(notears.notears_standard, data, notears.loss.least_squares_loss, notears.loss.least_squares_loss_grad, e=1e-8, verbose=False)


# #
# print('Acyclicity loss: {}'.format(output_dict['h']))
# print('Least squares loss: {}'.format(output_dict['loss']))


# #
# plt.matshow(output_dict['W'])
# plt.title("Learned adjacency matrix")
# plt.colorbar()


# #
# acyclic_W = notears.utils.threshold_output(output_dict['W'])


# #
# plt.matshow(acyclic_W)
# plt.title("Learned adjacency matrix (thresholded)")
# plt.colorbar()


#


