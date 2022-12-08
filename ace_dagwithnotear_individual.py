from src.utils import data_load
import pandas as pd
from src.s3_utils import pandas_from_csv_s3
import re
import networkx
import notears.notears as notears
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = data_load(data_keys={'phq9', 'generalized_anxiety_disorder_scale_gad7', 'ace', 'surveys', 'study_ids', 'check_in_adherence_log'})
outcomes = pd.merge(data['phq9'], data['generalized_anxiety_disorder_scale_gad7'],  how='outer', left_on=['record_id','redcap_event_name'], right_on = ['record_id','redcap_event_name'])
overall_df = pd.merge(data['ace'].drop(columns=['redcap_event_name']).dropna(), outcomes, how='left', on='record_id')

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

# process promis global 10 survey
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

#process final individual question dataframe
ace_lst = [f'ace_{x}' for x in range(1,11)]
phq9_lst = [f'phq9_{x}' for x in range(1,11)]
gad_lst = [f'gad_{x}' for x in range(1,9)]
question_labels = ace_lst + phq9_lst + gad_lst + promis_mental_lst + promis_physical_lst
processed_individual_questions_df = pd.DataFrame(columns=['user_id'] + question_labels)

for uid in overall_df['user_id'].unique():
    each_df = overall_df.loc[overall_df['user_id']==uid]
    ace_sum = each_df[ace_lst].mean(axis=0)
    phq9_sum = each_df[phq9_lst].mean(axis=0)
    gad_sum = each_df[gad_lst].mean(axis=0)
    promis_mental_sum = promis_survey_processed.loc[promis_survey_processed['user_id']==uid][promis_mental_lst].mean(axis=0)
    promis_physical_sum = promis_survey_processed.loc[promis_survey_processed['user_id']==uid][promis_physical_lst].mean(axis=0)
    processed_individual_questions_df = processed_individual_questions_df.append(pd.DataFrame([[uid] + ace_sum.tolist() + phq9_sum.tolist() + gad_sum.tolist() + promis_mental_sum.tolist() + promis_physical_sum.tolist()], columns=['user_id'] + question_labels), ignore_index = True)
processed_individual_questions_df = processed_individual_questions_df.dropna()

for num in range(7, 11):
    # randomly select 40 samples from the processed individual questions dataframe
    data = processed_individual_questions_df[question_labels].sample(n=40).to_numpy()
    np.save(f'results/inputs/trial_{num}.npy', data)
    
    output_dict = notears.run(notears.notears_standard, data.tolist(), notears.loss.least_squares_loss, notears.loss.least_squares_loss_grad, e=1e-8, verbose=False)
    np.save(f'results/outputs/trial_{num}.npy', output_dict['W'])

    acyclic_W = notears.utils.threshold_output(output_dict['W'])
    np.save(f'results/threshold_outputs/trial_{num}.npy', acyclic_W)
    print('Acyclicity loss: {}'.format(output_dict['h']))
    print('Least squares loss: {}'.format(output_dict['loss']))

    # save output matrice to results folder

    f1, ax1 = plt.subplots(figsize=(13, 14))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(output_dict['W'], cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=question_labels, yticklabels=question_labels)
    plt.savefig(f'results/diagrams/output_trial_{num}.png')
    
    f2, ax2 = plt.subplots(figsize=(13, 14))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(acyclic_W, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=question_labels, yticklabels=question_labels)
    plt.savefig(f'results/diagrams/threshold_output_trial_{num}.png')
    