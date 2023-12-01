import os
import json
import pandas as pd
import numpy as np

from transformers import BertTokenizer

import re
from tqdm import tqdm
from fuzzywuzzy import fuzz
from datetime import datetime
from textblob import TextBlob
import emoji

from sklearn.tree import DecisionTreeClassifier
from gplearn.genetic import SymbolicTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

emoji_pattern = re.compile(
    r"[\U0001F004-\U0001F0CF\U0001F170-\U0001F171\U0001F17E\U0001F17F\U0001F18E"
    r"\U0001F191-\U0001F19A\U0001F201-\U0001F202\U0001F21A\U0001F22F\U0001F232-\U0001F23A"
    r"\U0001F250-\U0001F251\U0001F300-\U0001F321\U0001F324-\U0001F393\U0001F396-\U0001F397"
    r"\U0001F399\U0001F39A-\U0001F39B\U0001F39E-\U0001F3F0\U0001F3F3-\U0001F3F5\U0001F3F7-\U0001F4FD"
    r"\U0001F4FF-\U0001F53D\U0001F549-\U0001F54E\U0001F550-\U0001F567\U0001F56F-\U0001F570"
    r"\U0001F573-\U0001F57A\U0001F587\U0001F58A-\U0001F58D\U0001F590\U0001F595-\U0001F596"
    r"\U0001F5A4-\U0001F5A5\U0001F5A8\U0001F5B1-\U0001F5B2\U0001F5BC\U0001F5C2-\U0001F5C4"
    r"\U0001F5D1-\U0001F5D3\U0001F5DC-\U0001F5DE\U0001F5E1\U0001F5E3\U0001F5E8\U0001F5EF"
    r"\U0001F5F3\U0001F5FA-\U0001F64F\U0001F680-\U0001F6C5\U0001F6CB-\U0001F6D2\U0001F6D5-\U0001F6D7"
    r"\U0001F6E0-\U0001F6E5\U0001F6E9\U0001F6EB-\U0001F6EC\U0001F6F0\U0001F6F3-\U0001F6FC"
    r"\U0001F7E0-\U0001F93A\U0001F93C-\U0001F945\U0001F947-\U0001F978\U0001F97A-\U0001F9CB"
    r"\U0001F9CD-\U0001FA74\U0001FA78-\U0001FA86\U0001FA90-\U0001FAA8\U0001FAB0-\U0001FAB6"
    r"\U0001FAC0-\U0001FAC2\U0001FAD0-\U0001FAD6\u00A9\u00AE\u203C\u2049\u2122\u2139\u2194-\u2199"
    r"\u21A9-\u21AA\u231A-\u231B\u2328\u23CF\u23E9\u23EA-\u23F3\u23F8-\u23FA\u24C2\u25AA-\u25AB"
    r"\u25B6\u25C0\u25FB-\u25FE\u2600-\u2604\u260E\u2611\u2614-\u2615\u2618\u261D\u2620\u2622-\u2623"
    r"\u2626\u262A\u262E-\u262F\u2638-\u263A\u2640\u2642\u2648-\u2653\u265F-\u2660\u2663\u2665-\u2666"
    r"\u2668\u267B\u267E-\u267F\u2692-\u2697\u2699\u269B-\u269C\u26A0-\u26A1\u26A7\u26AA-\u26AB"
    r"\u26B0-\u26B1\u26BD-\u26BE\u26C4-\u26C5\u26C8\u26CE-\u26CF\u26D1\u26D3-\u26D4\u26E9-\u26EA"
    r"\u26F0-\u26F5\u26F7-\u26FA\u26FD\u2702\u2705\u2708-\u270D\u270F\u2712\u2714\u2716\u271D"
    r"\u2721\u2728\u2733-\u2734\u2744\u2747\u274C\u274E\u2753-\u2755\u2757\u2763-\u2764\u2795-\u2797"
    r"\u27A1\u27B0\u27BF\u2934-\u2935\u2B05-\u2B07\u2B1B-\u2B1C\u2B50\u2B55\u3030\u303D\u3297\u3299]+",
    flags=re.UNICODE)


# 判断name中是否存在emoji、数字、特殊字符
def process_name(name):
    # whether contains digit
    contains_digit = any(char.isdigit() for char in name)
    # whether contains special characters
    contains_special_characters = bool(re.search(r'[!@#\$%^&*()_+{}\[\]:;<>,.?~\\-]', name))
    # whether contains emoji
    emojis = emoji_pattern.findall(name)
    emoji_list = [char for emoji in emojis for char in emoji]
    contains_emoji = len(emoji_list) > 0
    return contains_digit, contains_special_characters, contains_emoji

# 提取emoji
def process_emoij(text):
    # get emojis
    emojis = emoji_pattern.findall(text)
    emoji_list = [char for emoji in emojis for char in emoji]
    # remove emojis
    text_without_emoji = re.sub(emoji_pattern, '', text)
    return text_without_emoji, emoji_list

# 抽取description的情感特征
def get_description_sentiment(text):
    # convert emojis to text
    text_with_descriptions = emoji.demojize(text)
    # get sentiment
    blob = TextBlob(text_with_descriptions)
    sentiment = blob.sentiment
    return sentiment.polarity, sentiment.subjectivity

# 将url转换为数字
def process_profile_background_image_url(url):
    if url is None:
        return 0
    else:
        pattern = r'theme(\d+)'
        match = re.search(pattern, url)
        if match:
            return int(match.group(1))+1
        else:
            print(url)
            return 100

# 将月份转换为数字
def month_to_number(month):
    month_dict = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }
    month_number = month_dict.get(month)
    return month_number

# 将星期转换为数字
def week_to_number(week):
    week_dict = {
        'Mon': 1,
        'Tue': 2,
        'Wed': 3,
        'Thu': 4,
        'Fri': 5,
        'Sat': 6,
        'Sun': 7
    }
    week_number = week_dict.get(week)
    return week_number

# 将语言转换为数字
def lang_to_number(lang):
    languages = ['nl', 'pt', 'en', 'tr', 'it', 'th', 'de', 'es', 'id', 
                 'en-gb', 'fr', 'ru', 'ja', 'ca', 'ar', 'pl', 'ko']
    if lang not in languages:
        idx = 17
    else:
        idx = languages.index(lang)
    return idx

# 将16进制字符串转换为10进制整数
def hex2int(hex_str):
    return int(hex_str, 16)

#利用决策树获得最优分箱的边界值列表
def optimal_binning_boundary(x, y, max_bins, min_x, max_x):
    boundary = []
    
    x = x.values  
    y = y.values
    
    clf = DecisionTreeClassifier(criterion='gini',
                                 max_leaf_nodes=max_bins,
                                 random_state=42)

    clf.fit(x.reshape(-1, 1), y)
    
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    
    for i in range(n_nodes):
        # 获得决策树节点上的划分边界值
        if children_left[i] != children_right[i]:  
            boundary.append(threshold[i])

    boundary.sort()
    # 上下界
    boundary = [min_x] + boundary + [max_x]

    return boundary




def extract_features():
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    full_dataset = {}
    for split in ['train', 'dev', 'test']:
        # read data
        file_path = f'data/{split}.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            origin_dataset = json.load(f)
        
        # process data
        new_dataset = []
        for data in tqdm(origin_dataset):
            new_data = {}

            #new_data['name'] = data['user']['name']
            #new_data['screen_name'] = data['user']['screen_name']

            # get name similarity
            name_fuzz_ratio = fuzz.partial_ratio(data['user']['name'], data['user']['screen_name'])
            if name_fuzz_ratio == 0 :
                new_data['name_fuzz_ratio'] = 0
            elif (name_fuzz_ratio>0) and (name_fuzz_ratio<=40):
                new_data['name_fuzz_ratio'] = 1
            elif (name_fuzz_ratio>40) and (name_fuzz_ratio<=70):
                new_data['name_fuzz_ratio'] = 2
            elif (name_fuzz_ratio>70) and (name_fuzz_ratio<=90):
                new_data['name_fuzz_ratio'] = 3
            else:
                new_data['name_fuzz_ratio'] = 4
            
            # check name contains digit or special_characters or emojis
            contains_digit, contains_special_characters, contains_emoji = process_name(data['user']['name'])
            new_data['name_contains_digit'] = 1 if contains_digit else 0
            new_data['name_contains_special_characters'] = 1 if contains_special_characters else 0
            new_data['name_contains_emoji'] = 1 if contains_emoji else 0

            #new_data['location'] = data['user']['location']
            #new_data['description'] = data['user']['description']

            # get description length
            description = data['user']['description']
            description, emojis = process_emoij(description)    # split emojis
            tokens = tokenizer(description, add_special_tokens=False)['input_ids']
            new_data['description_length'] = len(tokens) + len(emojis)

            # get description sentiment
            polarity, subjectivity = get_description_sentiment(description)
            new_data['description_polarity'] = polarity
            new_data['description_subjectivity'] = subjectivity

            # get url number in description
            new_data['description_url_number'] = len(data['user']['entities']['description']['urls'])

            # whetehr has url or not
            new_data['url'] = 0 if data['user']['url'] is None else 1

            new_data['followers_count'] = np.log1p(data['user']['followers_count'])
            new_data['friends_count'] = np.log1p(data['user']['friends_count'])
            new_data['listed_count'] = np.log1p(data['user']['listed_count'])

            # split time
            time_str = data['user']['created_at']
            week, month, day, time, _, year = time_str.split(' ')
            hour, minute, second = time.split(':')
            new_data['created_at_week'] = week_to_number(week)
            #new_data['created_at_year'] = int(year)
            #new_data['created_at_month'] = month_to_number(month)
            #new_data['created_at_day'] = int(day)
            new_data['created_at_hour'] = int(hour)
            #new_data['created_at_minute'] = int(minute)
            #new_data['created_at_second'] = int(second)
            # convert to timestamp
            #date_object = datetime.strptime(time_str, "%a %b %d %H:%M:%S %z %Y")
            #new_data['created_at_timestamp'] = date_object.timestamp()

            new_data['favourites_count'] = np.log1p(data['user']['favourites_count'])
            new_data['geo_enabled'] = 1 if data['user']['geo_enabled'] else 0
            new_data['verified'] = 1 if data['user']['verified'] else 0
            new_data['statuses_count'] = np.log1p(data['user']['statuses_count'])

            new_data['lang'] = lang_to_number(data['user']['lang'])

            new_data['is_translation_enabled'] = 1 if data['user']['is_translation_enabled'] else 0
            #new_data['profile_background_color'] = hex2int(data['user']['profile_background_color'])

            new_data['profile_background_image_url'] = process_profile_background_image_url(data['user']['profile_background_image_url'])
            
            new_data['profile_background_tile'] = 1 if data['user']['profile_background_tile'] else 0
            new_data['profile_banner_url'] = 1 if 'profile_banner_url' in data['user'] else 0
            #new_data['profile_link_color'] = hex2int(data['user']['profile_link_color'])
            #new_data['profile_sidebar_border_color'] = hex2int(data['user']['profile_sidebar_border_color'])
            #new_data['profile_sidebar_fill_color'] = hex2int(data['user']['profile_sidebar_fill_color'])
            #new_data['profile_text_color'] = hex2int(data['user']['profile_text_color'])
            new_data['profile_use_background_image'] = 1 if data['user']['profile_use_background_image'] else 0
            new_data['has_extended_profile'] = 1 if data['user']['has_extended_profile'] else 0
            new_data['default_profile'] = 1 if data['user']['default_profile'] else 0
            new_data['default_profile_image'] = 1 if data['user']['default_profile_image'] else 0
           
            if data['user']['translator_type'] == 'regular':
                new_data['translator_type'] = 1
            elif data['user']['translator_type'] == 'none':
                new_data['translator_type'] = 2
            elif data['user']['translator_type'] == 'badged':
                new_data['translator_type'] = 3
            else:
                new_data['translator_type'] = 0

            if data['label']=='bot':
                new_data['label'] = 1
            elif data['label']=='human':
                new_data['label'] = 0
            else:
                new_data['label'] = ''

            new_dataset.append(new_data)
        
        full_dataset[split] = new_dataset
    
    return full_dataset


def split_into_bins(full_dataset):
    # merge train, dev, test
    full_df = {}
    for split in ['train', 'dev', 'test']:
        full_df[split] = pd.DataFrame(full_dataset[split])
    all_data = pd.concat([full_df['train'], full_df['dev']], axis=0, ignore_index=True)
    all_data_with_test = pd.concat([full_df['train'], full_df['dev'], full_df['test']], axis=0, ignore_index=True)

    # split into bins
    col_name = ['description_length', 'followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count']
    for col in col_name:
        min_value = all_data_with_test[col].min()-0.1
        max_value = all_data_with_test[col].max()+0.1
        bins = optimal_binning_boundary(all_data[col], all_data['label'], 20, min_value, max_value)
        #print(col, bins)
        for split in ['train', 'dev', 'test']:
            full_df[split][col] = pd.cut(full_df[split][col], bins=bins, labels=[i for i in range(len(bins)-1)], right=True).astype(np.int64)
    
    return full_df


def feature_mining(full_df):
    st = SymbolicTransformer(
        generations=20,
        population_size=1000,
        hall_of_fame=100,
        n_components=10,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'max', 'min'],
        parsimony_coefficient='auto',
        max_samples=0.8,
        metric='spearman',
        verbose=1,
        random_state=42,
        n_jobs=16
    )
    model = RandomForestClassifier(random_state=42)
    rfe = RFE(model, n_features_to_select=32, verbose=1)

    # merge train & dev
    all_data = pd.concat([full_df['train'], full_df['dev']], axis=0, ignore_index=True)
    X_train = all_data.drop(['label'], axis=1).values.copy().astype(np.float64)
    Y_train = all_data['label'].values.copy().astype(np.int32)
    print(X_train.shape, Y_train.shape)
    # mine features by genetic programming
    X_mined_train = st.fit_transform(X_train, Y_train)
    print(X_mined_train.shape)
    X_train = np.concatenate((X_train, X_mined_train), axis=1)
    print(X_train.shape)
    # remove unimportant features
    X_train = rfe.fit_transform(X_train, Y_train) 
    print(X_train.shape)

    for split in ['train', 'dev', 'test']:
        features = full_df[split].drop(['label'], axis=1)
        labels = full_df[split]['label']
        mined_features_value = st.transform(features.values.astype(np.float64))
        mined_features = pd.DataFrame(mined_features_value, 
                                      columns=[f'mined_feature_{i}' for i in range(mined_features_value.shape[1])])
        features = pd.concat([features, mined_features], axis=1)
        features = features[features.columns[rfe.support_]]
        print(features.shape)

        # save new data
        if not os.path.exists('data_new/'):
            os.mkdir('data_new')
        final_features = pd.concat([features, labels], axis=1)
        final_features.to_csv(f'data_new/{split}.csv', index=False)


if __name__ == "__main__":
    print('Extracting features...')
    full_dataset = extract_features()
    print('Splitting into bins...')
    full_df = split_into_bins(full_dataset)
    print('Feature mining...')
    feature_mining(full_df)
