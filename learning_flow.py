from project_utils import *

"""read train data from csv file, change some of the names to be
readable by further functions.
get the important words from file and create a bag of words presentation of the data.
learn 3 classification models and save it to files."""


def flow():
    all_data = pd.read_csv('Classified_Data_kaggle.csv', engine='python')
    all_data = all_data.replace("FAKE", '0')
    all_data = all_data.replace("REAL", '1')
    all_data = all_data.rename(index=str, columns={"text": "Body","label":'Label (1=true)'})
    data = all_data.loc[all_data['Label (1=true)'].isin(['0','1'])]
    print(data.shape)
    words = read_json("top_1000_imp_words.json")
    words = words[:1000]
    df_with_imp_words = build_table_form_words(data,words)
    df_with_imp_words = df_with_imp_words.replace(np.nan, 0)
    X = df_with_imp_words.iloc[:,6:]
    Y_train = df_with_imp_words['Label (1=true)']
    nn_clf = nn_func(X, Y_train)
    lr_clf = lr_func(X, Y_train)
    svm_clf = svm_func(X,Y_train)
    joblib.dump(svm_clf, 'svm_clf.pkl')
    joblib.dump(nn_clf, 'nn_clf.pkl')
    joblib.dump(lr_clf, 'lr_clf.pkl')


def main():
    flow()


if __name__ == "__main__":
    main()
