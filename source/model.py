import ast
from sys import argv
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import accuracy_score
from numpy import mean
from numpy import std



def classify(file, dict1):
    all_keys = [key[1] for key in dict1.keys()]
    #file = file.drop(file.loc[file['Start'] != all_keys[0]], axis=0)
    #file = file.loc[file['Start'] == all_keys[0]]

    # Drop everything but the selected regions
    for idx, row in file.iterrows():
        if row['Start'] not in all_keys:
            file.drop(idx, inplace=True)

    #file.to_csv('Train_117.csv')

    file = file.transpose()

    # Drop the first 4 columns eg. Chromosome, Start, etc.
    file.drop(['Chromosome', 'Start', 'End', 'Nclone'], inplace=True)

    with open('Train_clinical', 'r') as f:
        f = pd.read_csv(f, delimiter='\t')

    labels = np.array(f['Subgroup'])
    features = np.array(file)
    features_list = list(file.columns)

    X = features
    y = labels

    # configure the cross-validation procedure
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=42)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        # split data
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # configure the cross-validation procedure
        cv_inner = KFold(n_splits=4, shuffle=True, random_state=42)
        # define the model
        model = RandomForestClassifier(random_state=42)
        # define search space
        space = dict()
        space['n_estimators'] = [10, 100, 500]
        space['max_features'] = [10, 20, 50]
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X_train, y_train)
        # get the best performing model fit on the whole training set
        best_model = result.best_estimator_
        # evaluate model on the hold out dataset
        yhat = best_model.predict(X_test)
        # evaluate the model
        acc = accuracy_score(y_test, yhat)
        # store the result
        outer_results.append(acc)
        # report progress
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    # summarize the estimated performance of the model
    print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))




def pre_process(file, file2):

    flag = False
    if flag:
        # store all region of the genome into a dict
        dict1 = {}
        for chrm, start, end in zip(file.Chromosome,file.Start, file.End):
            dict1[chrm,start,end] = []


        # map all genes to the genome
        for idx, row in file2.iterrows():
            for key in dict1.keys():
                if int(row['Chromosome']) == key[0] and row['Gene_start'] >= key[1] and row['Gene_end'] <= key[2]:
                    dict1[key].append(row['ENSEMBL_gene_id'])

        with open('gene_map.txt', 'w') as output:
            output.write(str(dict1))


    flag = True
    if flag:

        with open('gene_map.txt', 'r') as f2:
            f2 = f2.read()

        dict1 = ast.literal_eval(f2)

        with open('gene_list.txt', 'r') as f:
            gene_list = f.readlines()
            gene_list = [line.strip() for line in gene_list]


        dict2 = {}
        not_in_value = 0
        for key, value in dict1.items():
            for gene in gene_list:

                if gene in value:

                    if key not in dict2.keys():
                        dict2[key] = [gene]
                    else:
                        dict2[key].append(gene)

                else:
                    not_in_value += 1

        print('Length of dict1 keys %i:' % len(dict1.keys()))
        print('Length of dic2 keys %i:' % len(dict2.keys()))

    flag = False
    if flag:
        f = open("final_regions.txt", "w")
        f.write("{\n")
        for k in dict2.keys():
            f.write("'{}':'{}'\n".format(k, dict2[k]))
        f.write("}")
        f.close()

    return classify(file, dict2)



def pca(file):                                                  # doesn't work
    pca = PCA(n_components=100)

    file.drop(['Chromosome', 'Start', 'End', 'Nclone'], axis=1, inplace=True)
    file = file.transpose()
    components = pca.fit_transform(file)
    pca_df = pca.explained_variance_ratio_.cumsum()
    print(pca_df)
    pass


def main():
    f_in = argv[1]
    f_in2 = argv[2]

    file = pd.read_csv(f_in, delimiter='\t')
    file2 = pd.read_csv(f_in2, delimiter='\t')

    for idx, row in file2.iterrows():
        if row['Chromosome'] == 'X' or row['Chromosome'] == 'Y':
            file2.drop(idx, axis=0, inplace=True)

    # call functions
    pre_process(file, file2)
    # pca(file)



if __name__ == '__main__':
    main()






