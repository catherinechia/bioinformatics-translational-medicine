import json
from sys import argv
import pandas as pd
from sklearn.decomposition import PCA
import json


def pre_process(file, file2):
    #file.drop(['Chromosome', 'Start', 'End', 'Nclone'], axis=1, inplace=True)
    #print(file.head())

    #print(file['Start'])

    flag = True

    if flag:
        # store all region of the genome into a dict
        dict1 = {}
        for start, end in zip(file.Start, file.End):
            dict1[start,end] = []

        # map all genes to the genome
        for idx, row in file2.iterrows():
            for key in dict1.keys():
                if row['Gene_start'] >= key[0] and row['Gene_end'] <= key[1]:
                    dict1[key].append(row['ENSEMBL_gene_id'])


        #print(dict1)

        #with open('region_to_gene.txt', 'w') as output:
        #    output.write(json.dumps(dict1))


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
        print(dict2)
        print(not_in_value)









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
    print(file2.head())

    # call functions
    pre_process(file, file2)
    # pca(file)




if __name__ == '__main__':
    main()






