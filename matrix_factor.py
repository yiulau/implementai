import pandas as pd
import numpy
import sklearn
from sklearn.decomposition import TruncatedSVD
def get_user_latent_matrix_factorization(csv_address,output_csv_name,num_components):
    # csv containing ratings only
    ubst = pd.read_csv(csv_address)
    unique_business_id = ubst[["business_id"]]["business_id"].unique()
    unique_user_id = ubst[["user_id"]]["user_id"].unique()
    utility_matrix = numpy.zeros((len(unique_business_id), len(unique_user_id)))

    for ell in range(len(ubst)):
        print(ell)
        i = numpy.asscalar(numpy.where(unique_business_id == ubst.iloc[ell]["business_id"])[0])
        j = numpy.asscalar(numpy.where(unique_user_id == ubst.iloc[ell]["user_id"])[0])
        utility_matrix[i, j] = ubst.iloc[ell]["stars"]


    SVD = TruncatedSVD(n_components=num_components, random_state=17)
    out_matrix = SVD.fit_transform(utility_matrix)

    new_df = pd.DataFrame(out_matrix)
    df_singleton = pd.DataFrame({"user_id":unique_user_id})
    new_new_df = pd.concat([new_df, df_singleton], axis=1)
    new_new_df.to_csv('latent_factor_users.csv')
    return(out_matrix)