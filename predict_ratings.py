import numpy
def predict_ratings(latent_probability_matrix,latent_user_factor_matrix,train_predicts):

    num_users = latent_user_factor_matrix.shape[0]
    num_restaurants = latent_probability_matrix[0]
    user_offset = numpy.zeros(num_users)
    restaurant_offset = numpy.zeros(num_restaurants)

    for i in range(num_users):
        temp = 0
        count = 0
        latent_user = latent_user_factor_matrix[i,:]
        for j in range(num_restaurants):
            if train_predicts[i,j]!=0:
                latent_restaurant = latent_probability_matrix[j,:]
                temp += train_predicts[i,j] - numpy.dot(latent_user,latent_restaurant)
                count += 1
        user_offset[i] = temp/(count-1)

    for j in range(num_restaurants):
        temp = 0
        count = 0
        latent_restaurant = latent_user_factor_matrix[j,:]
        for i in range(num_users):
            if train_predicts[i,j]!=0:
                latent_user = latent_probability_matrix[i,:]
                temp += train_predicts[i,j] - numpy.dot(latent_user,latent_restaurant)
                count += 1
        restaurant_offset[i] = temp/(count-1)


    return(user_offset,restaurant_offset)


def generate_predictions(latent_probability_matrix,latent_user_factor_matrix,user_offset,restaurant_offset):

    predicted_ratings = numpy.zeros((len(user_offset),len(restaurant_offset)))
    for i in range(len(user_offset)):
        latent_user = latent_user_factor_matrix[i,:]
        for j in range(len(restaurant_offset)):
            predicted_ratings[i,j] = numpy.dot(latent_user,latent_probability_matrix[j,:]) + restaurant_offset[j] + user_offset[i]


    return(predicted_ratings)

