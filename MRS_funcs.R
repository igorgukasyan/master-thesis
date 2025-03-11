if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr, tidyr, lsa, Metrics, foreach, doParallel)
load("MRS_v1.1_input.RData")

# Creates a user utility vector
create_user_utility_vector <- function(user_id_value) {
  rownum_user <- which(user_ids == user_id_value)
  user_utility_vector <- as.matrix(na.omit(utility_matrix[rownum_user, ]))
  return(user_utility_vector)
}

# Partitions a user utility vector
create_sampled_user_utility_vector <- function(user_id_value, fold) {
  set.seed(228)
  user_utility_vector <- create_user_utility_vector(user_id_value)
  
  f <- 5
  n <- length(user_utility_vector)
  fold_indices <- sample(rep(1:f, length.out = n))
  folds <- split(seq_len(n), fold_indices)
  folds <- setNames(folds, paste0("fold", 1:f))
  
  chosen_fold <- paste0("folds$fold", fold)
  training_indices <- setdiff(1:n, eval(parse(text = chosen_fold)))
  test_indices <- eval(parse(text = chosen_fold))
  
  utility_vector_sample_train <- as.matrix(user_utility_vector[training_indices, ])
  utility_vector_sample_test <- as.matrix(user_utility_vector[test_indices, ])
  set.seed(NULL)
  return(list(train_sample = utility_vector_sample_train, test_sample = utility_vector_sample_test))
}

# Weighted cosine formula
weighted_cosine <- function(user_profile_weighed, song_profile_vector, weights_vector) {
  numerator <- sum(user_profile_weighed * song_profile_vector * weights_vector)
  weighted_user_norm <- sqrt(sum(user_profile_weighed^2 * weights_vector))
  weighted_song_norm <- sqrt(sum(song_profile_vector^2 * weights_vector))
  denominator <- weighted_user_norm * weighted_song_norm
  similarity <- numerator / denominator
  return(similarity)
}

# Creates an train item profile for a user
create_user_song_profiles.train <- function(user_id_value, fold) {
  utility_vector_sample_train <- create_sampled_user_utility_vector(user_id_value, fold)[["train_sample"]]
  rownums_songs <- which(song_ids %in% rownames(utility_vector_sample_train))
  user_song_profiles <- as.matrix(song_profiles[rownums_songs, ])
  if (length(rownums_songs) > 1){
    user_song_profiles <- user_song_profiles[rownames(utility_vector_sample_train), , drop = FALSE]
    return(user_song_profiles)
  }
  else {
    user_song_profiles <- user_song_profiles
    return(user_song_profiles)
  }
}

song_profiles_split <- function(user_id_value, fold){
  train <- create_sampled_user_utility_vector(user_id_value, fold)[["train_sample"]]
  test <- create_sampled_user_utility_vector(user_id_value, fold)[["test_sample"]]
  
  user_listened_train <- which(song_ids %in% rownames(train))
  user_listened_test <- which(song_ids %in% rownames(test))
  
  remaining_songs <- setdiff(1:nrow(song_profiles), c(user_listened_train, user_listened_test))
  
  train_indices <- c(user_listened_train, remaining_songs)
  test_indices <- c(user_listened_test, remaining_songs)
  
  train_song_profiles <- as.matrix(song_profiles[train_indices, ])
  test_song_profiles <- as.matrix(song_profiles[test_indices, ])
  
  return(list(train_song_profiles = train_song_profiles, test_song_profiles = test_song_profiles))
}

# Creates unweighted user profile from train data
create_user_profile.train <- function(user_id_value, fold) {
  user_utility_vector <- create_sampled_user_utility_vector(user_id_value, fold)[["train_sample"]]
  user_song_profiles <- create_user_song_profiles.train(user_id_value, fold)
  if (ncol(user_song_profiles) > 1) {
    user_profile <- t(user_song_profiles) %*% user_utility_vector
  } else {
    user_profile <- user_song_profiles %*% user_utility_vector
  }
  total_weight <- sum(user_utility_vector)
  if (total_weight != 0) {
    user_profile <- user_profile / total_weight
  }
  return(user_profile)
}

# Normalization function
normalize <- function(x){
  return((x-min(x))/(max(x)-min(x)))
}

# Normalization function 2
normalize_2 <- function(weights_vector) {
  sum_weights <- sum(weights_vector)
  if (sum_weights != 0) {
    return(weights_vector / sum_weights)
  } else {
    return(weights_vector)
  }
}

# Adjusting negative weights not to lose information
adjust_weights <- function(weights_vector, epsilon = 1e-4) {
  weights_vector <- ifelse(weights_vector <= 0, epsilon, weights_vector)
  return(weights_vector)
}


calculate_user_weights <- function(user_song_profiles, user_utility_vector) {
  set.seed(7)
  data <- data.frame(cbind(user_song_profiles, user_utility_vector))
  colnames(data)[ncol(data)] <- "utility"
  data$utility <- ifelse(data$utility >= mean(data$utility), 1, 0)
  data$utility <- as.factor(data$utility)
  
  # Compute class weights inversely proportional to class frequencies
  w <- table(data$utility)
  inv_weights <- as.numeric(1 / w)
  names(inv_weights) <- names(w)
  
  result <- tryCatch({
    classifier <- svm(formula = utility ~ ., 
                      data = data, 
                      type = 'C-classification', 
                      kernel = 'linear',
                      class.weights = inv_weights)
    
    # Compute weights vector from support vectors
    weights_vector <- t(t(classifier$coefs) %*% classifier$SV)
    normalize(weights_vector)
  }, error = function(e) {
    # Return 0 if any error occurs
    0
  })
  
  set.seed(NULL)
  return(result)
}

# Calculates cosine / weighted cosine distance between a user and songs from their train/test data
calculate_user_song_similarity <- function(user_id_value,
                                           apply_weights = FALSE,
                                           train_or_test = "train",
                                           fold,
                                           return_weights = FALSE) {
  sampled_data <- create_sampled_user_utility_vector(user_id_value, fold)
  user_utility_vector_train <- sampled_data[["train_sample"]]
  user_utility_vector_test <- sampled_data[["test_sample"]]
  user_profile_train <- create_user_profile.train(user_id_value, fold)
  
  if (train_or_test == "train") {
    user_utility_vector <- user_utility_vector_train
    song_profiles_sampled <- song_profiles_split(user_id_value, fold)[["train_song_profiles"]]
  } else {
    user_utility_vector <- user_utility_vector_test
    song_profiles_sampled <- song_profiles_split(user_id_value, fold)[["test_song_profiles"]]
  }
  
  song_similarity <- data.frame(song_id = rownames(song_profiles_sampled))
  
  if (!apply_weights) {
    similarities <- sapply(1:nrow(song_profiles_sampled), function(i) {
      lsa::cosine(as.vector(user_profile_train), as.vector(song_profiles_sampled[i, ]))
    })
    
    user_utility_vector <- data.frame(user_utility_vector)
    user_utility_vector$song_id <- rownames(user_utility_vector)
    
    song_similarity$cos <- similarities
    song_similarity <- merge(song_similarity, user_utility_vector, by = "song_id", all.x = TRUE, incomparables = 0)
    return(song_similarity)
  } else {
    user_song_profiles_train <- create_user_song_profiles.train(user_id_value, fold)
    weights_vector <- calculate_user_weights(user_song_profiles_train, user_utility_vector_train)
    pos_weights_columns <- which(colnames(song_profiles_sampled) %in% rownames(weights_vector))
    
    similarities <- apply(song_profiles_sampled[, pos_weights_columns], 1, function(song_profile_vector) {
      weighted_cosine(user_profile_train[pos_weights_columns, ], song_profile_vector, weights_vector)
    })
    
    user_utility_vector <- data.frame(user_utility_vector)
    user_utility_vector$song_id <- rownames(user_utility_vector)
    
    song_similarity$cos <- similarities
    song_similarity <- merge(song_similarity, user_utility_vector, by = "song_id", all.x = TRUE, incomparables = 0)
    
    if (return_weights) {
      return(list(song_similarity, weights_to_keep))
    } else {
      return(song_similarity)
    }
  }
}

# Generates recommendations
generate_recommendations <- function(user_id_value,
                                     apply_weights = FALSE,
                                     N = 10,
                                     train_or_test = "train",
                                     fold) {
  similarities <- calculate_user_song_similarity(
    user_id_value,
    apply_weights,
    train_or_test,
    fold
  )
  recommended_songs <- similarities[order(-similarities$cos), ][1:N, ]
  recommended_song_ids <- recommended_songs$song_id
  return(recommended_song_ids)
}

# Returns all songs a user liked (i.e. listened to them more than on average)
get_liked_songs <- function(user_id_value) {
  rownum_user <- which(user_ids == user_id_value)
  liked_songs <- colnames(utility_matrix)[which(utility_matrix[rownum_user, ] > mean(utility_matrix[rownum_user, ], na.rm = TRUE))]
  return(liked_songs)
}

# Return all songs a user listened to
get_streamed_songs <- function(user_id_value, train_or_test = "train", fold) {
  if(train_or_test == "train") {
    rownum_user <- which(user_ids == user_id_value)
    user_utility_vector <- create_sampled_user_utility_vector(user_id_value, fold)[["test_sample"]]
    streamed_songs <- colnames(utility_matrix)[which(!is.na(utility_matrix[rownum_user, ]))]
    streamed_songs <- setdiff(streamed_songs, rownames(user_utility_vector))
    return(streamed_songs)
  } else {
    rownum_user <- which(user_ids == user_id_value)
    user_utility_vector <- create_sampled_user_utility_vector(user_id_value, fold)[["train_sample"]]
    streamed_songs <- colnames(utility_matrix)[which(!is.na(utility_matrix[rownum_user, ]))]
    streamed_songs <- setdiff(streamed_songs, rownames(user_utility_vector))
    return(streamed_songs)
  }
}

# Calculates precision at k
calculate_precision_at_k <- function(user_id_value,
                                     apply_weights = FALSE,
                                     N = 10,
                                     train_or_test = "train",
                                     fold) {
  recommendations <- generate_recommendations(
    user_id_value,
    apply_weights,
    N,
    train_or_test,
    fold
  )
  true_likes <- get_liked_songs(user_id_value)
  TP <- length(intersect(recommendations, true_likes))
  FP <- length(setdiff(recommendations, true_likes))
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  return(precision)
}

# Calculate precision at k for all users
calculate_precision_at_k_all_users <- function(apply_weights = FALSE,
                                               N = 10,
                                               train_or_test = "train",
                                               fold) {
  precisions <- sapply(
    user_ids, calculate_precision_at_k,
    apply_weights,
    N,
    train_or_test,
    fold
  )
  mean_precision <- mean(precisions, na.rm = TRUE)
  return(mean_precision)
}

# Generates random recommendations
generate_random_recommendations <- function(user_id_value,
                                            N = 10,
                                            train_or_test = "train",
                                            fold) {
  if (train_or_test == "train") {
    user_utility_vector <- create_sampled_user_utility_vector(user_id_value, fold)[["train_sample"]]
    user_utility_vector <- data.frame(user_utility_vector)
    user_utility_vector$song_id <- rownames(user_utility_vector)
    
    random_songs <- sample(rownames(song_profiles), N, replace = FALSE)
    random_songs <- data.frame(song_id = random_songs)
    random_songs <- merge(random_songs, user_utility_vector, by = "song_id", all.x = TRUE, incomparables = 0)
    
    colnames(random_songs)[2] <- "listen_count"
    return(random_songs)
  } else if (train_or_test == "test") {
    user_utility_vector <- create_sampled_user_utility_vector(user_id_value, fold)[["test_sample"]]
    user_utility_vector <- data.frame(user_utility_vector)
    user_utility_vector$song_id <- rownames(user_utility_vector)
    
    random_songs <- sample(rownames(song_profiles), N, replace = FALSE)
    random_songs <- data.frame(song_id = random_songs)
    random_songs <- merge(random_songs, user_utility_vector, by = "song_id", all.x = TRUE, incomparables = 0)
    
    colnames(random_songs)[2] <- "listen_count"
    return(random_songs)
  }
}

# Calculates precision at k for random recommendations
calculate_precision_at_k_random <- function(user_id_value,
                                            N = 10,
                                            train_or_test = "train",
                                            fold) {
  recommendations <- rownames(generate_random_recommendations(
    user_id_value,
    N,
    train_or_test,
    fold
  ))
  true_likes <- get_liked_songs(user_id_value)
  TP <- length(intersect(recommendations, true_likes))
  FP <- length(setdiff(recommendations, true_likes))
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0)
  return(precision)
}

# Calculates precision at k for all users based on random recommendations
calculate_precision_at_k_random_all_users <- function(N = 10,
                                                      train_or_test = "train",
                                                      fold) {
  precisions <- sapply(user_ids, function(user_id) {
    calculate_precision_at_k_random(user_id, N, train_or_test, fold)
  })
  
  mean_precision <- mean(precisions, na.rm = TRUE)
  return(mean_precision)
}

# Calculates ndcg at k for a user
calculate_ndcg_at_k <- function(user_id_value,
                                apply_weights = FALSE,
                                N = 10,
                                train_or_test = "train",
                                fold) {
  song_similarities <- calculate_user_song_similarity(
    user_id_value,
    apply_weights,
    train_or_test,
    fold
  )
  song_similarities_ordered_ideal <- song_similarities[order(-song_similarities$listen_count), ][1:N, ]
  song_similarities_ordered_recommended <- song_similarities[order(-song_similarities$cos), ][1:N, ]
  
  denominator <- log2(1L:N + 1L)
  DCG <- sum(song_similarities_ordered_recommended$listen_count[1L:N] / denominator)
  IDCG <- sum(song_similarities_ordered_ideal$listen_count[1L:N] / denominator)
  nDCG <- DCG / IDCG
  return(nDCG)
}

# Calculates ndcg at k for all users
calculate_ndcg_at_k_all_users <- function(apply_weights = FALSE,
                                          N = 10,
                                          train_or_test = "train",
                                          fold) {
  nDCGs <- sapply(
    user_ids, calculate_ndcg_at_k,
    apply_weights,
    N,
    train_or_test,
    fold
  )
  mean_NDCG <- mean(nDCGs, na.rm = TRUE)
  return(mean_NDCG)
}

# Calculates ndcg at k for a user based on random recommendations
calculate_ndcg_at_k_random <- function(user_id_value,
                                       N = 10,
                                       train_or_test = "train",
                                       fold) {
  recommendations_ordered_recommended <- generate_random_recommendations(
    user_id_value,
    N,
    train_or_test,
    fold
  )
  song_similarities <- calculate_user_song_similarity(user_id_value)
  recommendations_ordered_ideal <- recommendations_ordered_recommended[order(-recommendations_ordered_recommended$listen_count), ][1:N]
  song_similarities_ordered_ideal <- song_similarities[order(-song_similarities$listen_count), ][1:N, ]
  
  denominator <- log2(1L:N + 1L)
  DCG <- sum(recommendations_ordered_recommended$listen_count[1L:N] / denominator)
  IDCG <- sum(song_similarities_ordered_ideal$listen_count[1L:N] / denominator)
  nDCG <- DCG / IDCG
  return(nDCG)
}

# Calculates ndcg at k for all users based on random recommendations
calculate_ndcg_at_k_random_all_users <- function(N = 10,
                                                 train_or_test = "train",
                                                 fold) {
  nDCGs <- sapply(
    user_ids, calculate_ndcg_at_k_random,
    N,
    train_or_test,
    fold
  )
  mean_NDCG <- mean(nDCGs, na.rm = TRUE)
  return(mean_NDCG)
}

# APK for a user

calc_apk <- function(user_id_value,
                     apply_weights = FALSE,
                     k = 500,
                     train_or_test = "train",
                     fold) {
  recommendations <- generate_recommendations(
    user_id_value,
    apply_weights,
    N = k,
    train_or_test,
    fold
  )
  streamed_songs <- get_streamed_songs(user_id_value, train_or_test, fold)
  average_precision <- apk(k, actual = streamed_songs, predicted = recommendations)
  return(average_precision)
}

# APK random
calc_apk_random <- function(user_id_value,
                            apply_weights = FALSE,
                            k = 500,
                            train_or_test = "train",
                            fold) {
  recommendations <- generate_random_recommendations(
    user_id_value,
    N = k,
    train_or_test,
    fold
  )
  streamed_songs <- get_streamed_songs(user_id_value, train_or_test, fold)
  average_precision <- apk(k, actual = streamed_songs, predicted = recommendations)
  return(average_precision)
}

# mAP@k for all users
mapk_all <- function(apply_weights = FALSE, k = 500, 
                     train_or_test = "train", fold){
  cl <- makeCluster(6)
  on.exit(stopCluster(cl))
  clusterExport(cl, c("calc_apk", "user_ids", "utility_matrix", "song_profiles",
                      "create_sampled_user_utility_vector", "create_user_profile.train",
                      "get_streamed_songs", "generate_recommendations", "calculate_user_song_similarity",
                      "create_user_utility_vector", "create_user_song_profiles.train",
                      "song_ids", "weighted_cosine", "normalize",
                      "normalize_2", "adjust_weights", "song_profiles_split", "calculate_user_weights"))
  
  apks <- parSapply(cl, user_ids, function(user_id) {
    calc_apk(user_id, apply_weights, k, train_or_test, fold)
  })
  
  mapk_value <- mean(apks, na.rm = TRUE)
  return(mapk_value)
}

# pecision@k for all users
calculate_precision_at_k_all_users <- function(apply_weights = FALSE,
                                               N = 10,
                                               train_or_test = "train",
                                               fold) {
  precisions <- sapply(
    user_ids, calculate_precision_at_k,
    apply_weights,
    N,
    train_or_test,
    fold
  )
  mean_precision <- mean(precisions, na.rm = TRUE)
  return(mean_precision)
}

# mapk random
mapk_all_random <- function(apply_weights = FALSE, k = 500, 
                            train_or_test = "train", fold){
  
  apks <- sapply(user_ids, calc_apk_random, 
                 apply_weights,
                 k,
                 train_or_test,
                 fold
  )
  mapk_value <- mean(apks, na.rm = TRUE)
  return(mapk_value)
}

# apk popularity
calc_apk_pop <- function(user_id_value,
                         k = 500,
                         train_or_test = "train",
                         fold) {
  recommendations <- as.vector(unlist(song_listen_count[1:k, 1]))
  streamed_songs <- get_streamed_songs(user_id_value, train_or_test, fold)
  average_precision <- apk(k, actual = streamed_songs, predicted = recommendations)
  return(average_precision)
}

# mapk popularity
mapk_all_pop <- function(k = 500, train_or_test = "train", fold){
  apks <- sapply(user_ids, calc_apk_pop, k, train_or_test, fold)
  mapk_value <- mean(apks, na.rm = TRUE)
  return(mapk_value)
  
}

#Creates a partitioned utility matrix (not used)
create_sample_utility_matrix <- function(fold){
  sample_utility_vectors_all_users <- lapply(user_ids, create_sampled_user_utility_vector, 
                                             fold)
  
  all_song_ids <- unique(unlist(lapply(sample_utility_vectors_all_users, names)))
  
  sample_utility_matrix <- matrix(0, nrow = length(user_ids), ncol = length(all_song_ids))
  colnames(sample_utility_matrix) <- all_song_ids
  rownames(sample_utility_matrix) <- user_ids
  
  for (i in seq_along(user_ids)) {
    user_vector <- sample_utility_vectors_all_users[[i]]
    sample_utility_matrix[i, names(user_vector)] <- user_vector
  }
  sample_utility_matrix[sample_utility_matrix == 0] <- NA
  return(sample_utility_matrix)
}

song_profiles <- scale(song_profiles)

song_count <- data.frame(user_id = as.character(),
                         num_song = as.numeric())
# Remove users with less than 5 songs streamed
for (i in(1:length(user_ids))){
  song_count[i, 1] <- user_ids[i]
  song_count[i, 2] <- length(create_user_utility_vector(user_ids[i]))
}

remove_users <- song_count[song_count$num_song < 5, 1]
utility_matrix <- utility_matrix[-which(rownames(utility_matrix) %in% remove_users), ]
user_ids <- user_ids[-which(user_ids %in% remove_users)]
