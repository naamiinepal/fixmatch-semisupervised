mkdir -p ./isic_challenge
wget "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_Data.zip" -P ./isic_challenge 
unzip ./isic_challenge/ISBI2016_ISIC_Part3B_Training_Data.zip -d ./isic_challenge
wget "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Training_GroundTruth.csv" -P ./isic_challenge 
wget "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Test_Data.zip" -P ./isic_challenge 
unzip ./isic_challenge/ISBI2016_ISIC_Part3B_Test_Data.zip -d ./isic_challenge
wget "https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part3B_Test_GroundTruth.csv" -P ./isic_challenge 