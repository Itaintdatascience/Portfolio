#Michael Frasco
#In response to the following article from Business Insider
#http://www.businessinsider.com/the-most-financially-healthy-states-2014-7

#The code used to perform this analysis is at the bottom.

#When I read the article I believed that the author had made a very poor choice in selecting the variables that were used to measure the financial health of the fifty states. To me, three of the four variables seemed highly correlated: median household income, percent of total population in poverty in the past twelve months, and GDP per capita. The fourth variable included in the analysis was bankruptcy filings per 1,000 people.

#The first three variables listed are intuitively correlated. A state with high median household income is also going to have a lower percentage of people in poverty. The first three variables all seemed to measure how wealthy a state is. My intuition was confirmed by that data when I looked at my home state's statistics and saw that Connecticut ranked fifth in each of the wealth metrics. The bankruptcy variable could be independent of the wealth variables. For example poor state that made financially responsible decisions could have a low bankruptcy rate.

#Including three correlated variables in his analysis makes the author put too much weight on a state's wealth. Now, the author's variable selection would be fine if his aim was to provide a list of the wealthiest states. However, there is more to the financial health of a state than the wealth of its citizens. We could look at various metrics of government debt, unemployment, or inflation, just to name a few. I am not saying that I know the best collection of variables to choose. I am saying that the author of this article chose redundant variables that skew his results toward wealthy states.

#I am posting my analysis as an R script on Github. The link to the repository is here:

#To perform this analysis, I went to statesperform.org and downloaded the exact same data set that the author displayed in the article. After cleaning the data, I calculated a correlation matrix between the four variables. I found that Median_Income had correlations of -.218, -.809, and .611 with Bankruptcy_Filings, Percent_Poverty, and GDP_Per_Capita. Bankruptcy_Filings had correlations of .215 and -.306 with Pecent_Poverty and GDP_Per_Capita. Percent_Poverty and GDP_Per Capita had a correlation of -.536.

#What these numbers signify is that the three wealth variables are correlated fairly well and the the Bankruptcy variable is uncorrelated with the other three variables.

#Expecting the correlations to be higher, I realized that the author ranked all fifty states in each variable. So I calculated the rank variables and created another correlation matrix. As a result, the correlations between the wealth variables strengthened.

#I found that Ranked_Median_Income had correlations of .223, .832, and .701 with Bankruptcy_Filings, Percent_Poverty, and GDP_Per_Capita. Bankruptcy_Filings had correlations of .252 and .283 with Pecent_Poverty and GDP_Per_Capita. Percent_Poverty and GDP_Per Capita had a correlation of .618.

#Here you can see that the correlations between the wealth variables are strong.

#The last analysis that I wanted to perform was Principal Component Analysis. This is a technique from linear algebra that aims to reduce the number of dimensions in a high dimensional data set. It does this by finding redundancy in correlated variables. As a result of the analysis, we find the vector that contains that highest amount of variability through the data set.

#The result of principal component analysis that I want to present here is a biplot of the observations and the four variable vectors. From this biplot we can see that the three wealth variables point in the same direction and are of approximately equal length. This means that the three variables convey the same information. For example, two uncorrelated variables would be perpendicular on a biplot.

#Ultimately, the biplot and the correlation matrix shows that three of the four variables that the author chose to measure the financial health of the fifty states are redundant. As a result of the author's decision to add up the ranks of all fifty states in each of the four categories, the final rankings are skewed towards states that are wealthy (i.e. that do well in those three correlated categories).

#We can improve the author's analysis by eliminating two of the three wealth metrics. This would imply that financial health can be determined by how wealthy the state is and how few bankruptcies it has. We could look to add more variables to capture information on unemployment or debt.

data <- read.csv("states_perform_BIdata.csv")

data <- data[1:50,]

data1 <- as.character(data[,1])

data2_character <- as.character(data[,2])
data2_cleaned <- gsub("(\\$|,)", "", data2_character)
data2_numeric <- as.numeric(data2_cleaned)

data3 <- as.numeric(as.character(data[,3]))

data4_character <- as.character(data[,4])
data4_cleaned <- gsub(" %", "", data4_character)
data4_numeric <- as.numeric(data4_cleaned)

data5_character <- as.character(data[,5])
data5_cleaned <- gsub("(\\$|,)", "", data5_character)
data5_numeric <- as.numeric(data5_cleaned)

data_frame <- data.frame(data1, data2_numeric, data3, data4_numeric, data5_numeric)

names(data_frame) <- c("State", "Median_Income", "Bankruptcy_Filings", "Percent_Poverty", "GDP_Per_Capita")

numeric_data <- data_frame[,2:5]

cor(numeric_data)

ranked1 <- rank(-numeric_data[,1])
ranked2 <- rank(numeric_data[,2])
ranked3 <- rank(numeric_data[,3])
ranked4 <- rank(-numeric_data[,4])

rank_frame <- data.frame(ranked1, ranked2, ranked3, ranked4)
names(rank_frame) <- c("Median_Income", "Bankruptcy_Filings", "Percent_Poverty", "GDP_Per_Capita")

cor(rank_frame)

fit <- princomp(numeric_data, cor = TRUE)
summary(fit)
loadings(fit)
plot(fit, type='lines')
fit$scores
biplot(fit)

fit2 <- princomp(rank_frame, cor = TRUE)
summary(fit2)
loadings(fit2)
plot(fit2, type='lines')
fit2$scores
biplot(fit2)