###                 Text Mining and Graph Application on SFR Tweets
###                                   April 2019

###               Khanh TRUONG - Mickaël OLIFANT - Clarisse BOINAY

# Import packages --------------------------------------------------------------
library(knitr)       # used to make kable tables
library(tm)          # text mining package
library(magrittr)    # allows pipe operator
library(tidytext)    # tidy text for plots
library(ggplot2)     # used for plots
library(gridExtra)   # arrange plots
library(factoextra)  # used for plots PCA
library(NbClust)     # optimal k in kmeans
library(dplyr)       # Manipulate data frames
library(usedist)     # calcualate pairwise similarity
library(reshape2)    # for reshape similarity matrix

theme_update(plot.title = element_text(hjust = 0.5)) #adjust the tittle of plots


#fFr the graph part :
library(RColorBrewer)
library(microbenchmark)
library(igraph)


# Import files -----------------------------------------------------------------
temp = list.files("Parsing") # all files in folder Parsing
temp = paste0('Parsing/', temp) # add Parsing/ before the names' files
tweet_list = lapply(temp, read.delim, encoding='UTF-8') # import all the files
summary(tweet_list)
rm(temp)

# Merge data
tweet = data.frame()
for (i in 1:length(tweet_list)){
  
  tweet_i = data.frame(tweet_list[i])[c('text', 'lang',
                                        'entities_user_mentions_name')]
  tweet = rbind(tweet, tweet_i)
  
  rm(i, tweet_i)
}

# Take only sfr French tweets
tweet_fr = tweet[tweet$lang=='fr', ]
sfr = tweet_fr[grep("sfr",
                    tweet_fr$entities_user_mentions_name,
                    ignore.case=TRUE), ]
dim(sfr)


# Clean text data --------------------------------------------------------------
str(sfr$text) # data type of column 'text' is factor
sfr$text = as.character(sfr$text) # change to character
str(sfr$text) # check again

# Use regular expression to clean text

sfr = sfr[!grepl("^RT", sfr$text, ignore.case = TRUE),] #remove retweeted tweets
rownames(sfr) <- NULL # reset index

old = sfr[17, ]$text # take an example tweet

sfr$text <- gsub("#\\w+ *", "", sfr$text) # remove hash-tag (start by "#")
old; sfr[17, ]$text

sfr$text <- gsub("@\\w+ *", "", sfr$text) # remove entities (start by "@")
old; sfr[17, ]$text

sfr$text <- gsub('http\\S+\\s*', '', sfr$text) # remove link (start by "http")
old; sfr[17, ]$text

rm(old)


# Corpus Transformations -------------------------------------------------------
sfr_corpus <- VCorpus(VectorSource(sfr$text)) # transform in to corpus
sfr_corpus

getTransformations() # all transformations that tm supports

# take an example tweet to see effects of transformation
old = strwrap(sfr_corpus[[8]])

# remove punctuation
sfr_corpus <- tm_map(sfr_corpus, removePunctuation)
old; strwrap(sfr_corpus[[8]])

# remove number
sfr_corpus <- tm_map(sfr_corpus, removeNumbers)
old; strwrap(sfr_corpus[[8]])

# lowercase
sfr_corpus <- tm_map(sfr_corpus, content_transformer(tolower))
old; strwrap(sfr_corpus[[8]])

# remove stopwords
stop_word_add = c('sfr', 'chez', 'cest', 'plus', 'via', 'jai', 'merci',
                  'bonjour', 'alor', 'aussi', 'avant', 'aprè', 'tous' ,'fai')
sfr_corpus <- tm_map(sfr_corpus, removeWords,
                     c(stopwords("french"), stop_word_add))
old; strwrap(sfr_corpus[[8]])

# stemming
sfr_corpus <- tm_map(sfr_corpus, stemDocument, language = 'french')
old; strwrap(sfr_corpus[[8]])


# strip extra whitespace
sfr_corpus <- tm_map(sfr_corpus, stripWhitespace)
old; strwrap(sfr_corpus[[8]])

rm(old)


# Document Term Matrix ---------------------------------------------------------
# Coerces into a Document Term Matrix
# At the same time, remove words that appear in fewer than 1% of the data
# Those words may be special characters that
# haven't cleaned yet in corpus transformation
sfr_dtm <- DocumentTermMatrix(
  sfr_corpus,
  control=list(bounds = list(global = c(0.01*nrow(sfr), Inf))))

rowTotals <- apply(sfr_dtm, 1, sum) # sum of remaining words in each tweet
sfr_dtm <- sfr_dtm[rowTotals > 0, ] # remove all tweets without words
rm(rowTotals)

dim(sfr_dtm)
inspect(sfr_dtm[10:15, 100:105]) # inspects


# Word Frequency ---------------------------------------------------------------
# Sum all columns(words) to get frequency
words_frequency <- colSums(as.matrix(sfr_dtm))
ord <- order(words_frequency, decreasing=TRUE)

# get the top 10 words by frequency of appearance
words_frequency[head(ord, 10)] %>% 
  kable()

# Find words that are most correlated with "client"
findAssocs(sfr_dtm, "client", .03) %>%
  kable()

# Find words that are most correlated with "servic"
findAssocs(sfr_dtm, "servic", .04) %>%
  kable()

# Find words that are most correlated with "réseau"
findAssocs(sfr_dtm, "réseau", 0.04) %>%
  kable()

# Find words that are most correlated with "box"
findAssocs(sfr_dtm, "box", 0.05) %>%
  kable()

# Find words that are most correlated with "fibr"
findAssocs(sfr_dtm, "fibr", 0.03) %>%
  kable()

# Find words that are most correlated with "téléphone"
findAssocs(sfr_dtm, "téléphone", 0.03) %>%
  kable()

# Find words that are most correlated with "mobil"
findAssocs(sfr_dtm, "mobil", 0.05) %>%
  kable()


# TF:IDF -----------------------------------------------------------------------
# convert our "clean" corpus into a tfidf weighted dtm
sfr_dtm_tfidf = DocumentTermMatrix(
  sfr_corpus,
  control=list(bounds = list(global = c(0.01*nrow(sfr),Inf)),
               weighting = weightTfIdf))

# sum of remaining words in each tweet
rowTotals_tfidf <- apply(sfr_dtm_tfidf, 1, sum)

# remove all tweets without words
sfr_dtm_tfidf <- sfr_dtm_tfidf[rowTotals_tfidf > 0,]
rm(rowTotals_tfidf)

dim(sfr_dtm_tfidf)
inspect(sfr_dtm_tfidf[1:5, 1:5]) # inspects


# Convert to data frame --------------------------------------------------------
sfr_tf = as.data.frame(as.matrix(sfr_dtm))
sfr_tfidf = as.data.frame(as.matrix(sfr_dtm_tfidf))


# K-mean clustering ------------------------------------------------------------
# Find optimal k
## Elbow method
set.seed(1234)
elbow = fviz_nbclust(sfr_tfidf, kmeans, method = "wss") +
  geom_vline(xintercept = 6, linetype = 2, color='blue')+
  labs(subtitle = NULL)
elbow
optimal_k = 6

# Perform clustering with optimal k
set.seed(1234)
cluster_kmean <- kmeans(sfr_tfidf, optimal_k)

cluster_kmean$size
cluster_kmean$size/nrow(sfr_tfidf)

# Visualize on PCA plane
pca = prcomp(sfr_tfidf, center=FALSE, scale = FALSE)
fviz_eig(pca)


ggplot() +
  geom_point(aes(x = pca$x[,1],
                 y = pca$x[,2],
                 color = as.factor(cluster_kmean$cluster))) +
  labs(x='First Principal Component', y='Second Principal Component',
       title ='K-Means Clustering') +
  guides(color=guide_legend(title="Cluster"))


# Topic of each cluster --------------------------------------------------------

# add cluster to sfr_tfidf
sfr_tfidf['cluster'] <- as.factor(cluster_kmean$cluster)

# for each cluster, calculate every column mean
# the mean of columns represent how 'important' and 'representing' of the word
topic = sfr_tfidf %>%
  group_by(cluster) %>%
  summarise_all(mean)
sfr_tfidf['cluster'] <- NULL # give back the original sfr_tfidf

# 6 lines of code below are for data transformation, no need to read, just run
topic = as.data.frame(t(topic))
colnames(topic) <- c('cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5',
                     'cluster6')
topic = topic[-1, ]
topic['word'] <- rownames(topic)
topic[names(topic)!='word'] = data.frame(apply(topic[names(topic)!='word'],
                                               2,
                                               as.numeric))
rownames(topic) <- NULL # reset index

head(topic) # each row: average tf-idf score of a word in each cluster

# For each cluster, take 5 words have the highest tf-ipair_dist_list scores.
topic = data.frame(
  cluster1=topic[order(-topic$cluster1), ][1:5, 'word'],
  cluster2=topic[order(-topic$cluster2), ][1:5, 'word'],
  cluster3=topic[order(-topic$cluster3), ][1:5, 'word'],
  cluster4=topic[order(-topic$cluster4), ][1:5, 'word'],
  cluster5=topic[order(-topic$cluster5), ][1:5, 'word'],
  cluster6=topic[order(-topic$cluster6), ][1:5, 'word']
)
topic


# Louvain algorithm ------------------------------------------------------------
# create a functioin calculate the similarity score between 2 nodes
# it is equal to sum of all (min value in every column)
similar_score <- function (v1, v2) sum(mapply(min, v1, v2))

# it takes about 30 mins to execute the dist_make() function
start.time <- Sys.time()

# due to limited resources (not powerful private laptop)
# we analyze 1000 tweets (representing 1000 nodes), instead of ~5,000 tweets
n = 1000
set.seed(1234)
sfr_tfidf_sub = sample_n(sfr_tfidf, n)

pair_dist = dist_make(sfr_tfidf_sub,
                      similar_score, method = NULL)
end.time <- Sys.time()
time.taken <- end.time - start.time
time.taken # time of execution
rm(time.taken)

# Transform the pairwise similarity into edges list
pair_dist_list = melt(as.matrix(pair_dist))

# remove edges that have value 0
# value 0 indicates ther is no common words between 2 tweets
pair_dist_list = pair_dist_list[pair_dist_list$value>0, ]


# Create the graph object
sfr_net = graph_from_edgelist(as.matrix(pair_dist_list[,1:2]), directed = FALSE)
E(sfr_net)$weight <- pair_dist_list[ ,'value'] # add weight to the graph
sfr_net
vcount(sfr_net)
ecount(sfr_net)
is.connected(sfr_net)

# Descriptive statistics
head(V(sfr_net))
head(E(sfr_net))
edge_density(sfr_net, loops=FALSE)
transitivity(sfr_net)
diameter(sfr_net, weights = pair_dist_list[, 'value'])
radius(sfr_net)
girth(sfr_net)

hist(pair_dist_list[, 'value'], main = "Histogram of weights", xlab = "Weights")
summary(pair_dist_list[,3])

# Louvain clustering
sfr_louvain <- cluster_louvain(sfr_net, weights = pair_dist_list[, 'value'])
modularity(sfr_louvain)
sizes(sfr_louvain)
modularity(sfr_louvain)


# Topic of each cluster in Louvain ---------------------------------------------

# add cluster to sfr_tfidf
sfr_tfidf_sub['cluster'] <- as.factor(sfr_louvain$membership)

# for each cluster, calculate every column mean
# the mean of columns represent how 'important' and 'representing' of the word
topic_louvain = sfr_tfidf_sub %>%
  group_by(cluster) %>%
  summarise_all(mean)
sfr_tfidf_sub['cluster'] <- NULL # give back the original sfr_tfidf

# 6 lines of code below are for data transformation, no need to read, just run
topic_louvain = as.data.frame(t(topic_louvain))
colnames(topic_louvain) <-
  c('cluster1', 'cluster2', 'cluster3', 'cluster4', 'cluster5', 'cluster6',
    'cluster7', 'cluster8', 'cluster9', 'cluster10', 'cluster11', 'cluster12')

topic_louvain = topic_louvain[-1, ]
topic_louvain['word'] <- rownames(topic_louvain)
topic_louvain[names(topic_louvain)!='word'] <-
  data.frame(apply(topic_louvain[names(topic_louvain)!='word'],
                   2,
                   as.numeric))
rownames(topic_louvain) <- NULL # reset index

# each row: average tf-ipair_dist_list score of a word in each cluster
head(topic_louvain)

# For each cluster, take 5 words have the highest tf-ipair_dist_list scores.
topic_louvain = data.frame(
  cluster1=topic_louvain[order(-topic_louvain$cluster1), ][1:5, 'word'],
  cluster2=topic_louvain[order(-topic_louvain$cluster2), ][1:5, 'word'],
  cluster3=topic_louvain[order(-topic_louvain$cluster3), ][1:5, 'word'],
  cluster4=topic_louvain[order(-topic_louvain$cluster4), ][1:5, 'word'],
  cluster5=topic_louvain[order(-topic_louvain$cluster5), ][1:5, 'word'],
  cluster6=topic_louvain[order(-topic_louvain$cluster6), ][1:5, 'word'],
  cluster7=topic_louvain[order(-topic_louvain$cluster7), ][1:5, 'word'],
  cluster8=topic_louvain[order(-topic_louvain$cluster8), ][1:5, 'word'],
  cluster9=topic_louvain[order(-topic_louvain$cluster9), ][1:5, 'word'],
  cluster10=topic_louvain[order(-topic_louvain$cluster10), ][1:5, 'word'],
  cluster11=topic_louvain[order(-topic_louvain$cluster11), ][1:5, 'word'],
  cluster12=topic_louvain[order(-topic_louvain$cluster12), ][1:5, 'word']
)
topic_louvain