library(tidyverse)
library(janitor)

outcome = read.table('/Users/shangyimeng/2022Fall/datamining/gene_expression_sample/GEUVADIS_normalized_expression_chr20',header = T) 
select = 1
start = outcome$start[select]
end = outcome$end[select]
outcome = outcome %>% select(starts_with("HG"), starts_with("NA")) 
y = outcome[select,] %>% t() 

df = read.table('/Users/shangyimeng/2022Fall/datamining/gene_expression_sample/GEUVADIS_chr20_processed.traw',header = T) 
df = df %>% filter(POS >= start-500000) %>% filter(POS <= end + 500000) %>% as.data.frame()
df = df %>% select(SNP, starts_with("HG"), starts_with("NA")) %>% t()
df = df %>% row_to_names(row_number = 1) 
rownames(df) = sub('.*_', '', rownames(df))

data = merge(y, df,by="row.names",all.x=TRUE) 
data = data %>% column_to_rownames(var="Row.names")  
colnames(data)[1] = 'outcome'

library(glmnet)
X = data[,2:dim(data)[2]] %>% as.data.frame()
y=data$outcome
normalize = function(x) {
  x = as.numeric(x)
  x = x/sqrt(sum(x*x))
  return(x)
}

norm_X = apply(X,2,normalize)
fit <- glmnet(x=norm_X, y=y, lambda = 0.1, alpha = 1, 
              standardize = FALSE, intercept = FALSE)
beta = fit$beta
test = coef(fit) %>% as.matrix()

# Toy example:
X<-matrix(c(-1,-4,0,0,1,16),nrow=3,ncol=2,byrow=TRUE)
y<-matrix(c(-2.2,0,3.8),nrow=3,ncol=1,byrow=TRUE)
norm_X = apply(X,2,normalize)
fit <- glmnet(x=norm_X, y=y, lambda = 0.1, alpha = 1, 
              standardize = FALSE, intercept = FALSE)
beta = fit$beta
test = coef(fit) %>% as.matrix()