install.packages("caret")

library(caret)
library(rpart)
library(dplyr)

mushroomdata <- read.csv("AfterEDA_dataset.csv")

glimpse(mushroomdata)

#As we can see we have 8124 observations and 19 columns which has 1 dependent variable called "class"
# and 17 independent variables.

set.seed(43)
randomized=mushroomdata[sample(1:nrow(mushroomdata),nrow(mushroomdata)),] #Random sampling for creating test and train data
tridx=sample(1:nrow(mushroomdata),0.7*nrow(mushroomdata),replace=F) 
trdf=randomized[tridx,] # Training data set
tstdf=randomized[-tridx,] # Testing data set

#Now we will check if the distribution is similar in the actual data, training data and testing data

table(mushroomdata$class)/nrow(mushroomdata)


table(trdf$class)/nrow(trdf)

table(tstdf$class)/nrow(tstdf)

# As we can see the distribution is almost same. So the training dataset and testing dataset is a proper 
# representation of the actual data 

varEst_tridx=sample(1:nrow(trdf), 0.9*nrow(trdf), replace=F) # Taking 90% of the total number of samples
varEst_trdf=trdf[varEst_tridx,] # Training data for variance estimation
varEst_tstdf=trdf[-varEst_tridx,] # Testing data for variance estimation

varEst=function(trdf,tstdf,percent,type){
    target_idx=which(names(trdf)=="class")
    acc_varEstp=c(); # Initialize a variable to store the accuracies computed in the loop
    for(i in 1:10){
        varEstp_tridx=sample(1:nrow(trdf), percent/100*nrow(trdf), replace=F) # Take samples, percent% of the data
        varEstp_trdf=trdf[varEstp_tridx,]
        
        # For logistic
        if(type=="glm"){
            mn_model_varEstp=glm(formstr, varEstp_trdf,family=gaussian()) #Train a Logistic model
            pred_varEstp=predict(mn_model_varEstp, tstdf[,-target_idx], type="response") # Predict with variance estimation partition
        } 
        
        # For SVM
        else if(type=="svm"){
            svm_model_varEstp = svm(class~., varEstp_trdf, scale = FALSE)
            pred_varEstp1=predict(svm_model_varEstp, tstdf[,-target_idx])
            pred_varEstp = ifelse(pred_varEstp1 > 0.5, "1", "0")
        }
            
        # For Tree
        else if(type=="rpart"){ 
            tree_model_varEstp=rpart(formstr, varEstp_trdf,method = 'class')
            pred_varEstp=predict(tree_model_varEstp, tstdf[,-target_idx], type="class")
            
        }
            
        # For KNN 
        else if(type=="knn"){
            trclass=factor(varEstp_trdf[,target_idx])
            tstclass=factor(tstdf[,target_idx])
            pred_varEstp=knn(varEstp_trdf[,-target_idx], tstdf[,-target_idx], trclass, k = 15, prob=TRUE)
        } 
        
        else {
            print("Type should be Logistic or tree or KNN or SVM")
            return()
        }
        u_varEstp=union(pred_varEstp, tstdf[,target_idx]) # Avoids issues when number of classes are not equal
        t_varEstp=table(factor(pred_varEstp, u_varEstp), factor(tstdf[,target_idx],u_varEstp))
        mn_cfm_varEstp=confusionMatrix(t_varEstp) # Confusion Matrix
        mn_acc_varEstp=mn_cfm_varEstp$overall[["Accuracy"]] # Accuracy of predictions
        acc_varEstp=c(acc_varEstp,mn_acc_varEstp) # Store
    }
    mean_varEstp=signif(mean(acc_varEstp),4)
    var_varEstp=signif(var(acc_varEstp),4)
    varEstp=data.frame(mean_varEstp,var_varEstp)
    names(varEstp)=c("Mean of Accuracies","Variance of Accuracies")
    return(t(varEstp))
}

# We will use this function to caluculate Variance Estimation for 4 algorithms we are going to work on

# Create a logistic regression model 
formstr="class~." # Formula argument with all features
glm_model1 = glm(formstr,trdf,family=gaussian()) # Train the model

summary(glm_model1)

# Predict using the train data (Learning Phase)
glm_pred1_tr_=predict(glm_model1,trdf[, -which(names(trdf)=="class")], type = "response")
# glm_pred1_tr_ will the predicted values from 1 to 2(can be any value, decimal also) as it is a binomial distribution. 
# To make the glm_pred1_tr_ in 1 and 2, using the below if loop. 
glm_pred1_tr = ifelse(glm_pred1_tr_ > 1, "2", "1")

print(glm_pred1_tr)

glm_cfm1_tr=confusionMatrix(table(trdf[, which(names(trdf)=="class")], glm_pred1_tr)) # Confusion Matrix for train data

print("Learning Phase Confusion Matrix")
glm_cfm1_tr

glm_acc1_tr=round(glm_cfm1_tr$overall[["Accuracy"]],4) # Accuracy of predictions with train data

print(paste("Classification accuracy of learning phase =",glm_acc1_tr))

# Predict using the test data (Generalization Phase)
glm_pred1_tst_=predict(glm_model1,tstdf[, -which(names(tstdf)=="class")], type="response")
# glm_pred1_tr_ will the predicted values from 0 to 1(can be any value, decimal also) as it is a binomial distribution. 
# To make the glm_pred1_tr_ in 1 and 0, using the below if loop.
glm_pred1_tst = ifelse(glm_pred1_tst_ > 1, "2", "1")

glm_cfm1_tst=confusionMatrix(table(tstdf[, which(names(tstdf)=="class")], glm_pred1_tst)) # Confusion Matrix for test data

print("Generalization Phase Confusion Matrix")
glm_cfm1_tst

glm_acc1_tst=round(glm_cfm1_tst$overall[["Accuracy"]],4) # Accuracy of predictions with test data

print(paste("Classification accuracy of generalization phase =",glm_acc1_tst))

# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
glm_model1_isOF=abs((glm_acc1_tr-glm_acc1_tst)/glm_acc1_tr)

glm_model1_isOF=round(glm_model1_isOF,4)

print(paste("Accuracy drop from training data to test data is",glm_model1_isOF*100,"%"))

if(glm_model1_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")

glm_PM1_tr=glm_cfm1_tr$byClass[c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]

print("Logistic-Regression Learning-Phase Performance Parameters:")
glm_PM1_tr

glm_prob1_tr=predict(glm_model1, trdf[,-which(names(trdf)=="class")], type="response")

require(pROC)

glm_AUC1_tr = roc(trdf[,which(names(trdf)=="class")],glm_prob1_tr)

print(paste("Logistic-Regression Learning-Phase AUC:",round(glm_AUC1_tr$auc,4)))

# ROC curves
plot.roc(glm_AUC1_tr)

# Generalization Phase
glm_PM1_tst=glm_cfm1_tst$byClass[c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]

print("Logistic-Regression Generalization-Phase Performance Parameters:")
glm_PM1_tst

glm_prob1_tst=predict(glm_model1, tstdf[,-which(names(tstdf)=="class")], type="response")

glm_AUC1_tst=roc(tstdf[, which(names(tstdf)=="class")], glm_prob1_tst)

print(paste("Logistic-Regression Generalization-Phase AUC:", round(glm_AUC1_tst$auc,4)))

# ROC curves
plot.roc(glm_AUC1_tst)

glm_cfm1_tst

glm_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="glm") # Variance estimation using 30% of the data
glm_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="glm") # Variance estimation using 60% of the data
glm_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="glm") # Variance estimation using 100% of the data

print("Logistic-Regression Variance Estimation using 30% of data:")
glm_varEst30

print("Logistic-Regression Variance Estimation using 60% of data:")
glm_varEst60

print("Logistic-Regression Variance Estimation using 100% of data:")
glm_varEst100

require(e1071)

svm_model1 = svm(class~., data = trdf,method = "svmLinear")

print(svm_model1)

# Predict using train data (Learning Phase)
svm_pred1_tr=predict(svm_model1, trdf[, -which(names(trdf)=="class")])
svm_pred1_tr1 = ifelse(svm_pred1_tr > 1, "2", "1")

svm_cfm1_tr=confusionMatrix(table(trdf$class,svm_pred1_tr1)) # Confusion Matrix for train data


print("SVM Learning Phase Confusion Matrix")
svm_cfm1_tr

svm_acc1_tr=round(svm_cfm1_tr$overall[["Accuracy"]],4)

print(paste("SVM Learning Phase Accuracy =",svm_acc1_tr))

# Predict using test data (Generalization Phase)
svm_pred1_tst=predict(svm_model1, tstdf[, -which(names(tstdf)=="class")], type="class")
svm_pred1_tst1 = ifelse(svm_pred1_tst > 1, "2", "1")

svm_cfm1_tst=confusionMatrix(table(tstdf[, which(names(tstdf)=="class")], svm_pred1_tst1)) # Confusion Matrix for test data


print("SVM Generalization Phase Confusion Matrix")
svm_cfm1_tst

svm_acc1_tst=round(svm_cfm1_tst$overall[["Accuracy"]],4) # Accuracy 

print(paste("SVM Generalization Phase Accuracy =",svm_acc1_tst))

# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
svm_model1_isOF=abs((svm_acc1_tr-svm_acc1_tst)/svm_acc1_tr)
svm_model1_isOF=round(svm_model1_isOF,4)
print(paste("Accuracy drop from training data to test data is",svm_model1_isOF*100,"%"))

if(svm_model1_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")

svm_PM1_tr=svm_cfm1_tr$byClass[c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]


print("SVM Learning-Phase Performance Parameters:")
svm_PM1_tr

svm_prob1_tr=predict(svm_model1, trdf[, -which(names(trdf)=="class")],)

svm_AUC1_tr=roc(trdf[, which(names(trdf)=="class")], svm_prob1_tr)

print(paste("SVM Learning-Phase AUC:", round(svm_AUC1_tr$auc, 4)))

plot.roc(svm_AUC1_tr)

svm_PM1_tst=svm_cfm1_tst$byClass[c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]


print("SVM Generalization-Phase Performance Parameters:")
svm_PM1_tst

svm_prob1_tst=predict(svm_model1, tstdf[, -which(names(tstdf)=="class")], type="raw")
svm_prob1_tst1 = ifelse(svm_prob1_tst > 1, "2", "1")

svm_AUC1_tst=roc(tstdf[, which(names(tstdf)=="class")],svm_prob1_tst)

print(paste("SVM Generalization-Phase AUC:", round(svm_AUC1_tst$auc, 4)))

# ROC curves - Curve of Sensitivity and Specificity
plot.roc(svm_AUC1_tst)

svm_cfm1_tst

# Variance Estimation for SVM

svm_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="svm") # Variance estimation using 30% of the data


svm_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="svm") # Variance estimation using 60% of the data


svm_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="svm") # Variance estimation using 100% of the data


print("SVM Variance Estimation using 30% of data:")
svm_varEst30

print("SVM Variance Estimation using 60% of data:")
svm_varEst60

print("SVM Variance Estimation using 100% of data:")
svm_varEst100

# As we have many categorical variables, regression tree is an ideal classification tools for such situation.
# We’ll use the rpart package.

install.packages("rpart.plot")

library(rpart)
library(rpart.plot)
tree_model1 = rpart(class~., data = trdf, method = 'class')


rpart.plot(tree_model1)

printcp(tree_model1)

# Predict using train data (Learning Phase)
tree_pred1_tr=predict(tree_model1, trdf[, -which(names(trdf)=="class")],type="class")

tree_cfm1_tr=confusionMatrix(table(trdf[, which(names(trdf)=="class")],tree_pred1_tr)) # Confusion Matrix for train data


print("Tree Learning Phase Confusion Matrix")
tree_cfm1_tr

tree_acc1_tr=round(tree_cfm1_tr$overall[["Accuracy"]],4) # Accuracy


print(paste("Tree Learning Phase Accuracy =",tree_acc1_tr))


tree_pred1_tst=predict(tree_model1, tstdf[, -which(names(trdf)=="class")], type = 'class')


tree_cfm1_tst=confusionMatrix(table(tstdf[, which(names(tstdf)=="class")],tree_pred1_tst)) # Confusion Matrix for test data


print("Tree Generalization-Phase Confusion Matrix")
tree_cfm1_tst

tree_acc1_tst=round(tree_cfm1_tst$overall[["Accuracy"]],4) # Accuracy


print(paste("Tree Generalization-Phase Accuracy =",tree_acc1_tst))


# Check for over-fitting. Criteria: Accuracy change from train to test > 25%
tree_model1_isOF=abs((tree_acc1_tr-tree_acc1_tst)/tree_acc1_tr)
tree_model1_isOF=round(tree_model1_isOF,4)
print(paste("Accuracy drop from training data to test data is",tree_model1_isOF*100,"%"))

if(tree_model1_isOF>0.25) print("Model is over-fitting") else print("Model is not over-fitting")


tree_PM1_tr=tree_cfm1_tr$byClass[c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]


print("Tree Learning-Phase Performance Parameters:")
tree_PM1_tr

tree_prob1_tr=predict(tree_model1, trdf[, -which(names(trdf)=="class")], type = 'prob')


tree_prob1_tr1 = tree_prob1_tr[,2]


tree_AUC1_tr=multiclass.roc(trdf$class, tree_prob1_tr1, percent = TRUE)
print(paste("tree Learning-Phase AUC:",round(tree_AUC1_tr$auc,4)))

tree_ROC1_tr <- tree_AUC1_tr[['rocs']]


plot.roc(tree_ROC1_tr[[1]])


tree_PM1_tst=tree_cfm1_tst$byClass[c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]


print("Tree Generalization-Phase Performance Parameters:")
tree_PM1_tst

tree_prob1_tst=predict(tree_model1, tstdf[, -which(names(tstdf)=="class")], type = 'prob')

tree_prob1_tst1 = tree_prob1_tst[,2]

tree_AUC1_tst=multiclass.roc(tstdf$class, tree_prob1_tst1, percent = TRUE)
print(paste("tree Generalization-Phase AUC:",round(tree_AUC1_tst$auc,4)))

tree_ROC1_tst <- tree_AUC1_tst[['rocs']]

plot.roc(tree_ROC1_tst[[1]])

tree_cfm1_tst

# Variance Extimation for Tree

tree_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="rpart") # Variance estimation using 30% of the data
tree_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="rpart") # Variance estimation using 60% of the data
tree_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="rpart") # Variance estimation using 100% of the data

print("Tree Variance Estimation using 30% of data:")
tree_varEst30

print("Tree Variance Estimation using 60% of data:")
tree_varEst60

print("Tree Variance Estimation using 100% of data:")
tree_varEst100

trdf_knn=trdf[, -which(names(trdf)=="class")]

tstdf_knn=tstdf[, -which(names(trdf)=="class")]

trclass_knn=factor(trdf[, which(names(trdf)=="class")])

tstclass_knn=factor(tstdf[, which(names(tstdf)=="class")])

library(class)

knn_pred1=knn(trdf_knn,tstdf_knn,trclass_knn, k = 15, prob=TRUE)

# Predict using test data (Generalization Phase)
knn_cfm1_tst=confusionMatrix(table(tstclass_knn,knn_pred1)) # Confusion Matrix for test data

knn_cfm1_tst

knn_acc1_tst=round(knn_cfm1_tst$overall[["Accuracy"]],4) # Accuracy of predictions with test data

print(paste("kNN Generalization Phase Accuracy =",knn_acc1_tst))

knn_PM1_tst=knn_cfm1_tst$byClass[c("Balanced Accuracy", "Precision", "Sensitivity", "Specificity", "Recall")]

print("kNN Generalization-Phase Performance Parameters:")
knn_PM1_tst

knn_prob1_tst=attr(knn_pred1,"prob")

knn_AUC1_tst=multiclass.roc(tstclass_knn, as.ordered(knn_pred1))

print(paste("kNN Generalization-Phase AUC:",round(knn_AUC1_tst$auc,4)))

# ROC curves
knn_ROC1_tst=knn_AUC1_tst$rocs
plot.roc(knn_ROC1_tst[[1]], col=1)

knn_cfm1_tst

# Variance Estimation
knn_varEst30=varEst(varEst_trdf, varEst_tstdf, 30, type="knn") # 30% of data
knn_varEst60=varEst(varEst_trdf, varEst_tstdf, 60, type="knn") # 60% of data
knn_varEst100=varEst(varEst_trdf, varEst_tstdf, 100, type="knn") # 100% of data

print("kNN Variance Estimation using 30% of data:")
knn_varEst30

print("kNN Variance Estimation using 60% of data:")
knn_varEst60

print("kNN Variance Estimation using 100% of data:")
knn_varEst100
