clear all;
images_Train = loadMNISTImages('train-images.idx3-ubyte');
images_Test = loadMNISTImages('t10k-images.idx3-ubyte');
images = [ images_Train , images_Test ] ;

labels_Train = loadMNISTLabels('train-labels.idx1-ubyte') ;
labels_Test = loadMNISTLabels('t10k-labels.idx1-ubyte') ;
labels = [ labels_Train;labels_Test] ;

index = randperm(40000);
train = images(:,index(1:30000));
train_labels = labels(index(1:30000));
test =  images(:,index(30001:40000));
test_labels_true = labels(index(30001:40000));
labels1 = full(ind2vec(train_labels' + 1)) ;
A1=labels1*pinv(train);
pre_labels=(A1*test);
[ind,n] = vec2ind(pre_labels);
ind = ind - 1;
accuracy = ((10000 - nnz((ind' - test_labels_true)))/10000)*100;
% accuracy_table = [83.89,85.04,84.02,84.64,84.53];
% mn = mean(accuracy_table);
% plot(accuracy_table)
% 
% figure(1)
% plot(accuracy_table,'ro','LineWidth',3)
% xlabel('iteration')
% ylabel('prediction accuracy (%)')
% set(gca,'Fontsize',[20],'FontWeight','bold')
% title('Prediction accuracy using cross-validation')






