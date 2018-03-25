clear all;
images_Train = loadMNISTImages('train-images.idx3-ubyte');
images_Test = loadMNISTImages('t10k-images.idx3-ubyte');
images = [ images_Train , images_Test ] ;

labels_Train = loadMNISTLabels('train-labels.idx1-ubyte') ;
labels_Test = loadMNISTLabels('t10k-labels.idx1-ubyte') ;
labels = [ labels_Train;labels_Test] ;

% first 16 images
for i=1:16
    A = images_Train(:,i);
    A = reshape(A, 28, 28);
    B(:,:,i) = A;
    subplot(4,4,i)
    imshow(B(:,:,i))
end

% labels1 = full(ind2vec(labels_Train' + 1)) ;
% A1=labels1*pinv(images_Train);
% test_labels=sign(A1*images_Test);
% [ind,n] = vec2ind(test_labels);

index = randperm(60000);
train = images_Train(:,index(1:50000));
train_labels = labels_Train(index(1:50000));
test =  images_Train(:,index(50001:60000));
test_labels_true = labels_Train(index(50001:60000));
labels1 = full(ind2vec(train_labels' + 1)) ;
A1=labels1*pinv(train);
pre_labels=(A1*test);
[ind,n] = vec2ind(pre_labels);
ind = ind - 1;
accuracy = ((10000 - nnz((ind' - test_labels_true)))/10000)*100;
accuracy_table = [83.85,84.59,84.4,83.9,84.89];






