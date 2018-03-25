% case 1
clear all; close all;
cd('F:\UW\Comp_DA\Homework3/genres/artists');
basepath = pwd;
allpaths = dir(basepath);
subFolders = [allpaths(:).isdir];
foldersNames = {allpaths(subFolders).name};
foldersNames(ismember(foldersNames,{'.','..'})) = []; 

Spec_mat = [];

for i=1:length(foldersNames)
    tmp = foldersNames{i};
    p = strcat([basepath '\']);
    currentPath = strcat([p tmp]);
    cd(currentPath);
    pgm_file = dir('*.au');
    index = {pgm_file.name};
    
     for j=1:length(index)
         [y,Fs] = audioread(index{j});
          clear y,Fs;
          samples = [10*Fs, 15*Fs];
          [y,Fs] = audioread(index{j},samples);
          sig = y(:,1);
         
          window = hamming(1024);
          nooverlap = 256;
          [b,f,t] = spectrogram(sig, window, nooverlap);
          b = abs(b);
          nRows = length(f);
          nSample = 100;
          rndIDX = randperm(nRows);
          new_b = b(rndIDX(1:nSample),:);
          b_vec = reshape(new_b,[],1);
          Spec_mat = [Spec_mat,b_vec];
             
     end
end

 save('Spec_mat_task3.mat','Spec_mat');
 clear;
 file3 = matfile('Spec_mat_task3.mat');
 Spec_mat3 = file3.Spec_mat; 
 [m,n] = size(Spec_mat3);
  mn = mean(Spec_mat3,2);
  Spec_mat3 = Spec_mat3 - repmat(mn,1,n);
  [U3,S3,V3] = svd(Spec_mat3,'econ');
  eig_val = diag(S3);
% %x = [1:7:77];
  eig_val_per = diag(S3)/sum(diag(S3));
  
 figure(1)
 subplot(1,2,1)
 plot(eig_val,'ro')
 xlabel('rank of matrix','LineWidth',2)
 ylabel('Singular value');
 set(gca,'Fontsize',[20],'FontWeight','bold')
 title ( 'Singular value spectrum (Task1)' ) ;

subplot(1,2,2)
plot(1:300,V3(1,:),'r');
hold on
plot(1:300,V3(101,:),'g');
hold on
plot(1:300,V3(201,:),'b');
hold on
plot(1:300,V3(1:100,:),'r');
hold on
plot(1:300,V3(101:200,:),'g');
hold on
plot(1:300,V3(201:300,:),'b');
xlabel('Feature index','LineWidth',2)
ylabel('Feature value','LineWidth',2);
set(gca,'Fontsize',[20],'FontWeight','bold')
title ( 'Spectrogram (Task 1)' ) ;
legend ( ' John Adams' ,'Pitbull' , 'BobMarley' ) ;



q1 = randperm(100);
q2 = randperm(100);

XClass = V3(1:100,2:151);
XJazz = V3(101:200,2:151);
XPop = V3(201:300,2:151);

xtrain = [XClass(q1(1:70),:);XJazz(q1(1:70),:);XPop(q1(1:70),:)];
xtest = [XClass(q1(71:end),:);XJazz(q1(71:end),:);XPop(q1(71:end),:)];
% for naive bayes, label taining data
ctrain = [ones(70,1);2*ones(70,1);3*ones(70,1)];
ctest = [ones(30,1);2*ones(30,1);3*ones(30,1)];

%naivebayes
nb = fitcnb(xtrain,ctrain);
pre = nb.predict(xtest);
bar(pre)
ctest = [ones(30,1);2*ones(30,1);3*ones(30,1)];
k = pre - ctest;
display((size(ctest,1)-nnz(k))/size(ctest,1));
% 

%LDA
class = classify(xtest,xtrain,ctrain,'linear') ;
k = class - ctest;
display((size(ctest,1)-nnz(k))/size(ctest,1));
bar(class)
set(gca,'Fontsize',[20],'FontWeight','bold')
xlabel('sample index')
ylabel('class index')
title('LDA classification')



% 






cd('F:\UW\Comp_DA\Homework3');
