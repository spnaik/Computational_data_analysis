clc; close all; 
clear all;

% ------- cropped yale faces ---------------

a = 1:39; a(14) = [];
p = genpath('CroppedYale');
addpath(p)
A = zeros(32256,2432); % alloting space for A matrix to make it fast
for i=1:38
    newFolder = sprintf('./CroppedYale/yaleB%d', a(i));
    cd(newFolder)
    pgmFiles = dir('*.pgm');
    nfiles = length(pgmFiles);
    for ii=1:nfiles
        currentfilename = pgmFiles(ii).name;
        currentimage = double(imread(currentfilename));
        A(:,nfiles*(i-1)+ii)= reshape(currentimage,[],1); % reshaping/vectorizing the matrix
    end
    
   cd 'F:\UW\Comp_DA\Homework1'
end
A = A - mean(A,1); % centering the matrix
[U,S,V] = svd(A,'econ'); % singular value decomposition

%plotting all the images of the 1st person under different illumination
for i=1:8
for j=1:8
SS(1+(i-1)*192:i*192,1+(j-1)*168:j*168) = reshape(A(:,j+8*(i-1)),[192,168]);
end 
end
%imagesc(SS); colormap(gray); title('images for a specific person under different illumination'); set(gca,'XTick',[],'Fontsize',[20]); set(gca,'YTick',[],'Fontsize',[20])

% A single image for each person in different folder, here the first image
% in each folder is chosen
for i=1:6
for j=1:6
SS1(1+(i-1)*192:i*192,1+(j-1)*168:j*168) = reshape(A(:,64*(j+6*(i-1))-64 + 1),[192,168]);
end 
end
%imagesc(SS1); colormap(gray); title('A single image for each person'); set(gca,'XTick',[],'Fontsize',[20]); set(gca,'YTick',[],'Fontsize',[20])

% plot both the images
subplot(1,2,1); imagesc(SS); colormap(gray); title('images for a specific person'); set(gca,'XTick',[],'Fontsize',[20]); set(gca,'YTick',[],'Fontsize',[20])
subplot(1,2,2); imagesc(SS1); colormap(gray); title('A single image for each person'); set(gca,'XTick',[],'Fontsize',[20]); set(gca,'YTick',[],'Fontsize',[20])

% Figure plotting of singular value spectrum
sig = diag(S)/sum(diag(S)); % percent energy in each singular value
figure(1)
subplot(1,2,1),plot(sig,'ro','Linewidth',[1.5])
xlabel('total number of singular values')
ylabel('energy in each singular value')
set(gca,'Fontsize',[20])
text(2000,0.05,'(a)','Fontsize',[13])
title('singular value spectrum')

subplot(1,2,2),semilogy(sig,'ro','Linewidth',[1.5])
xlabel('total number of singular values')
ylabel('energy in each singular value')
set(gca,'Fontsize',[20])
text(2000,0.05,'(b)','Fontsize',[13])
title('Log plot - singular value spectrum')

% ----- reconstructing the images using low rank r -----
mat1 = [80,100,200,400,800,1200,1600];
for k=1:6
ff=U(:,1:mat1(k))*S(1:mat1(k),1:mat1(k))*V(:,1:mat1(k))';
a= ff(:,35*64 + 1);
b = int8(reshape(a,[192,168]));
a1 = A(:,35*64 + 1);b1 = int8(reshape(a1,[192,168]));
subplot(1,7,1) ;imshow(b1)
title('original image')
subplot(1,7,k+1) ;imshow(b)
title(['rank r = ' num2str(mat1(k)) ''])
end

for k=1:6
ff=U(:,1:mat1(k))*S(1:mat1(k),1:mat1(k))*V(:,1:mat1(k))';
a= ff(:,35*64 + 47);
b = int8(reshape(a,[192,168]));
a1 = A(:,35*64 + 47);b1 = int8(reshape(a1,[192,168]));
subplot(1,7,1) ;imshow(b1)
title('original image')
subplot(1,7,k+1) ;imshow(b)
title(['rank r = ' num2str(mat1(k)) ''])
end
% ---------------- yalefaces ------------
cd 'yalefaces'
imageFiles = dir('*');
    nimages = length(imageFiles);
    for j=3:nimages
        currfilename = imageFiles(j).name;
        currimage = double(imread(currfilename));
        B(:,j-2)= reshape(currimage,[],1);
    end
    
   cd 'F:\UW\Comp_DA\Homework1'
   
   B = B - mean(B,1);
   [U1,S1,V1] = svd(B,'econ');
   
% Figure plotting of singular value spectrum
sig1 = diag(S1)/sum(diag(S1)); % percent energy in each singular value
figure(3)
subplot(1,2,1),plot(sig1,'ro','Linewidth',[1.5])
xlabel('total number of singular values')
ylabel('energy in each singular value')
set(gca,'Fontsize',[20])
text(150,0.13,'(a)','Fontsize',[13])
title('singular value spectrum')

subplot(1,2,2),semilogy(sig1,'ro','Linewidth',[1.5])
xlabel('total number of singular values')
ylabel('energy in each singular value')
set(gca,'Fontsize',[20])
text(150,0.1,'(b)','Fontsize',[13])
title('Log plot - singular value spectrum')


% reconstructing the eigenfaces
ff1=U1(:,1:90)*S1(1:90,1:90)*V1(:,1:90)';
a1= ff1(:,1); b1 = int8(reshape(a1,[243,320]));
imshow(b1)

% image - with glasses
mat2 = [25,50,75,100,120,150,165];
for k=1:6
ff1=U1(:,1:mat2(k))*S1(1:mat2(k),1:mat2(k))*V1(:,1:mat2(k))';
a2= ff1(:,2);
b2 = int8(reshape(a2,[243,320]));
cd 'yalefaces' ; subplot(1,7,1) ; trial = imread('subject01.glasses'); imshow(trial); title('original image');
cd 'F:\UW\Comp_DA\Homework1'
subplot(1,7,k+1) ;imshow(b2)
title(['rank r = ' num2str(mat2(k)) ''])
end

% image - with leftlight
mat2 = [25,50,75,100,120,150,165];
for k=1:6
ff1=U1(:,1:mat2(k))*S1(1:mat2(k),1:mat2(k))*V1(:,1:mat2(k))';
a2= ff1(:,15);
b2 = int8(reshape(a2,[243,320]));
cd 'yalefaces' ; subplot(1,7,1) ; trial = imread('subject02.leftlight'); imshow(trial); title('original image');
cd 'F:\UW\Comp_DA\Homework1'
subplot(1,7,k+1) ;imshow(b2)
title(['rank r = ' num2str(mat2(k)) ''])
end
