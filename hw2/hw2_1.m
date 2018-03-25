% load('cam3_4.mat');
% load('cam3_3.mat');
% load('cam3_2.mat');
load('cam3_1.mat');
% load('cam2_4.mat');
% load('cam2_3.mat');
% load('cam2_2.mat');
load('cam2_1.mat');
% load('cam1_4.mat');
% load('cam1_3.mat');
% load('cam1_2.mat');
load('cam1_1.mat');



numframes1 = size(vidFrames1_1,4);
for k1 = 1 : numframes1
mov(k1).cdata = vidFrames1_1(:,:,:,k1);
mov(k1).colormap = [];
end

numframes2 = size(vidFrames2_1,4);
for k1 = 1 : numframes2
mov(k1).cdata = vidFrames2_1(:,:,:,k1);
mov(k1).colormap = [];
end

numframes3 = size(vidFrames3_1,4);
for k1 = 1 : numframes3
mov(k1).cdata = vidFrames3_1(:,:,:,k1);
mov(k1).colormap = [];
end


% plotting the images


x1 = zeros(1,numframes1);
y1 = zeros(1,numframes1);
for j=1:numframes1
X=frame2im(mov(j));
X = rgb2gray(X); X=X';% first convert to grayscale, then take the adjoint so that the motion is along z direction                        
X = X(200:end,220:end-150); % remove all the white lights in the background - like the tubelight reflection; this also makes it lot faster
[val1, idx1] = max(max(X));
[val2, idx2] = max(max(X'));
x1(j) = idx1; y1(j) = idx2;
%imshow(X); drawnow 
end

x2 = zeros(1,numframes2);
y2 = zeros(1,numframes2);
for j=1:numframes2
X=frame2im(mov(j));
X = rgb2gray(X); 
X=X'; % first convert to grayscale, then take the adjoint so that the motion is along z direction                        
X = X(200:end,220:end-150); % remove all the white lights in the background - like the tubelight reflection; this also makes it lot faster
[val1, idx1] = max(max(X));
[val2, idx2] = max(max(X'));
x2(j) = idx1; y2(j) = idx2;
%imshow(X); drawnow 
end

x3 = zeros(1,numframes3);
y3 = zeros(1,numframes3);
for j=1:numframes3
X=frame2im(mov(j));
X = rgb2gray(X); X=X'; % first convert to grayscale, then take the adjoint so that the motion is along z direction                        
X = X(200:end,220:end-150); % remove all the white lights in the background - like the tubelight reflection; this also makes it lot faster
[val1, idx1] = max(max(X));
[val2, idx2] = max(max(X'));
x3(j) = idx1; y3(j) = idx2;
%imshow(X); drawnow 
end

A = cat(1,x1,y1,x2(1:226),y2(1:226),x3(1:226),y3(1:226));
[m,n]=size(A); % compute data size
mn=mean(A,2); % compute mean for each row
A=A-repmat(mn,1,n); % subtract mean
Cx=(1/(n-1))*(A*A'); % covariance
[V,D]=eig(Cx); % eigenvectors(V)/eigenvalues(D)
lambda=diag(D); % get eigenvalues
[dummy,m_arrange]=sort(-1*lambda); % sort in decreasing order
lambda=lambda(m_arrange);
V=V(:,m_arrange);
Y=V'*A; % produce the principal components projection

% figure
% plot(x1,y1,'r.') 
% a1 = [0.3 0.7];
% b1 = [0.3 0.6];
% a2 = [0.5 0.4];
% b2 = [0.4 0.6];
% a=annotation('textarrow',a1,b1,'String','\sigma^2_{signal}');
% a.Color = 'black';
% a.FontSize = 15;
% b=annotation('textarrow',a2,b2,'String','\sigma^2_{noise}');
% b.Color = 'blue';
% b.FontSize = 15;
%hold on
%quiver(62,255,100,20,0)
%hold on
%quiver(105,260,-10,20,0)

% plot(x1,y1,'r.')
% hold on
% plot(V(1:2),1)

[u,s,v]=svd(A'/sqrt(n-1)); % perform the SVD
lambda1=diag(s).^2; % produce diagonal variances
Y1=v'*A;

% plot(x1)
%set(gca,'Fontsize',[20])
% hold on
% plot(Y(1,:))

%plot the initial motion 
figure(1)
subplot(2,3,1) 
plot(x1,'LineWidth',2)
xlabel('time') 
ylabel('x coordinate')
set(gca,'Fontsize',[17],'FontWeight','bold')
title('Camera1')
subplot(2,3,2) 
plot(x2,'LineWidth',2)
xlabel('time') 
ylabel('x coordinate')
set(gca,'Fontsize',[17],'FontWeight','bold')
title('Camera2 - X direction motion')
subplot(2,3,3) 
plot(x3,'LineWidth',2)
xlabel('time') 
ylabel('x coordinate')
set(gca,'Fontsize',[17],'FontWeight','bold')
title('Camera3 - X direction motion')
subplot(2,3,4) 
plot(y1,'LineWidth',2)
xlabel('time') 
ylabel('y coordinate')
set(gca,'Fontsize',[17],'FontWeight','bold')
title('Camera1 - Y direction motion')
subplot(2,3,5) 
plot(y2,'LineWidth',2)
xlabel('time') 
ylabel('y coordinate')
set(gca,'Fontsize',[17],'FontWeight','bold')
title('Camera2 - Y direction motion')
subplot(2,3,6) 
plot(y3,'LineWidth',2)
xlabel('time') 
ylabel('y coordinate')
set(gca,'Fontsize',[17],'FontWeight','bold')
title('Camera3 - Y direction motion')

figure(2)
plot(Y1(1,:),'LineWidth',2)
title('Principal components for CASE 1')
xlabel('time (number of frames)')
ylabel('displacement')
hold on
plot(Y1(2,:),'LineWidth',2)
hold on
plot(Y1(3,:),'LineWidth',2)
set(gca,'Fontsize',[20],'FontWeight','bold')
legend('PCA - model','PCA - mode2','PCA - mode3')

figure(3)
subplot(1,3,1)
plot(x1,y1,'ro','LineWidth',2)
xlabel('x coordinate') 
ylabel('y coordinate')
set(gca,'Fontsize',[20],'FontWeight','bold')
title('Camera1 - Original data')
subplot(1,3,2)
plot(x2,y2,'ro','LineWidth',2)
xlabel('x coordinate') 
ylabel('y coordinate')
set(gca,'Fontsize',[20],'FontWeight','bold')
title('Camera2 - Original data')
subplot(1,3,3)
plot(x3,y3,'ro','LineWidth',2)
xlabel('x coordinate') 
ylabel('y coordinate')
set(gca,'Fontsize',[20],'FontWeight','bold')
title('Camera3 - Original data')


%plot the eigenvalue spectrum at each mode
figure(4)
plot(lambda1,'ro-','LineWidth',2)
title('Eigenvalue spectrum')
ylabel('magnitude of eigenvalue')
xlabel('modes')
set(gca,'Fontsize',[20], 'FontWeight','bold')
 
