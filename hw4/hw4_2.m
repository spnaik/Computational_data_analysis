%% Read Film 1
clear; clc;
video=[];
v = VideoReader('people_walk_change.mp4');
% play video
% videoFReader = vision.VideoFileReader('film11.mp4');
% videoPlayer = vision.VideoPlayer;
% while ~isDone(videoFReader)
%   videoFrame = videoFReader();
%   videoPlayer(videoFrame);
% end

while hasFrame(v)
  frame = readFrame(v) ;
  frame = rgb2gray(frame) ;
  frame = reshape(frame ,[ ],1) ;
  video = [ video,frame ] ;
end
%% Original video

[m, n1] = size(video) ;
n1 = n1-10;
video = video (:,1: n1) ;
%%
video = double(video) ;
v1 = video (:,1:end - 1);
v2 = video (:,2:end) ;
[U2,Sigma2,V2] = svd(v1,'econ') ;
%%
r=100; U=U2(:,1:r) ;
Sigma = Sigma2( 1:r , 1:r ) ;
L =n1 -1;
n= 307200;
slices = n1-1;
t = linspace( 0,1,slices +1);
dt = t(2) - t(1);
nf = 1;
slicesf = slices*nf;
tf = linspace(0,nf,slicesf +1)';
% 
% %%
V = V2(:,1:r);
Atilde = U'*v2*V/Sigma;
[W,D] = eig(Atilde);
Phi = v2*V/Sigma*W;
lamda = diag(D);
omega = log(lamda)/dt;
y0 = Phi\video(:,1);
%y0 = pinv(Phi)*video(:,1);

% %%
small = find (abs(omega ) == min (abs(omega))) ;
big = find (abs(omega) ~=min(abs(omega))) ;
omega1 = omega(small) ;
omega2 = omega(big) ;
y01 = y0(small) ;
y02 =y0(big) ;
%%
r1 = length(small) ;
r2 = length(big) ;
u_bak = zeros( r1,length(t)) ;
u_for = zeros( r2,length(t)) ;
for iter = 1 : length (tf )
 u_bak (:,iter ) = (y01.*exp(omega1*(tf(iter))));
end
for iter = 1 : length (tf)
u_for (:,iter) = ( y02.*exp ( omega2*(tf(iter)))) ;
end

%%
X_low = Phi(:,small)*u_bak ;
X_sparse = Phi(:, big )*u_for ;
%%
X = X_low+X_sparse ;
X_sparse = real(X-abs(X_low)) ;
X_sparse = X_sparse - min(min(X_sparse)) ;
X_low = abs (X_low) + min(min(X_sparse));

%%
figure (1),
subplot(3,4,1) ,
temp = video( : , 30) ;
temp = reshape(temp,480,640 ) ;
imagesc(temp) ;
title('t = 1s')
colormap(gray) ;
axis off ;
subplot(3,4,2 ) ,
temp = video(:,60) ;
temp = reshape (temp ,480,640) ;
imagesc(temp) ;
title('t = 2s')
colormap(gray) ;
axis off ;

subplot (3,4,3) ,
temp = video ( : , 70) ;
temp = reshape(temp , 480,640 ) ;
imagesc(temp) ;
title('t = 3s')
colormap(gray) ;
axis off ;
subplot (3,4,4) ,
temp = video ( : , 90) ;
temp = reshape ( temp , 480,640 ) ;
imagesc(temp) ;
title('t = 4s')
colormap(gray) ;
axis off ;
% for i = 1 : n1
% temp = video ( : , i ) ;
% temp = reshape ( temp , 360, 640) ;
% imagesc ( temp ) ;
% colormap ( gray ) ;
% drawnow
% end
subplot (3,4,5) ,
temp = X_low (:,30) ;
temp = reshape ( temp ,480, 640) ;
imagesc (temp) ;
title('t = 1s')
colormap (gray) ;
axis off ;
subplot (3,4,6 ) ,
temp = X_low ( : , 60 ) ;
temp = reshape ( temp , 480 , 640) ;
imagesc(temp) ;
title('t = 2s')
colormap(gray) ;

axis off ;
subplot ( 3 , 4 , 7 ) ,
temp = X_low ( : , 70 ) ;
temp = reshape ( temp , 480 , 640) ;
imagesc ( temp ) ;
title('t = 3s')
colormap ( gray ) ;
axis off ;
subplot ( 3 , 4 , 8 ) ,
temp = X_low ( : , 90 ) ;
temp = reshape ( temp , 480 , 640) ;
imagesc ( temp ) ;
title('t = 4s')
colormap ( gray ) ;
axis off ;
subplot ( 3 , 4 , 9 ) ,
temp = X_sparse ( : , 30 ) ;
temp = reshape ( temp , 480 , 640 ) ;
imagesc ( temp ) ;
title('t = 1s')
colormap ( gray ) ;
axis off ;
subplot ( 3 , 4 , 10 ) ,
temp = X_sparse ( : , 60 ) ;
temp = reshape ( temp , 480 , 640) ;
imagesc ( temp ) ;
title('t = 2s')
colormap ( gray ) ;
axis off ;
subplot ( 3 , 4 , 11 ) ,
temp = X_sparse ( : , 70 ) ;
temp = reshape ( temp , 480 , 640) ;
imagesc (temp) ;
title('t = 3s')

colormap ( gray ) ;
axis off ;
subplot ( 3 , 4 , 12 ) ,
temp = X_sparse ( : , 90) ;
temp = reshape( temp , 480 , 640) ;
imagesc ( temp ) ;
title('t = 4s')
colormap ( gray ) ;
axis off ;
% 
