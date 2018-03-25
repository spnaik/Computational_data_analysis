import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from astroquery.sdss import SDSS 
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from astroML.datasets import fetch_sdss_galaxy_colors
from astroML.plotting import scatter_contour
from astroML.datasets import fetch_sdss_spectrum
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

query = """SELECT TOP 20000 
p.objid,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z, p.run, p.rerun, p.camcol, p.field, 
s.specobjid, s.class, s.z as redshift, s.plate, s.mjd, s.fiberid 
FROM PhotoObj AS p 
JOIN SpecObj AS s ON s.bestobjid = p.objid 
WHERE 
p.u BETWEEN 0 AND 19.6 
AND g BETWEEN 0 AND 20 """
        
data = SDSS.query_sql(query).to_pandas()
k= data['class'].value_counts()
k1 = data.columns.values
k2 = data.head()
#print(k)
#print(k1)

################################
#plot the spectrum
# Fetch single spectrum
plate = 2645
mjd = 54477
fiber = 007

spec = fetch_sdss_spectrum(plate, mjd, fiber)
#ax = plt.axes()
#ax.plot(spec.wavelength(), spec.spectrum, '-b', label='spectrum')
#ax.legend(loc=1,fontsize =20)
#ax.set_title('Plate = %(plate)i, MJD = %(mjd)i, Fiber = %(fiber)i' % locals(),fontsize =20)
#ax.set_xlabel(r'$\lambda (\AA)$',fontsize =20)
#ax.set_ylabel('Flux',fontsize =20)
#ax.xaxis.set_tick_params(labelsize=20)
#ax.yaxis.set_tick_params(labelsize=20)
#ax.set_ylim(-10, 20)

#plt.show()

###############################


X=data.drop("class",1)
y=data["class"]
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, random_state=42)

# KNN
clf=KNeighborsClassifier(n_neighbors=10)
clf.fit(train_X, train_y)
#print(clf.score(test_X,test_y))

#Naive Bayes
gnb = GaussianNB()
gnb.fit(train_X, train_y)
#print(gnb.score(test_X,test_y))

#Random Forest
rf = RandomForestClassifier(n_estimators=60)
rf.fit(train_X, train_y)
print(rf.score(test_X,test_y))


#SVM
svc = SVC()
svc.fit(train_X,train_y)
#print(svc.score(test_X,test_y))




#changed code
#data1 = fetch_sdss_galaxy_colors()
ug = train_X['u'] - train_X['g']
gr = train_X['g'] - train_X['r']
ri = train_X['r'] - train_X['i']
iz = train_X['i'] - train_X['z']
#spec_class = data['class']
#stars = (spec_class == 2)
#qsos = (spec_class == 3)
stars = (train_y == 'STAR')
glax = (train_y == 'GALAXY')
qsar = (train_y == 'QSO')
#red = train_X.redshift
#print(red[qsar])
#------------------------------------------------------------
# Plot stars, quasars, galaxies

#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.plot(ug[stars], gr[stars], '.', ms=4, c='b', label='stars')
#ax.plot(ug[qsar], gr[qsar], '.', ms=4, c='r', label='qsos')
#ax.plot(ug[glax], gr[glax], '.', ms=4, c='g', label='galaxy')
#ax.xaxis.set_tick_params(labelsize=20)
#ax.yaxis.set_tick_params(labelsize=20)
#ax.set_title('Distribution of stars, quasars and galaxies in the data',fontsize =20)
#ax.legend(loc=4,fontsize=20)
#ax.set_xlabel('$u-g$',fontsize=20)
#ax.set_ylabel('$g-r$',fontsize=20)

#plt.show()

#data["ug"] = data.psfMag_u - data.psfMag_g
#data["gr"] = data.psfMag_g - data.psfMag_r
#data["ri"] = data.psfMag_r - data.psfMag_i
#data["iz"] = data.psfMag_i - data.psfMag_z

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_y,rf.predict(test_X))
cmNorm = cm.astype(float)/cm.sum(axis = 1)[:,np.newaxis]

#plot the confusion matrix
plt.imshow(cmNorm, interpolation = 'Nearest', cmap = 'rainbow', vmin = 0, vmax = 1)
plt.grid()
plt.colorbar()
tick_names = ["GALAXY", "QSO","STAR"]
tick_marks = np.arange(len(tick_names))
plt.xticks(tick_marks, tick_names, rotation=45)
plt.yticks(tick_marks, tick_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# plot the redshift
#fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(16, 4))
#ax = sns.distplot(data[data['class']=='STAR'].redshift,color="g", bins = 30, ax = axes[0], kde = False)
#ax.set_title('Star',fontsize=20)
#ax = sns.distplot(data[data['class']=='GALAXY'].redshift,color="r", bins = 30, ax = axes[1], kde = False)
#ax.set_title('Galaxy',fontsize=20)
#ax = sns.distplot(data[data['class']=='QSO'].redshift,color="b", bins = 30, ax = axes[2], kde = False)
#ax = ax.set_title('QSO',fontsize=20)

#plt.show()
#star
#fig, axes = plt.subplots(nrows=1, ncols=5,figsize=(16, 4))
#ax = sns.distplot(data[data['class']=='QSO'].u,color="r", bins = 30, ax = axes[0], kde = False)
#ax.set_title('Quasar_u',fontsize=20)
#ax = sns.distplot(data[data['class']=='QSO'].g,color="r", bins = 30, ax = axes[1], kde = False)
#ax.set_title('Quasar_g',fontsize=20)
#ax = sns.distplot(data[data['class']=='QSO'].r,color="r", bins = 30, ax = axes[2], kde = False)
#ax.set_title('Quasar_r',fontsize=20)
#ax = sns.distplot(data[data['class']=='QSO'].i,color="r", bins = 30, ax = axes[3], kde = False)
#ax.set_title('Quasar_i',fontsize=20)
#ax = sns.distplot(data[data['class']=='QSO'].z,color="r", bins = 30, ax = axes[4], kde = False)
#ax.set_title('Quasar_z',fontsize=20)
#plt.show()
#galaxies

