
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')

from sklearn.linear_model import LinearRegression

val =  pd.read_csv('sixup.csv')
col_names = list(val.columns.values)
clf = LinearRegression()
input_name = raw_input('enter which value you want to plot, (2-9): ')
Y = np.array(val['Max of Federal Loan 3-Year Default Rate'])
X = np.array(val.icol(int(input_name)))
X = X[:,np.newaxis]
clf.fit(X,Y)


X1 = []
Y1 = []
N = len(X)
for i in range(0,N):
    if (X[i] != 0 and Y[i] != 0):
        
        X1.append(X[i])
        Y1.append(Y[i])
    else:
        X1.append(0)
        Y1.append(0)
        
X1= np.array(X1)       
X1= X1[:,np.newaxis]
plt.plot(X1,Y1,'k.')
plt.plot(X1, clf.predict(X1), color='red')

plt.legend(loc=2)
plt.xlabel(col_names[1])
plt.ylabel(col_names[int(input_name)])
plt.title("%s versus %s" % (col_names[1],col_names[int(input_name)]))

xarray = np.array([i*0.001 for i in range(0,N)])
xarray = xarray[:,np.newaxis]

difference = clf.predict(xarray) - Y1

plt.grid()

#PLOT
fig1 = plt.figure(1)
#Plot Data-model
#Residual plot
frame2=fig1.add_axes((0.15,-.3,.76,.2))        
plt.plot(xarray,difference,'--')
plt.title('Residuals')

plt.grid()
plt.show()
print 'rsquared:',clf.score(X,Y), 'coefficients',clf.coef_