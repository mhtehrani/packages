"""
matplotlib package

https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.html
"""

import matplotlib.pyplot as plt

#figure()
#------------------------------------------------------------------------------
#Create a new figure, or activate an existing figure.
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib.pyplot.figure


#markers
#------------------------------------------------------------------------------
#https://matplotlib.org/3.5.1/api/markers_api.html#module-matplotlib.markers


#colors
#------------------------------------------------------------------------------
#https://matplotlib.org/stable/gallery/color/named_colors.html






#scatter
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.scatter.html#matplotlib.pyplot.scatter
if True:
    #data
    import numpy as np
    x = np.random.randint(-10, high=10, size=20)
    y = np.random.randint(-20, high=20, size=20)
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.hlines(0, xmin=-10, xmax=10, colors='black', alpha=0.5)
    plt.vlines(0, ymin=-20, ymax=20, colors='black', alpha=0.5)
    #plt.scatter(x, y, marker='*', c='r', alpha=0.5)
    plt.scatter(x, y, marker='*', c=y, cmap='plasma', alpha=0.5)
    plt.arrow(0, 0, 4, 8,head_width=1, overhang=-1)
    plt.arrow(0, 0, 4, -8,head_width=1, overhang=1)
    plt.arrow(0, 0, -4, -8,head_width=1, overhang=0)
    plt.arrow(0, 0, -4, 8, head_width=1, overhang=0, shape='left')
    plt.text(4, 8, 'text', color = 'g', ha = 'center', va = 'center')
    plt.xlim((-10, 10)) #or plt.xlim(left=-10, right=10)
    plt.ylim((-20, 20)) #or plt.ylim(bottom=-20, top=20)
    plt.xlabel('X variable', fontsize=20, labelpad=5) #leappad: whitespace between the label and the axis
    plt.ylabel('y variable', fontsize=20, labelpad=0)
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.tick_params(axis='y', which='minor', labelsize=10)
    plt.legend(labels='a', loc='upper right')
    plt.show()



#plot
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.plot.html#matplotlib.pyplot.plot
if True:
    #data
    import numpy as np
    x = np.random.randint(-10, high=10, size=20)
    y = np.random.randint(-20, high=20, size=20)
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=8)
    #plt.plot('xlabel', 'ylabel', data=df) df is pandas dataframe
    plt.xlim((-10, 10)) #or plt.xlim(left=-10, right=10)
    plt.ylim((-20, 20)) #or plt.ylim(bottom=-20, top=20)
    plt.xlabel('X variable', fontsize=20, labelpad=5) #leappad: whitespace between the label and the axis
    plt.ylabel('y variable', fontsize=20, labelpad=0)
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.tick_params(axis='y', which='minor', labelsize=10)
    plt.legend(labels='a')
    plt.show()



#pie
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.pie.html#matplotlib.pyplot.pie
if True:
    #data
    x = [1, 33, 4.2, 8, 23]
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.pie(x, labels=list(map(str,x)), autopct='test', shadow=False)
    plt.legend(labels=['a','b','c','d','e'])
    plt.show()



#bar
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.bar.html#matplotlib.pyplot.bar
if True:
    #data
    import numpy as np
    x = np.arange(-10, 10, 2)
    y = np.random.randint(0, high=20, size=10)
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.hlines(0, xmin=-10, xmax=10, colors='black', alpha=0.5)
    plt.bar(x, y, width=1, color='blue', edgecolor='red', linewidth=2, 
            tick_label=['a','b','c','d','e','f','g','h','i','j'],
            yerr=2, ecolor='green')
    plt.bar(x, -y, width=1, color='blue', edgecolor='red', linewidth=2, 
            tick_label=['a','b','c','d','e','f','g','h','i','j'],
            yerr=2, ecolor='green')
    
    plt.ylim((-20, 20))
    plt.xlabel('X variable', fontsize=20, labelpad=5) #leappad: whitespace between the label and the axis
    plt.ylabel('y variable', fontsize=20, labelpad=0)
    plt.title('bar plot')
    plt.show()



#hist
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.hist.html#matplotlib.pyplot.hist
if True:
    #data
    import numpy as np
    x = np.random.normal(loc=1.5, scale=3.5, size=1000)
    y = np.random.normal(loc=100.5, scale=30.5, size=100)
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    #plt.hist(x, bins=[-15, -10, -5,-4,-3,-2,-1,  0, 2, 4, 6, 8, 10, 15, 20])
    plt.hist(x, bins=10, color='blue', alpha=0.25, linewidth=2, rwidth=0.95)
    plt.hist(x, bins=10, color='blue', histtype='step', linewidth=2)
    plt.hist(y, bins=10, cumulative=False, histtype='step', orientation='horizontal', color='red')
    plt.xlabel('X variable', fontsize=20, labelpad=5) #leappad: whitespace between the label and the axis
    plt.ylabel('counts', fontsize=20, labelpad=0)
    plt.show()



#hist2d
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.hist2d.html#matplotlib.pyplot.hist2d
if True:
    #data
    import numpy as np
    x = np.random.normal(loc=1.5, scale=3.5, size=3000)
    y = np.random.normal(loc=100.5, scale=30.5, size=3000)
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.hist2d(x, y, bins=20, density=True)
    plt.xlabel('X variable', fontsize=20, labelpad=5) #leappad: whitespace between the label and the axis
    plt.ylabel('Y variable', fontsize=20, labelpad=5)
    plt.show()



#boxplot
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.boxplot.html#matplotlib.pyplot.boxplot
if True:
    #data
    import numpy as np
    x = np.random.randint(-10, high=10, size=20)
    y = np.random.randint(-20, high=20, size=20)
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.boxplot([x, y], notch=False, #notch shows the location of confidence intervals
                vert=True, labels=['x','y'], showcaps=False, showfliers=False, showbox=True)
    plt.xlabel('variables', fontsize=20, labelpad=5) #leappad: whitespace between the label and the axis
    plt.ylabel('distribution', fontsize=20, labelpad=0)
    plt.show()



#subplot
###############################################################################
#https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot
if True:
    #data
    import numpy as np
    x = np.random.randint(-100, high=100, size=200)
    y = np.random.randint(-20, high=20, size=200)
    
    plt.figure(num=1, figsize=[6.4, 4.8], dpi=600)
    plt.subplot(2,3,1)
    plt.boxplot(x)
    plt.subplot(2,3,2)
    plt.boxplot(y)
    plt.subplot(2,3,4)
    plt.hist(x)
    plt.subplot(2,3,5)
    plt.hist(y)
    plt.subplot(2,3,3)
    plt.boxplot([x,y])
    plt.subplot(2,3,6)
    plt.hist(x, color='red', alpha=0.4)
    plt.hist(y, color='blue', alpha=0.4)
    plt.show()














