import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.lines import Line2D  # Correct import

sample_1 = 100
sample_2 = 10
centers = [[0.0 ,0.0 ], [2.0 , 2.0]]
clusters_std = [1.5 , 0.5]

X , y = make_blobs(
    n_samples=[sample_1 , sample_2],
    centers=centers ,
    cluster_std=clusters_std ,
    random_state=0 ,
    shuffle=False
)
clf=svm.SVC(kernel= 'linear' , C=1.0)
clf.fit(X , y )

wclf=svm.SVC(kernel= 'linear' , class_weight={1:10})
wclf.fit(X , y )

plt.scatter(X[:, 0 ], X [:, 1] , c=y  , cmap= plt.cm.Paired , edgecolors= "k")

ax = plt.gca()
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X, 
    plot_method="contour",
    colors = "g",
    levels= [0],
    alpha = 0.5,
    linestyle=["-"],
    ax=ax,  
)
wdisp = DecisionBoundaryDisplay.from_estimator(
    wclf,
    X, 
    plot_method="contour",
    colors = "r",
    levels= [0],
    alpha = 0.5,
    linestyle=["-"],
    ax=ax,  
)
plt.legend(
    [Line2D([], [], color="g", label="weighted"),
     Line2D([], [], color="r", label="non_weighted")],
    ["non_weighted", "weighted"],
    loc="upper right"
)

plt.show()

