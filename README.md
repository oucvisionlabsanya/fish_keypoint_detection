#The current version of the code is the initial version, the latest version of the code will be updated later.
#There are two types of datasets, target detection and keypoint detection, which overlap but are not identical, in the detectiondata and keypointdata directories respectively
# Stacked-Hourglass-for-keypoint-detection
Keras implement of Stacked Hourglass. The model is for keypoint detection, and can be migrated to other tasks.
If you're using tf2.0 or higher, replace `keras` with `tf.keras` at import.
Way to build a model is exampled in StackedHourglass.py. 
Full train-eval-test pipeline is exampled in test.py.
After inference OKSEvaluator.py can be used to evaluate average OKS of the result.
