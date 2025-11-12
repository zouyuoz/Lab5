import matplotlib.pyplot as plt
import re

# 你的長字串
long_string = """

Starting training...

Starting epoch 1 / 50
Epoch [1/50], Iter [30/129], Loss: total=1065.105, box=0.423, obj_pos=0.183, obj_neg=1062.799, cls=1.278, num_pos=172.000, num_neg=681236.000
Epoch [1/50], Iter [60/129], Loss: total=285.405, box=0.350, obj_pos=0.398, obj_neg=283.788, cls=0.519, num_pos=208.000, num_neg=681200.000
Epoch [1/50], Iter [90/129], Loss: total=131.050, box=0.397, obj_pos=0.417, obj_neg=129.460, cls=0.380, num_pos=205.000, num_neg=681203.000
Epoch [1/50], Iter [120/129], Loss: total=77.414, box=0.334, obj_pos=0.573, obj_neg=75.809, cls=0.364, num_pos=143.000, num_neg=681265.000
Learning Rate for this epoch: 9.990e-04
Validation Loss: 7823.2653
Updating best val loss: 7823.26528

Starting epoch 2 / 50
Epoch [2/50], Iter [30/129], Loss: total=48.830, box=0.385, obj_pos=0.506, obj_neg=47.227, cls=0.327, num_pos=201.000, num_neg=681207.000
Epoch [2/50], Iter [60/129], Loss: total=36.978, box=0.372, obj_pos=0.601, obj_neg=35.310, cls=0.323, num_pos=189.000, num_neg=681219.000
Epoch [2/50], Iter [90/129], Loss: total=29.677, box=0.337, obj_pos=0.625, obj_neg=28.069, cls=0.308, num_pos=165.000, num_neg=681243.000
Epoch [2/50], Iter [120/129], Loss: total=23.392, box=0.323, obj_pos=0.700, obj_neg=21.752, cls=0.293, num_pos=176.000, num_neg=681232.000
Learning Rate for this epoch: 9.961e-04
Validation Loss: 6336.7121
Updating best val loss: 6336.71210

Starting epoch 3 / 50
Epoch [3/50], Iter [30/129], Loss: total=18.779, box=0.327, obj_pos=0.746, obj_neg=17.082, cls=0.298, num_pos=164.000, num_neg=681244.000
Epoch [3/50], Iter [60/129], Loss: total=16.186, box=0.371, obj_pos=0.621, obj_neg=14.526, cls=0.298, num_pos=176.000, num_neg=681232.000
Epoch [3/50], Iter [90/129], Loss: total=13.641, box=0.342, obj_pos=0.700, obj_neg=11.923, cls=0.335, num_pos=194.000, num_neg=681214.000
Epoch [3/50], Iter [120/129], Loss: total=12.490, box=0.359, obj_pos=0.723, obj_neg=10.726, cls=0.323, num_pos=182.000, num_neg=681226.000
Learning Rate for this epoch: 9.912e-04
Validation Loss: 4811.8909
Updating best val loss: 4811.89091

Starting epoch 4 / 50
Epoch [4/50], Iter [30/129], Loss: total=10.609, box=0.338, obj_pos=0.732, obj_neg=8.892, cls=0.308, num_pos=155.000, num_neg=681253.000
Epoch [4/50], Iter [60/129], Loss: total=9.430, box=0.346, obj_pos=0.742, obj_neg=7.699, cls=0.297, num_pos=180.000, num_neg=681228.000
Epoch [4/50], Iter [90/129], Loss: total=8.784, box=0.344, obj_pos=0.753, obj_neg=7.011, cls=0.332, num_pos=150.000, num_neg=681258.000
Epoch [4/50], Iter [120/129], Loss: total=8.022, box=0.329, obj_pos=0.784, obj_neg=6.274, cls=0.304, num_pos=183.000, num_neg=681225.000
Learning Rate for this epoch: 9.843e-04
Validation Loss: 3485.2035
Updating best val loss: 3485.20352

Starting epoch 5 / 50
Epoch [5/50], Iter [30/129], Loss: total=7.218, box=0.322, obj_pos=0.838, obj_neg=5.421, cls=0.316, num_pos=158.000, num_neg=681250.000
Epoch [5/50], Iter [60/129], Loss: total=6.732, box=0.326, obj_pos=0.836, obj_neg=4.923, cls=0.322, num_pos=181.000, num_neg=681227.000
Epoch [5/50], Iter [90/129], Loss: total=6.322, box=0.356, obj_pos=0.805, obj_neg=4.519, cls=0.287, num_pos=160.000, num_neg=681248.000
Epoch [5/50], Iter [120/129], Loss: total=5.923, box=0.387, obj_pos=0.705, obj_neg=4.132, cls=0.311, num_pos=225.000, num_neg=681183.000
Learning Rate for this epoch: 9.756e-04
Validation Loss: 2437.7064
Updating best val loss: 2437.70640

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:37<00:00,  1.60it/s]0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle (no predictions for this class)
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car (no predictions for this class)
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair (no predictions for this class)
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable (no predictions for this class)
0.00000 AP of class dog (no predictions for this class)
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike (no predictions for this class)
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor (no predictions for this class)
--- MAP: 0.00000 ---
Epoch 5, mAP: 0.00000

Starting epoch 6 / 50

Epoch [6/50], Iter [30/129], Loss: total=5.545, box=0.346, obj_pos=0.786, obj_neg=3.743, cls=0.324, num_pos=181.000, num_neg=681227.000
Epoch [6/50], Iter [60/129], Loss: total=5.223, box=0.350, obj_pos=0.781, obj_neg=3.429, cls=0.313, num_pos=161.000, num_neg=681247.000
Epoch [6/50], Iter [90/129], Loss: total=5.142, box=0.363, obj_pos=0.757, obj_neg=3.330, cls=0.329, num_pos=200.000, num_neg=681208.000
Epoch [6/50], Iter [120/129], Loss: total=4.826, box=0.348, obj_pos=0.821, obj_neg=2.961, cls=0.346, num_pos=184.000, num_neg=681224.000
Learning Rate for this epoch: 9.649e-04
Validation Loss: 1633.4174
Updating best val loss: 1633.41740

Starting epoch 7 / 50
Epoch [7/50], Iter [30/129], Loss: total=4.512, box=0.352, obj_pos=0.797, obj_neg=2.701, cls=0.310, num_pos=198.000, num_neg=681210.000
Epoch [7/50], Iter [60/129], Loss: total=4.445, box=0.334, obj_pos=0.858, obj_neg=2.610, cls=0.310, num_pos=182.000, num_neg=681226.000
Epoch [7/50], Iter [90/129], Loss: total=4.243, box=0.324, obj_pos=0.864, obj_neg=2.432, cls=0.299, num_pos=173.000, num_neg=681235.000
Epoch [7/50], Iter [120/129], Loss: total=4.080, box=0.378, obj_pos=0.759, obj_neg=2.250, cls=0.315, num_pos=184.000, num_neg=681224.000
Learning Rate for this epoch: 9.525e-04
Validation Loss: 1062.5284
Updating best val loss: 1062.52844

Starting epoch 8 / 50
Epoch [8/50], Iter [30/129], Loss: total=3.957, box=0.360, obj_pos=0.791, obj_neg=2.111, cls=0.335, num_pos=172.000, num_neg=681236.000
Epoch [8/50], Iter [60/129], Loss: total=3.815, box=0.348, obj_pos=0.838, obj_neg=1.973, cls=0.308, num_pos=199.000, num_neg=681209.000
Epoch [8/50], Iter [90/129], Loss: total=3.737, box=0.325, obj_pos=0.898, obj_neg=1.898, cls=0.292, num_pos=170.000, num_neg=681238.000
Epoch [8/50], Iter [120/129], Loss: total=3.638, box=0.359, obj_pos=0.814, obj_neg=1.782, cls=0.325, num_pos=156.000, num_neg=681252.000
Learning Rate for this epoch: 9.382e-04
Validation Loss: 676.3524
Updating best val loss: 676.35244

Starting epoch 9 / 50
Epoch [9/50], Iter [30/129], Loss: total=3.520, box=0.364, obj_pos=0.802, obj_neg=1.675, cls=0.316, num_pos=189.000, num_neg=681219.000
Epoch [9/50], Iter [60/129], Loss: total=3.431, box=0.378, obj_pos=0.803, obj_neg=1.589, cls=0.282, num_pos=217.000, num_neg=681191.000
Epoch [9/50], Iter [90/129], Loss: total=3.383, box=0.353, obj_pos=0.828, obj_neg=1.533, cls=0.317, num_pos=188.000, num_neg=681220.000
Epoch [9/50], Iter [120/129], Loss: total=3.300, box=0.387, obj_pos=0.788, obj_neg=1.448, cls=0.290, num_pos=177.000, num_neg=681231.000
Learning Rate for this epoch: 9.222e-04
Validation Loss: 425.1751
Updating best val loss: 425.17509

Starting epoch 10 / 50
Epoch [10/50], Iter [30/129], Loss: total=3.265, box=0.318, obj_pos=0.925, obj_neg=1.410, cls=0.293, num_pos=167.000, num_neg=681241.000
Epoch [10/50], Iter [60/129], Loss: total=3.202, box=0.331, obj_pos=0.904, obj_neg=1.304, cls=0.331, num_pos=166.000, num_neg=681242.000
Epoch [10/50], Iter [90/129], Loss: total=3.179, box=0.351, obj_pos=0.870, obj_neg=1.316, cls=0.291, num_pos=220.000, num_neg=681188.000
Epoch [10/50], Iter [120/129], Loss: total=3.072, box=0.329, obj_pos=0.912, obj_neg=1.205, cls=0.297, num_pos=178.000, num_neg=681230.000
Learning Rate for this epoch: 9.046e-04
Validation Loss: 268.0803
Updating best val loss: 268.08032

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:37<00:00,  1.58it/s]0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle (no predictions for this class)
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car (no predictions for this class)
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair (no predictions for this class)
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable (no predictions for this class)
0.00000 AP of class dog (no predictions for this class)
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike (no predictions for this class)
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor (no predictions for this class)
--- MAP: 0.00000 ---
Epoch 10, mAP: 0.00000

Starting epoch 11 / 50

Epoch [11/50], Iter [30/129], Loss: total=2.990, box=0.363, obj_pos=0.827, obj_neg=1.142, cls=0.296, num_pos=159.000, num_neg=681249.000
Epoch [11/50], Iter [60/129], Loss: total=2.963, box=0.345, obj_pos=0.856, obj_neg=1.110, cls=0.307, num_pos=203.000, num_neg=681205.000
Epoch [11/50], Iter [90/129], Loss: total=2.929, box=0.322, obj_pos=0.934, obj_neg=1.051, cls=0.299, num_pos=188.000, num_neg=681220.000
Epoch [11/50], Iter [120/129], Loss: total=2.903, box=0.346, obj_pos=0.865, obj_neg=1.040, cls=0.306, num_pos=196.000, num_neg=681212.000
Learning Rate for this epoch: 8.854e-04
Validation Loss: 169.7027
Updating best val loss: 169.70268

Starting epoch 12 / 50
Epoch [12/50], Iter [30/129], Loss: total=2.880, box=0.332, obj_pos=0.925, obj_neg=0.971, cls=0.321, num_pos=154.000, num_neg=681254.000
Epoch [12/50], Iter [60/129], Loss: total=2.868, box=0.349, obj_pos=0.890, obj_neg=0.942, cls=0.337, num_pos=147.000, num_neg=681261.000
Epoch [12/50], Iter [90/129], Loss: total=2.765, box=0.352, obj_pos=0.871, obj_neg=0.897, cls=0.293, num_pos=205.000, num_neg=681203.000
Epoch [12/50], Iter [120/129], Loss: total=2.774, box=0.361, obj_pos=0.859, obj_neg=0.881, cls=0.313, num_pos=221.000, num_neg=681187.000
Learning Rate for this epoch: 8.646e-04
Validation Loss: 107.7064
Updating best val loss: 107.70636

Starting epoch 13 / 50
Epoch [13/50], Iter [30/129], Loss: total=2.706, box=0.363, obj_pos=0.827, obj_neg=0.844, cls=0.310, num_pos=210.000, num_neg=681198.000
Epoch [13/50], Iter [60/129], Loss: total=2.713, box=0.378, obj_pos=0.840, obj_neg=0.822, cls=0.295, num_pos=216.000, num_neg=681192.000
Epoch [13/50], Iter [90/129], Loss: total=2.665, box=0.352, obj_pos=0.870, obj_neg=0.783, cls=0.306, num_pos=208.000, num_neg=681200.000
Epoch [13/50], Iter [120/129], Loss: total=2.656, box=0.342, obj_pos=0.897, obj_neg=0.764, cls=0.312, num_pos=179.000, num_neg=681229.000
Learning Rate for this epoch: 8.424e-04
Validation Loss: 69.2038
Updating best val loss: 69.20377

Starting epoch 14 / 50
Epoch [14/50], Iter [30/129], Loss: total=2.658, box=0.352, obj_pos=0.876, obj_neg=0.759, cls=0.318, num_pos=176.000, num_neg=681232.000
Epoch [14/50], Iter [60/129], Loss: total=2.617, box=0.331, obj_pos=0.931, obj_neg=0.716, cls=0.307, num_pos=170.000, num_neg=681238.000
Epoch [14/50], Iter [90/129], Loss: total=2.616, box=0.345, obj_pos=0.899, obj_neg=0.697, cls=0.330, num_pos=189.000, num_neg=681219.000
Epoch [14/50], Iter [120/129], Loss: total=2.573, box=0.375, obj_pos=0.838, obj_neg=0.685, cls=0.301, num_pos=174.000, num_neg=681234.000
Learning Rate for this epoch: 8.189e-04
Validation Loss: 45.3201
Updating best val loss: 45.32015

Starting epoch 15 / 50
Epoch [15/50], Iter [30/129], Loss: total=2.521, box=0.395, obj_pos=0.758, obj_neg=0.664, cls=0.310, num_pos=189.000, num_neg=681219.000
Epoch [15/50], Iter [60/129], Loss: total=2.574, box=0.357, obj_pos=0.875, obj_neg=0.658, cls=0.327, num_pos=178.000, num_neg=681230.000
Epoch [15/50], Iter [90/129], Loss: total=2.524, box=0.362, obj_pos=0.859, obj_neg=0.629, cls=0.313, num_pos=225.000, num_neg=681183.000
Epoch [15/50], Iter [120/129], Loss: total=2.491, box=0.362, obj_pos=0.854, obj_neg=0.602, cls=0.311, num_pos=171.000, num_neg=681237.000
Learning Rate for this epoch: 7.941e-04
Validation Loss: 30.4563
Updating best val loss: 30.45627

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:36<00:00,  1.65it/s]0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle (no predictions for this class)
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car (no predictions for this class)
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair (no predictions for this class)
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable (no predictions for this class)
0.00000 AP of class dog (no predictions for this class)
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike (no predictions for this class)
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor (no predictions for this class)
--- MAP: 0.00000 ---
Epoch 15, mAP: 0.00000

Starting epoch 16 / 50

Epoch [16/50], Iter [30/129], Loss: total=2.485, box=0.329, obj_pos=0.946, obj_neg=0.585, cls=0.296, num_pos=179.000, num_neg=681229.000
Epoch [16/50], Iter [60/129], Loss: total=2.491, box=0.341, obj_pos=0.916, obj_neg=0.587, cls=0.307, num_pos=187.000, num_neg=681221.000
Epoch [16/50], Iter [90/129], Loss: total=2.439, box=0.351, obj_pos=0.877, obj_neg=0.570, cls=0.290, num_pos=214.000, num_neg=681194.000
Epoch [16/50], Iter [120/129], Loss: total=2.479, box=0.357, obj_pos=0.886, obj_neg=0.545, cls=0.334, num_pos=183.000, num_neg=681225.000
Learning Rate for this epoch: 7.681e-04
Validation Loss: 21.0545
Updating best val loss: 21.05448

Starting epoch 17 / 50
Epoch [17/50], Iter [30/129], Loss: total=2.400, box=0.369, obj_pos=0.843, obj_neg=0.529, cls=0.290, num_pos=181.000, num_neg=681227.000
Epoch [17/50], Iter [60/129], Loss: total=2.415, box=0.357, obj_pos=0.858, obj_neg=0.531, cls=0.312, num_pos=207.000, num_neg=681201.000
Epoch [17/50], Iter [90/129], Loss: total=2.428, box=0.326, obj_pos=0.940, obj_neg=0.507, cls=0.328, num_pos=183.000, num_neg=681225.000
Epoch [17/50], Iter [120/129], Loss: total=2.434, box=0.373, obj_pos=0.905, obj_neg=0.495, cls=0.288, num_pos=190.000, num_neg=681218.000
Learning Rate for this epoch: 7.411e-04
Validation Loss: 15.0359
Updating best val loss: 15.03591

Starting epoch 18 / 50
Epoch [18/50], Iter [30/129], Loss: total=2.358, box=0.358, obj_pos=0.872, obj_neg=0.484, cls=0.286, num_pos=210.000, num_neg=681198.000
Epoch [18/50], Iter [60/129], Loss: total=2.451, box=0.300, obj_pos=1.043, obj_neg=0.493, cls=0.315, num_pos=187.000, num_neg=681221.000
Epoch [18/50], Iter [90/129], Loss: total=2.367, box=0.344, obj_pos=0.920, obj_neg=0.461, cls=0.297, num_pos=172.000, num_neg=681236.000
Epoch [18/50], Iter [120/129], Loss: total=2.379, box=0.323, obj_pos=0.981, obj_neg=0.463, cls=0.289, num_pos=167.000, num_neg=681241.000
Learning Rate for this epoch: 7.132e-04
Validation Loss: 11.1035
Updating best val loss: 11.10347

Starting epoch 19 / 50
Epoch [19/50], Iter [30/129], Loss: total=2.414, box=0.362, obj_pos=0.901, obj_neg=0.463, cls=0.327, num_pos=152.000, num_neg=681256.000
Epoch [19/50], Iter [60/129], Loss: total=2.330, box=0.319, obj_pos=0.956, obj_neg=0.436, cls=0.300, num_pos=168.000, num_neg=681240.000
Epoch [19/50], Iter [90/129], Loss: total=2.376, box=0.365, obj_pos=0.875, obj_neg=0.469, cls=0.302, num_pos=246.000, num_neg=681162.000
Epoch [19/50], Iter [120/129], Loss: total=2.350, box=0.337, obj_pos=0.944, obj_neg=0.428, cls=0.304, num_pos=196.000, num_neg=681212.000
Learning Rate for this epoch: 6.844e-04
Validation Loss: 8.4981
Updating best val loss: 8.49813

Starting epoch 20 / 50
Epoch [20/50], Iter [30/129], Loss: total=2.318, box=0.333, obj_pos=0.918, obj_neg=0.420, cls=0.314, num_pos=164.000, num_neg=681244.000
Epoch [20/50], Iter [60/129], Loss: total=2.310, box=0.313, obj_pos=0.966, obj_neg=0.411, cls=0.306, num_pos=145.000, num_neg=681263.000
Epoch [20/50], Iter [90/129], Loss: total=2.284, box=0.381, obj_pos=0.820, obj_neg=0.406, cls=0.295, num_pos=194.000, num_neg=681214.000
Epoch [20/50], Iter [120/129], Loss: total=2.261, box=0.348, obj_pos=0.889, obj_neg=0.387, cls=0.289, num_pos=203.000, num_neg=681205.000
Learning Rate for this epoch: 6.549e-04
Validation Loss: 6.7401
Updating best val loss: 6.74006

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:36<00:00,  1.64it/s]0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle (no predictions for this class)
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car (no predictions for this class)
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair (no predictions for this class)
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable (no predictions for this class)
0.00000 AP of class dog (no predictions for this class)
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike (no predictions for this class)
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor (no predictions for this class)
--- MAP: 0.00000 ---
Epoch 20, mAP: 0.00000

Starting epoch 21 / 50

Epoch [21/50], Iter [30/129], Loss: total=2.261, box=0.387, obj_pos=0.798, obj_neg=0.396, cls=0.293, num_pos=193.000, num_neg=681215.000
Epoch [21/50], Iter [60/129], Loss: total=2.309, box=0.355, obj_pos=0.921, obj_neg=0.380, cls=0.297, num_pos=206.000, num_neg=681202.000
Epoch [21/50], Iter [90/129], Loss: total=2.286, box=0.313, obj_pos=0.970, obj_neg=0.375, cls=0.315, num_pos=180.000, num_neg=681228.000
Epoch [21/50], Iter [120/129], Loss: total=2.272, box=0.341, obj_pos=0.925, obj_neg=0.376, cls=0.290, num_pos=197.000, num_neg=681211.000
Learning Rate for this epoch: 6.247e-04
Validation Loss: 5.5318
Updating best val loss: 5.53181

Starting epoch 22 / 50
Epoch [22/50], Iter [30/129], Loss: total=2.244, box=0.318, obj_pos=0.971, obj_neg=0.361, cls=0.275, num_pos=154.000, num_neg=681254.000
Epoch [22/50], Iter [60/129], Loss: total=2.292, box=0.327, obj_pos=0.968, obj_neg=0.362, cls=0.307, num_pos=180.000, num_neg=681228.000
Epoch [22/50], Iter [90/129], Loss: total=2.294, box=0.311, obj_pos=1.018, obj_neg=0.360, cls=0.294, num_pos=177.000, num_neg=681231.000
Epoch [22/50], Iter [120/129], Loss: total=2.213, box=0.359, obj_pos=0.869, obj_neg=0.353, cls=0.273, num_pos=212.000, num_neg=681196.000
Learning Rate for this epoch: 5.941e-04
Validation Loss: 4.6817
Updating best val loss: 4.68169

Starting epoch 23 / 50
Epoch [23/50], Iter [30/129], Loss: total=2.264, box=0.340, obj_pos=0.925, obj_neg=0.334, cls=0.323, num_pos=169.000, num_neg=681239.000
Epoch [23/50], Iter [60/129], Loss: total=2.271, box=0.333, obj_pos=0.959, obj_neg=0.338, cls=0.310, num_pos=210.000, num_neg=681198.000
Epoch [23/50], Iter [90/129], Loss: total=2.233, box=0.330, obj_pos=0.933, obj_neg=0.328, cls=0.312, num_pos=163.000, num_neg=681245.000
Epoch [23/50], Iter [120/129], Loss: total=2.258, box=0.324, obj_pos=0.988, obj_neg=0.330, cls=0.292, num_pos=191.000, num_neg=681217.000
Learning Rate for this epoch: 5.631e-04
Validation Loss: 6.0280

Starting epoch 24 / 50
Epoch [24/50], Iter [30/129], Loss: total=2.236, box=0.332, obj_pos=0.928, obj_neg=0.319, cls=0.326, num_pos=162.000, num_neg=681246.000
Epoch [24/50], Iter [60/129], Loss: total=2.225, box=0.297, obj_pos=1.010, obj_neg=0.322, cls=0.299, num_pos=141.000, num_neg=681267.000
Epoch [24/50], Iter [90/129], Loss: total=2.215, box=0.346, obj_pos=0.903, obj_neg=0.324, cls=0.297, num_pos=182.000, num_neg=681226.000
Epoch [24/50], Iter [120/129], Loss: total=2.229, box=0.295, obj_pos=1.009, obj_neg=0.320, cls=0.312, num_pos=159.000, num_neg=681249.000
Learning Rate for this epoch: 5.319e-04
Validation Loss: 33.4708

Starting epoch 25 / 50
Epoch [25/50], Iter [30/129], Loss: total=2.236, box=0.324, obj_pos=0.971, obj_neg=0.322, cls=0.295, num_pos=157.000, num_neg=681251.000
Epoch [25/50], Iter [60/129], Loss: total=2.208, box=0.324, obj_pos=0.981, obj_neg=0.309, cls=0.270, num_pos=193.000, num_neg=681215.000
Epoch [25/50], Iter [90/129], Loss: total=2.230, box=0.341, obj_pos=0.957, obj_neg=0.301, cls=0.291, num_pos=221.000, num_neg=681187.000
Epoch [25/50], Iter [120/129], Loss: total=2.183, box=0.352, obj_pos=0.908, obj_neg=0.302, cls=0.269, num_pos=200.000, num_neg=681208.000
Learning Rate for this epoch: 5.005e-04
Validation Loss: 71.0217

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:37<00:00,  1.58it/s]0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car (no predictions for this class)
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable
0.00000 AP of class dog
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor
--- MAP: 0.00000 ---
Epoch 25, mAP: 0.00000

Starting epoch 26 / 50

Epoch [26/50], Iter [30/129], Loss: total=2.205, box=0.280, obj_pos=1.069, obj_neg=0.294, cls=0.281, num_pos=162.000, num_neg=681246.000
Epoch [26/50], Iter [60/129], Loss: total=2.193, box=0.324, obj_pos=0.964, obj_neg=0.286, cls=0.296, num_pos=160.000, num_neg=681248.000
Epoch [26/50], Iter [90/129], Loss: total=2.267, box=0.318, obj_pos=1.031, obj_neg=0.290, cls=0.311, num_pos=191.000, num_neg=681217.000
Epoch [26/50], Iter [120/129], Loss: total=2.221, box=0.337, obj_pos=0.974, obj_neg=0.280, cls=0.292, num_pos=212.000, num_neg=681196.000
Learning Rate for this epoch: 4.691e-04
Validation Loss: 104.6710

Starting epoch 27 / 50
Epoch [27/50], Iter [30/129], Loss: total=2.173, box=0.358, obj_pos=0.889, obj_neg=0.286, cls=0.282, num_pos=223.000, num_neg=681185.000
Epoch [27/50], Iter [60/129], Loss: total=2.226, box=0.335, obj_pos=0.980, obj_neg=0.287, cls=0.289, num_pos=206.000, num_neg=681202.000
Epoch [27/50], Iter [90/129], Loss: total=2.205, box=0.338, obj_pos=0.957, obj_neg=0.278, cls=0.293, num_pos=188.000, num_neg=681220.000
Epoch [27/50], Iter [120/129], Loss: total=2.233, box=0.279, obj_pos=1.084, obj_neg=0.278, cls=0.312, num_pos=175.000, num_neg=681233.000
Learning Rate for this epoch: 4.379e-04
Validation Loss: 138.2057

Starting epoch 28 / 50
Epoch [28/50], Iter [30/129], Loss: total=2.213, box=0.297, obj_pos=1.071, obj_neg=0.265, cls=0.283, num_pos=169.000, num_neg=681239.000
Epoch [28/50], Iter [60/129], Loss: total=2.194, box=0.313, obj_pos=1.009, obj_neg=0.275, cls=0.284, num_pos=167.000, num_neg=681241.000
Epoch [28/50], Iter [90/129], Loss: total=2.166, box=0.316, obj_pos=0.982, obj_neg=0.261, cls=0.291, num_pos=155.000, num_neg=681253.000
Epoch [28/50], Iter [120/129], Loss: total=2.161, box=0.330, obj_pos=0.941, obj_neg=0.266, cls=0.294, num_pos=159.000, num_neg=681249.000
Learning Rate for this epoch: 4.069e-04
Validation Loss: 195.3300

Starting epoch 29 / 50
Epoch [29/50], Iter [30/129], Loss: total=2.240, box=0.326, obj_pos=1.003, obj_neg=0.263, cls=0.322, num_pos=176.000, num_neg=681232.000
Epoch [29/50], Iter [60/129], Loss: total=2.199, box=0.323, obj_pos=0.972, obj_neg=0.249, cls=0.332, num_pos=163.000, num_neg=681245.000
Epoch [29/50], Iter [90/129], Loss: total=2.185, box=0.352, obj_pos=0.934, obj_neg=0.249, cls=0.297, num_pos=171.000, num_neg=681237.000
Epoch [29/50], Iter [120/129], Loss: total=2.130, box=0.333, obj_pos=0.888, obj_neg=0.265, cls=0.312, num_pos=160.000, num_neg=681248.000
Learning Rate for this epoch: 3.763e-04
Validation Loss: 265.5912

Starting epoch 30 / 50
Epoch [30/50], Iter [30/129], Loss: total=2.222, box=0.325, obj_pos=1.000, obj_neg=0.256, cls=0.316, num_pos=187.000, num_neg=681221.000
Epoch [30/50], Iter [60/129], Loss: total=2.202, box=0.321, obj_pos=0.980, obj_neg=0.264, cls=0.316, num_pos=161.000, num_neg=681247.000
Epoch [30/50], Iter [90/129], Loss: total=2.189, box=0.334, obj_pos=0.951, obj_neg=0.249, cls=0.320, num_pos=152.000, num_neg=681256.000
Epoch [30/50], Iter [120/129], Loss: total=2.182, box=0.373, obj_pos=0.879, obj_neg=0.244, cls=0.312, num_pos=176.000, num_neg=681232.000
Learning Rate for this epoch: 3.461e-04
Validation Loss: 351.0126

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:37<00:00,  1.61it/s]
0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle (no predictions for this class)
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car (no predictions for this class)
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair (no predictions for this class)
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable (no predictions for this class)
0.00000 AP of class dog
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor
--- MAP: 0.00000 ---
Epoch 30, mAP: 0.00000

Starting epoch 31 / 50
Epoch [31/50], Iter [30/129], Loss: total=2.215, box=0.301, obj_pos=1.046, obj_neg=0.240, cls=0.327, num_pos=160.000, num_neg=681248.000
Epoch [31/50], Iter [60/129], Loss: total=2.173, box=0.356, obj_pos=0.893, obj_neg=0.260, cls=0.308, num_pos=182.000, num_neg=681226.000
Epoch [31/50], Iter [90/129], Loss: total=2.172, box=0.345, obj_pos=0.936, obj_neg=0.247, cls=0.299, num_pos=170.000, num_neg=681238.000
Epoch [31/50], Iter [120/129], Loss: total=2.231, box=0.293, obj_pos=1.083, obj_neg=0.245, cls=0.318, num_pos=207.000, num_neg=681201.000
Learning Rate for this epoch: 3.166e-04
Validation Loss: 464.2208

Starting epoch 32 / 50
Epoch [32/50], Iter [30/129], Loss: total=2.149, box=0.324, obj_pos=0.968, obj_neg=0.263, cls=0.270, num_pos=184.000, num_neg=681224.000
Epoch [32/50], Iter [60/129], Loss: total=2.146, box=0.362, obj_pos=0.890, obj_neg=0.244, cls=0.288, num_pos=213.000, num_neg=681195.000
Epoch [32/50], Iter [90/129], Loss: total=2.163, box=0.332, obj_pos=0.949, obj_neg=0.231, cls=0.318, num_pos=187.000, num_neg=681221.000
Epoch [32/50], Iter [120/129], Loss: total=2.161, box=0.354, obj_pos=0.917, obj_neg=0.243, cls=0.293, num_pos=206.000, num_neg=681202.000
Learning Rate for this epoch: 2.878e-04
Validation Loss: 591.0277

Starting epoch 33 / 50
Epoch [33/50], Iter [30/129], Loss: total=2.121, box=0.333, obj_pos=0.910, obj_neg=0.256, cls=0.289, num_pos=178.000, num_neg=681230.000
Epoch [33/50], Iter [60/129], Loss: total=2.168, box=0.330, obj_pos=0.968, obj_neg=0.233, cls=0.308, num_pos=211.000, num_neg=681197.000
Epoch [33/50], Iter [90/129], Loss: total=2.149, box=0.304, obj_pos=1.031, obj_neg=0.230, cls=0.280, num_pos=159.000, num_neg=681249.000
Epoch [33/50], Iter [120/129], Loss: total=2.141, box=0.287, obj_pos=1.051, obj_neg=0.244, cls=0.271, num_pos=164.000, num_neg=681244.000
Learning Rate for this epoch: 2.599e-04
Validation Loss: 680.0600

Starting epoch 34 / 50
Epoch [34/50], Iter [30/129], Loss: total=2.124, box=0.325, obj_pos=0.951, obj_neg=0.228, cls=0.295, num_pos=155.000, num_neg=681253.000
Epoch [34/50], Iter [60/129], Loss: total=2.119, box=0.309, obj_pos=0.960, obj_neg=0.252, cls=0.288, num_pos=134.000, num_neg=681274.000
Epoch [34/50], Iter [90/129], Loss: total=2.181, box=0.317, obj_pos=1.025, obj_neg=0.228, cls=0.295, num_pos=183.000, num_neg=681225.000
Epoch [34/50], Iter [120/129], Loss: total=2.172, box=0.317, obj_pos=0.997, obj_neg=0.220, cls=0.320, num_pos=159.000, num_neg=681249.000
Learning Rate for this epoch: 2.329e-04
Validation Loss: 740.0390

Starting epoch 35 / 50
Epoch [35/50], Iter [30/129], Loss: total=2.144, box=0.347, obj_pos=0.927, obj_neg=0.226, cls=0.298, num_pos=160.000, num_neg=681248.000
Epoch [35/50], Iter [60/129], Loss: total=2.252, box=0.318, obj_pos=1.072, obj_neg=0.233, cls=0.311, num_pos=233.000, num_neg=681175.000
Epoch [35/50], Iter [90/129], Loss: total=2.153, box=0.319, obj_pos=0.992, obj_neg=0.219, cls=0.304, num_pos=180.000, num_neg=681228.000
Epoch [35/50], Iter [120/129], Loss: total=2.111, box=0.303, obj_pos=0.994, obj_neg=0.223, cls=0.288, num_pos=165.000, num_neg=681243.000
Learning Rate for this epoch: 2.069e-04
Validation Loss: 816.0749

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:39<00:00,  1.53it/s]0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle (no predictions for this class)
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair (no predictions for this class)
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable (no predictions for this class)
0.00000 AP of class dog (no predictions for this class)
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor
--- MAP: 0.00000 ---
Epoch 35, mAP: 0.00000

Starting epoch 36 / 50

Epoch [36/50], Iter [30/129], Loss: total=2.108, box=0.322, obj_pos=0.959, obj_neg=0.216, cls=0.289, num_pos=163.000, num_neg=681245.000
Epoch [36/50], Iter [60/129], Loss: total=2.106, box=0.360, obj_pos=0.882, obj_neg=0.220, cls=0.284, num_pos=215.000, num_neg=681193.000
Epoch [36/50], Iter [90/129], Loss: total=2.140, box=0.331, obj_pos=0.956, obj_neg=0.219, cls=0.302, num_pos=168.000, num_neg=681240.000
Epoch [36/50], Iter [120/129], Loss: total=2.150, box=0.301, obj_pos=1.010, obj_neg=0.215, cls=0.323, num_pos=150.000, num_neg=681258.000
Learning Rate for this epoch: 1.821e-04
Validation Loss: 849.2801

Starting epoch 37 / 50
Epoch [37/50], Iter [30/129], Loss: total=2.148, box=0.353, obj_pos=0.910, obj_neg=0.226, cls=0.306, num_pos=197.000, num_neg=681211.000
Epoch [37/50], Iter [60/129], Loss: total=2.156, box=0.303, obj_pos=1.042, obj_neg=0.222, cls=0.286, num_pos=163.000, num_neg=681245.000
Epoch [37/50], Iter [90/129], Loss: total=2.107, box=0.335, obj_pos=0.927, obj_neg=0.221, cls=0.289, num_pos=170.000, num_neg=681238.000
Epoch [37/50], Iter [120/129], Loss: total=2.190, box=0.355, obj_pos=0.953, obj_neg=0.210, cls=0.318, num_pos=190.000, num_neg=681218.000
Learning Rate for this epoch: 1.586e-04
Validation Loss: 871.2166

Starting epoch 38 / 50
Epoch [38/50], Iter [30/129], Loss: total=2.128, box=0.329, obj_pos=0.992, obj_neg=0.205, cls=0.271, num_pos=226.000, num_neg=681182.000
Epoch [38/50], Iter [60/129], Loss: total=2.175, box=0.317, obj_pos=1.004, obj_neg=0.214, cls=0.323, num_pos=185.000, num_neg=681223.000
Epoch [38/50], Iter [90/129], Loss: total=2.085, box=0.356, obj_pos=0.861, obj_neg=0.215, cls=0.296, num_pos=166.000, num_neg=681242.000
Epoch [38/50], Iter [120/129], Loss: total=2.099, box=0.314, obj_pos=0.963, obj_neg=0.207, cls=0.300, num_pos=141.000, num_neg=681267.000
Learning Rate for this epoch: 1.364e-04
Validation Loss: 962.8017

Starting epoch 39 / 50
Epoch [39/50], Iter [30/129], Loss: total=2.125, box=0.327, obj_pos=0.970, obj_neg=0.213, cls=0.287, num_pos=185.000, num_neg=681223.000
Epoch [39/50], Iter [60/129], Loss: total=2.143, box=0.314, obj_pos=1.012, obj_neg=0.205, cls=0.298, num_pos=164.000, num_neg=681244.000
Epoch [39/50], Iter [90/129], Loss: total=2.138, box=0.314, obj_pos=1.013, obj_neg=0.208, cls=0.289, num_pos=161.000, num_neg=681247.000
Epoch [39/50], Iter [120/129], Loss: total=2.127, box=0.301, obj_pos=1.011, obj_neg=0.207, cls=0.307, num_pos=167.000, num_neg=681241.000
Learning Rate for this epoch: 1.156e-04
Validation Loss: 1142.7054

Starting epoch 40 / 50
Epoch [40/50], Iter [30/129], Loss: total=2.139, box=0.345, obj_pos=0.942, obj_neg=0.206, cls=0.301, num_pos=167.000, num_neg=681241.000
Epoch [40/50], Iter [60/129], Loss: total=2.110, box=0.349, obj_pos=0.917, obj_neg=0.211, cls=0.285, num_pos=204.000, num_neg=681204.000
Epoch [40/50], Iter [90/129], Loss: total=2.152, box=0.360, obj_pos=0.893, obj_neg=0.206, cls=0.332, num_pos=218.000, num_neg=681190.000
Epoch [40/50], Iter [120/129], Loss: total=2.156, box=0.309, obj_pos=1.009, obj_neg=0.221, cls=0.308, num_pos=165.000, num_neg=681243.000
Learning Rate for this epoch: 9.640e-05
Validation Loss: 1080.6939

Evaluating on validation set...
---Evaluate model on validation samples---
100%|██████████| 60/60 [00:38<00:00,  1.55it/s]
0.00000 AP of class aeroplane (no predictions for this class)
0.00000 AP of class bicycle (no predictions for this class)
0.00000 AP of class bird (no predictions for this class)
0.00000 AP of class boat (no predictions for this class)
0.00000 AP of class bottle (no predictions for this class)
0.00000 AP of class bus (no predictions for this class)
0.00000 AP of class car
0.00000 AP of class cat (no predictions for this class)
0.00000 AP of class chair (no predictions for this class)
0.00000 AP of class cow (no predictions for this class)
0.00000 AP of class diningtable (no predictions for this class)
0.00000 AP of class dog (no predictions for this class)
0.00000 AP of class horse (no predictions for this class)
0.00000 AP of class motorbike (no predictions for this class)
0.00000 AP of class person (no predictions for this class)
0.00000 AP of class pottedplant (no predictions for this class)
0.00000 AP of class sheep (no predictions for this class)
0.00000 AP of class sofa (no predictions for this class)
0.00000 AP of class train (no predictions for this class)
0.00000 AP of class tvmonitor
--- MAP: 0.00000 ---
Epoch 40, mAP: 0.00000

Starting epoch 41 / 50
Epoch [41/50], Iter [30/129], Loss: total=2.151, box=0.287, obj_pos=1.111, obj_neg=0.195, cls=0.272, num_pos=204.000, num_neg=681204.000
Epoch [41/50], Iter [60/129], Loss: total=2.131, box=0.315, obj_pos=0.987, obj_neg=0.215, cls=0.300, num_pos=201.000, num_neg=681207.000
Epoch [41/50], Iter [90/129], Loss: total=2.190, box=0.297, obj_pos=1.068, obj_neg=0.216, cls=0.312, num_pos=173.000, num_neg=681235.000
Epoch [41/50], Iter [120/129], Loss: total=2.141, box=0.349, obj_pos=0.950, obj_neg=0.207, cls=0.285, num_pos=154.000, num_neg=681254.000
Learning Rate for this epoch: 7.876e-05
Validation Loss: 1165.4194

Starting epoch 42 / 50
Epoch [42/50], Iter [30/129], Loss: total=2.098, box=0.322, obj_pos=0.943, obj_neg=0.212, cls=0.300, num_pos=154.000, num_neg=681254.000
Epoch [42/50], Iter [60/129], Loss: total=2.152, box=0.312, obj_pos=1.031, obj_neg=0.211, cls=0.285, num_pos=209.000, num_neg=681199.000
Epoch [42/50], Iter [90/129], Loss: total=2.146, box=0.323, obj_pos=0.979, obj_neg=0.219, cls=0.301, num_pos=190.000, num_neg=681218.000
Epoch [42/50], Iter [120/129], Loss: total=2.085, box=0.367, obj_pos=0.865, obj_neg=0.207, cls=0.279, num_pos=212.000, num_neg=681196.000
Learning Rate for this epoch: 6.278e-05
Validation Loss: 1120.6691

Starting epoch 43 / 50
Epoch [43/50], Iter [30/129], Loss: total=2.104, box=0.353, obj_pos=0.877, obj_neg=0.220, cls=0.302, num_pos=198.000, num_neg=681210.000
Epoch [43/50], Iter [60/129], Loss: total=2.102, box=0.302, obj_pos=1.023, obj_neg=0.203, cls=0.272, num_pos=182.000, num_neg=681226.000
Epoch [43/50], Iter [90/129], Loss: total=2.075, box=0.340, obj_pos=0.915, obj_neg=0.201, cls=0.279, num_pos=199.000, num_neg=681209.000
Epoch [43/50], Iter [120/129], Loss: total=2.152, box=0.300, obj_pos=1.050, obj_neg=0.213, cls=0.290, num_pos=176.000, num_neg=681232.000
Learning Rate for this epoch: 4.854e-05
Validation Loss: 1048.0227

Starting epoch 44 / 50
Epoch [44/50], Iter [30/129], Loss: total=2.097, box=0.284, obj_pos=1.033, obj_neg=0.202, cls=0.293, num_pos=158.000, num_neg=681250.000
Epoch [44/50], Iter [60/129], Loss: total=2.145, box=0.299, obj_pos=1.028, obj_neg=0.203, cls=0.317, num_pos=144.000, num_neg=681264.000
Epoch [44/50], Iter [90/129], Loss: total=2.180, box=0.328, obj_pos=1.001, obj_neg=0.214, cls=0.310, num_pos=229.000, num_neg=681179.000
Epoch [44/50], Iter [120/129], Loss: total=2.111, box=0.350, obj_pos=0.885, obj_neg=0.207, cls=0.319, num_pos=170.000, num_neg=681238.000
Learning Rate for this epoch: 3.608e-05
Validation Loss: 1009.0611

Starting epoch 45 / 50
"""

grad_norms = []
prefix = "Epoch ["

# 逐行處理字串
for line in long_string.splitlines():
    # 檢查是否以 "Grad norm: " 開頭
    if line.startswith(prefix):
        boolA = line[20:23] == "120"
        boolB = line[21:24] == "120"
        if not boolA  and not boolB: continue
        try:
            # 移除前綴並轉換為浮點數
            value_str = line[42:47] if boolA else line[43:48]
            value = float(value_str)
            grad_norms.append(value)
        except ValueError:
            # 如果轉換失敗 (例如: "Grad norm: NaN")，則跳過
            print(f"Skipping line, cannot convert to float: {line}")

# 輸出擷取到的 list
# print("擷取到的值:")
# print(grad_norms)

# 使用 matplotlib.pyplot 繪圖
if grad_norms:
    plt.plot(grad_norms) # 'o' 標記每個點
    plt.title("Grad Norm values over time")
    plt.xlabel("Index (Sample Order)")
    plt.ylabel("Grad Norm Value")
    plt.grid(True) # 加入網格
    plt.show()
# else:
#     print("沒有找到任何 'Grad norm: ' 資料，無法繪圖。")

"""
val loss for each epoch:
[7823.2653, 6336.7121, 4811.8909, 3485.2035, 2437.7064, 1633.4174, 1062.5284, 676.3524, 425.1751, 268.0803, 169.7027, 107.7064, 69.2038, 45.3201, 30.4563, 21.0545, 15.0359, 11.1035, 8.4981, 6.7401, 5.5318, 4.6817, 6.028, 33.4708, 71.0217, 104.671, 138.2057, 195.33, 265.5912, 351.0126, 464.2208, 591.0277, 680.06, 740.039, 816.0749, 849.2801, 871.2166, 962.8017, 1142.7054, 1080.6939, 1165.4194, 1120.6691, 1048.0227, 1009.0611]
train loss for each epoch:
[77.41, 23.39, 12.49, 8.022, 5.923, 4.826, 4.08, 3.638, 3.3, 3.072, 2.903, 2.774, 2.656, 2.573, 2.491, 2.479, 2.434, 2.379, 2.35, 2.261, 2.272, 2.213, 2.258, 2.229, 2.183, 2.221, 2.233, 2.161, 2.13, 2.182, 2.231, 2.161, 2.141, 2.172, 2.111, 2.15, 2.19, 2.099, 2.127, 2.156, 2.141, 2.085, 2.152, 2.111]
所以我覺得很明顯 ema model 在更新權重的時候有問題
"""