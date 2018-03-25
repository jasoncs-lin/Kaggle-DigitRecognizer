from __future__ import print_function
import csv

import numpy as np
import caffe

test_csv_path = './data/test.csv'
model_path = 'deploy.prototxt'

#----K5-----START
K5_pretrained1_path = 'Shapshot_backup/K5/lenet_iter_102000.caffemodel'  #Val=99.53
K5_pretrained2_path = 'Shapshot_backup/K5/lenet_iter_112000.caffemodel'  #Val=99.47
K5_pretrained3_path = 'Shapshot_backup/K5/lenet_iter_217000.caffemodel'  #Val=99.47
K5_pretrained4_path = 'Shapshot_backup/K5/lenet_iter_110000.caffemodel'
K5_pretrained5_path = 'Shapshot_backup/K5/lenet_iter_125000.caffemodel'
K5_pretrained6_path = 'Shapshot_backup/K5/lenet_iter_120000.caffemodel'
K5_pretrained7_path = 'Shapshot_backup/K5/lenet_iter_206000.caffemodel'
K5_pretrained8_path = 'Shapshot_backup/K5/lenet_iter_118000.caffemodel'
K5_pretrained9_path = 'Shapshot_backup/K5/lenet_iter_201000.caffemodel'
#----K5-----END

#----K7-----START
K7_pretrained1_path = 'Shapshot_backup/K7/lenet_iter_55000.caffemodel'   #Val=99.39
K7_pretrained2_path = 'Shapshot_backup/K7/lenet_iter_121000.caffemodel'  #Val=99.36
K7_pretrained3_path = 'Shapshot_backup/K7/lenet_iter_151000.caffemodel'  #Val=99.35
#----K7-----END

#----K10-----START
K10_pretrained1_path = 'snapshot_Backup/K10/lenet_iter_282000.caffemodel'  #Val=99.61 LB=99.514
K10_pretrained2_path = 'snapshot_Backup/K10/lenet_iter_240000.caffemodel'  #Val=99.64
K10_pretrained3_path = 'snapshot_Backup/K10/lenet_iter_232000.caffemodel'  #Val=99.62
K10_pretrained4_path = 'snapshot_Backup/K10/lenet_iter_248000.caffemodel'  #Val=99.60
K10_pretrained5_path = 'snapshot_Backup/K10/lenet_iter_122000.caffemodel'  #Val=99.68 
#----K10-----END


#----K15-----START
K15_pretrained1_path = 'snapshot/lenet_iter_107000.caffemodel'   #Val=99.53
#K15_pretrained2_path = 'Shapshot_backup/K15/lenet_iter_214000.caffemodel'  #Val=99.55
#K15_pretrained3_path = 'Shapshot_backup/K15/lenet_iter_183000.caffemodel'  #Val=99.54
#K15_pretrained4_path = 'Shapshot_backup/K15/lenet_iter_106000.caffemodel'  #Val=99.53
#K15_pretrained5_path = 'Shapshot_backup/K15/lenet_iter_102000.caffemodel'  #Val=99.52
#K15_pretrained6_path = 'Shapshot_backup/K15/lenet_iter_104000.caffemodel'  #Val=99.51
#K15_pretrained7_path = 'Shapshot_backup/K15/lenet_iter_186000.caffemodel'  #Val=99.51
#----K15-----END

#----K20-----START
K20_pretrained1_path = 'snapshot_Backup/K20/lenet_iter_204000.caffemodel'  #Val=99.61
K20_pretrained2_path = 'snapshot_Backup/K20/lenet_iter_219000.caffemodel'  #Val=99.57
K20_pretrained3_path = 'snapshot_Backup/K20/lenet_iter_206000.caffemodel'  #Val=99.57
K20_pretrained4_path = 'snapshot_Backup/K20/lenet_iter_211000.caffemodel'  #Val=99.56
K20_pretrained5_path = 'snapshot_Backup/K20/lenet_iter_63000.caffemodel'
#----K20-----END

#----K25-----START
K25_pretrained1_path = 'snapshot_Backup/K25/lenet_iter_123000.caffemodel'  #Val=99.58
K25_pretrained2_path = 'snapshot_Backup/K25/lenet_iter_195000.caffemodel'  #Val=99.47
K25_pretrained3_path = 'snapshot_Backup/K25/lenet_iter_243000.caffemodel'  #Val=99.47
K25_pretrained4_path = 'snapshot_Backup/K25/lenet_iter_186000.caffemodel'
#----K25-----END

caffe.set_mode_gpu()
caffe.set_device(0)


clf1 = caffe.Classifier(model_path, K10_pretrained1_path, image_dims=(28, 28))
clf2 = caffe.Classifier(model_path, K10_pretrained2_path, image_dims=(28, 28))
clf3 = caffe.Classifier(model_path, K10_pretrained3_path, image_dims=(28, 28))
clf4 = caffe.Classifier(model_path, K10_pretrained4_path, image_dims=(28, 28))
clf5 = caffe.Classifier(model_path, K10_pretrained5_path, image_dims=(28, 28))
clf6 = caffe.Classifier(model_path, K20_pretrained1_path, image_dims=(28, 28))
clf7 = caffe.Classifier(model_path, K20_pretrained2_path, image_dims=(28, 28))
clf8 = caffe.Classifier(model_path, K20_pretrained3_path, image_dims=(28, 28))
clf9 = caffe.Classifier(model_path, K20_pretrained4_path, image_dims=(28, 28))

print('ImageId,Label')
with open(test_csv_path) as f:
    reader = csv.reader(f)
    next(reader)
    X = np.array([np.reshape([float(v) / 255 for v in row], (28, 28, 1)) for row in reader])
    y1=clf1.predict(X, oversample=False)
    y2=clf2.predict(X, oversample=False)
    y3=clf3.predict(X, oversample=False)
    y4=clf4.predict(X, oversample=False)
    y5=clf5.predict(X, oversample=False)
    y6=clf6.predict(X, oversample=False)
    y7=clf7.predict(X, oversample=False)
    y8=clf8.predict(X, oversample=False)
    y9=clf9.predict(X, oversample=False)
    
    y=y1+y2+y3+y4+y5+y6+y7+y8+y9
    for i, y_en in enumerate(y):
        print(i+1, np.argmax(y_en), sep=',')
        
