import os
import shutil
import random







# nondecimadatasetpath = '/media/mayur/Projects/Cobra/MayurRD/KerasDemo/BillAddedData/training-images/9/'
# pathtoCopy = 'Dataset/training-images/nondecimal/'
#
# print('before no decimal dataset length',len(os.listdir(pathtoCopy)))
#
# imagelist = os.listdir(nondecimadatasetpath)
#
#
# for i in range(100):
#     #make random choice
#     randomIndex = random.randint(1,len(os.listdir(nondecimadatasetpath)))
#
#     if(not os.path.exists(pathtoCopy+imagelist[randomIndex])):
#         shutil.copy(os.path.join(nondecimadatasetpath,imagelist[randomIndex]), pathtoCopy)
#     else :
#         i -=1 #decreasing counter
#
# print('after no decimal dataset length', len(os.listdir(pathtoCopy)))
#
#



















#
#
#
# path = 'Dataset/training-images'
#
#
# decimalpath, non_decimalpath = os.listdir(path)


#Decimal class
# decimalfiles = os.path.join(path, decimalpath)
#
# decimalImagesList = os.listdir(decimalfiles)
#
# decimalfilesTestingLimit = int(len(decimalImagesList) * 0.2)
# print('Testing DECIMAL ', decimalfilesTestingLimit)
#
# for i in range(decimalfilesTestingLimit):
#     #make random choice
#     randomIndex = random.randint(1,len(decimalImagesList))
#
#     if(not os.path.exists('Dataset/testing-images/'+decimalpath+'/'+decimalImagesList[randomIndex])):
#         shutil.move(os.path.join(decimalfiles,decimalImagesList[randomIndex]), 'Dataset/testing-images/'+decimalpath)
#     else :
#         i -=1 #decreasing counter


#
# #Non decimal class
#
# nondecimalfiles = os.path.join(path, non_decimalpath)
#
# nondecimalList = os.listdir(nondecimalfiles)
#
# nondecimalLimit = int(len(nondecimalList) * 0.2)
# print('Testing NON-DECIMAL ', nondecimalLimit)
#
# for j in range(nondecimalLimit):
#     nonRandomIndex = random.randint(1, len(nondecimalList))
#
#     if( not os.path.exists('Dataset/testing-images/'+non_decimalpath+'/'+nondecimalList[nonRandomIndex])):
#         shutil.move(os.path.join(nondecimalfiles,nondecimalList[nonRandomIndex]), 'Dataset/testing-images/'+non_decimalpath)
#     else:
#         j -=1
#
#
# print('Done !!')