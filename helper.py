import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs  
    
def getKeyframes(path):
  # Cell 1

  cap = cv2.VideoCapture(path) 
  arr = np.empty((0, 1944), int)
  D = dict()
  count = 0
  start_time = time.time()
  while cap.isOpened():
      ret, frame = cap.read()
      if ret == True:
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          D[count] = frame_rgb   
          height, width, channels = frame_rgb.shape

          if height % 3 == 0:
              h_chunk = int(height/3)
          else:
              h_chunk = int(height/3) + 1

          if width % 3 == 0:
              w_chunk = int(width/3)
          else:
              w_chunk = int(width/3) + 1

          h = 0
          w = 0 
          feature_vector = []
          for a in range(1,4):
              h_window = h_chunk*a
              for b in range(1,4):
                  frame = frame_rgb[h : h_window, w : w_chunk*b , :]
                  hist = cv2.calcHist(frame, [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256])#finding histograms for each block  
                  hist1= hist.flatten()  #flatten the hist to one-dimensinal vector 
                  feature_vector += list(hist1)
                  w = w_chunk*b
                  
              h = h_chunk*a
              w = 0
          arr = np.vstack((arr, feature_vector ))
          count+=1
      else:
          break
  final_arr = arr.transpose()

  # Cell 2
  A = csc_matrix(final_arr, dtype=float)
  u, s, vt = svds(A, k = 63)

  # Cell 3
  v1_t = vt.transpose()
  projections = v1_t @ np.diag(s)

  # Cell 4
  f=projections
  C = dict()
  for i in range(f.shape[0]):
      C[i] = np.empty((0,63), int)
  C[0] = np.vstack((C[0], f[0]))   
  C[0] = np.vstack((C[0], f[1]))

  E = dict()
  for i in range(projections.shape[0]):
      E[i] = np.empty((0,63), int)
      
  E[0] = np.mean(C[0], axis=0)

  count = 0
  for i in range(2,f.shape[0]):
      similarity = np.dot(f[i], E[count])/( (np.dot(f[i],f[i]) **.5) * (np.dot(E[count], E[count]) ** .5)) 
      if similarity < 0.9:
          count+=1         
          C[count] = np.vstack((C[count], f[i])) 
          E[count] = np.mean(C[count], axis=0)   
      else:
          C[count] = np.vstack((C[count], f[i])) 
          E[count] = np.mean(C[count], axis=0)

  # Cell 5
  b = []
  for i in range(f.shape[0]):
      b.append(C[i].shape[0])
  last = b.index(0) 
  b1=b[:last]

  # Cell 6
  res = [idx for idx, val in enumerate(b1) if val >= 25]

  # Cell 7
  GG = C
  for i in range(last):
      p1= np.repeat(i, b1[i]).reshape(b1[i],1)
      GG[i] = np.hstack((GG[i],p1))

  # Cell 8
  F =  np.empty((0,64), int) 
  for i in range(last):
      F = np.vstack((F,GG[i]))
    
  # Cell 9
  colnames = []
  for i in range(1, 65):
      col_name = "v" + str(i)
      colnames+= [col_name]
  df = pd.DataFrame(F, columns= colnames)

  # Cell 10
  df['v64']= df['v64'].astype(int)

  # Cell 11
  df1 =  df[df.v64.isin(res)]

  # Cell 12
  new = df1.groupby('v64').tail(1)['v64']

  # Cell 13
  new1 = new.index
  keyframes = []
  for c in new1:
    keyframes.append(D[c])
  # return D, new1
  keyframes = np.array(keyframes)
  return keyframes



