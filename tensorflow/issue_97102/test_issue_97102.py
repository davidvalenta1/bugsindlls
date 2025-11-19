import tensorflow as tf
import numpy as np
import pytest


def test_matrix_rank_cpu_gpu_bug_reproduced():
   """
   Reproduces the TensorFlow matrix_rank CPU vs GPU bug.
   The test passes if outputs differ (bug is present).
   """
   a = tf.ones((48, 74), dtype=tf.float64) * -88917319269045.
   tol = 6.


   with tf.device('/CPU:0'):
       output_cpu = tf.linalg.matrix_rank(a, tol=tol).numpy()


   with tf.device('/GPU:0'):
       output_gpu = tf.linalg.matrix_rank(a, tol=tol).numpy()


   output_np = np.linalg.matrix_rank(a.numpy(), tol=tol)


   print("CPU output:", output_cpu)
   print("GPU output:", output_gpu)
   print("NumPy output:", output_np)


   # Test passes if there is a mismatch
   bug_reproduced = (output_cpu != output_gpu) or (output_cpu != output_np) or (output_gpu != output_np)
   assert bug_reproduced, "Bug not reproduced: CPU, GPU, and NumPy outputs match unexpectedly."
