// Define constants
#define EPOCH 50
#define RANDOM_SEED 1234567890
#define FILTER_SIZE 7
#define eta 0.01f
#define BATCH_SIZE 200
#define CONV_W_SIZE 245
#define CONV_B_SIZE 3920
#define DENSE_W_SIZE 117600
#define DENSE_B_SIZE 120
#define DENSE_W2_SIZE 1200
#define DENSE_B2_SIZE 10
//#define g_size 1024

/* Helper functions */
float sigmoid(float x) {
  if (x > 500)
    x = 500;
  if (x < -500)
    x = -500;
  return 1 / (1 + exp(-x));
}
float d_sigmoid(float x) {
  float sig = sigmoid(x);
  return sig * (1 - sig);
}

// function to generate a random float between -1 and 1
float kernel_RNG(unsigned long seed) {
  seed = (seed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
  unsigned result = seed >> 16;
  float res = 2 * (float)(result % 0x7fff) / 0x7fff - 1;
  return res;
}

__kernel void
initialize_weight(__global float *d_conv_w, __global float *d_conv_b,
                  __global float *d_dense_w, __global float *d_dense_b,
                  __global float *d_dense_w2, __global float *d_dense_b2) {

  int g_size = get_global_size(0);
  // Get the index of the current element
  // 0 ~ 1023, 1024 work items in totoal in one work group)
  int id = get_global_id(0);
  // printf("id: %d\n", id);

  // 1) initialize weights
  int random_idx = RANDOM_SEED + id;
  // conv_w
  if (id < CONV_W_SIZE)
    d_conv_w[id] = kernel_RNG(random_idx);
  random_idx += CONV_W_SIZE;
  // conv_b
  if (id < CONV_B_SIZE)
    d_conv_b[id] = kernel_RNG(random_idx);
  random_idx += CONV_B_SIZE;
  // dense_w
  int i = 0;
  for (; i + g_size < DENSE_W_SIZE; i = i + g_size) {
    d_dense_w[id + i] = kernel_RNG(random_idx + i);
  }
  if (id + i < DENSE_W_SIZE) {
    d_dense_w[id + i] = kernel_RNG(random_idx + i);
  }
  random_idx += DENSE_W_SIZE;
  // dense_b
  if (id < DENSE_B_SIZE)
    d_dense_b[id] = kernel_RNG(random_idx);
  random_idx += DENSE_B_SIZE;
  // dense_w2
  if (id < DENSE_W2_SIZE)
    d_dense_w2[id] = kernel_RNG(random_idx);
  random_idx += DENSE_W2_SIZE;
  // dense_b2
  if (id < DENSE_B2_SIZE) {
    d_dense_b2[id] = kernel_RNG(random_idx);
  }
}

// the forward_pass kernel
__kernel void forward_pass(
    __global int *d_img, __global int *d_vector_y, __global float *d_conv_w,
    __global float *d_conv_b, __global float *d_dense_w,
    __global float *d_dense_b, __global float *d_dense_w2,
    __global float *d_dense_b2, __global float *d_conv_layer,
    __global float *d_sig_layer, __global char *d_max_pooling,
    __global float *d_max_layer, __global float *d_dense_input,
    __global float *d_dense_sum, __global float *d_dense_sigmoid,
    __global float *d_dense_sum2, __global float *d_dense_softmax,
    __global float *d_dw2, __global float *d_db2, __global float *d_dw1,
    __global float *d_db1, __global float *d_dw_max, __global float *d_dw_conv,
    __global float *d_db_conv, __global float *delta4, __global float *delta3,
    __global float *delta2) {

  int g_size = get_global_size(0);
  // Get the index of the current element
  // 0 ~ 1023, 1024 work items in totoal in one work group)
  int id = get_global_id(0);
  // printf("id: %d\n", id);

  // Do the operation in work group 0 and work item 0 only

  // 2) forward_pass(img);
  // Convolution Operation + Sigmoid Activation
  if (id < 28) {
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {

      for (int j = 0; j < 28; j++) {
        d_max_pooling[filter_dim * 784 + id * 28 + j] = 0;
        d_conv_layer[filter_dim * 784 + id * 28 + j] = 0;
        d_sig_layer[filter_dim * 784 + id * 28 + j] = 0;
        for (int k = 0; k < FILTER_SIZE; k++) {
          for (int l = 0; l < FILTER_SIZE; l++) {
            if ((id + k - 2) >= 0 && (j + l - 2) >= 0)
              d_conv_layer[filter_dim * 784 + id * 28 + j] =
                  d_img[(id + k - 2) * 32 + (j + l - 2)] *
                  d_conv_w[filter_dim * 49 + k * 7 + l];
            else
              d_conv_layer[filter_dim * 784 + id * 28 + j] = 0;
          }
        }
        d_sig_layer[filter_dim * 784 + id * 28 + j] =
            sigmoid(d_conv_layer[filter_dim * 784 + id * 28 + j] +
                    d_conv_b[filter_dim * 784 + id * 28 + j]);
      }
    }
  }
  // barrier(CLK_GLOBAL_MEM_FENCE);
  // MAX Pooling (max_pooling, max_layer)
  float cur_max = 0;
  int max_i = 0, max_j = 0;
  if (id < 28 && id % 2 == 0) {
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {

      for (int j = 0; j < 28; j += 2) {
        max_i = id;
        max_j = j;
        cur_max = d_sig_layer[filter_dim * 784 + id * 28 + j];
        for (int k = 0; k < 2; k++) {
          for (int l = 0; l < 2; l++) {
            if (d_sig_layer[filter_dim * 784 + (id + k) * 28 + (j + l)] >
                cur_max) {
              max_i = id + k;
              max_j = j + l;
              cur_max = d_sig_layer[filter_dim * 784 + max_i * 28 + max_j];
            }
          }
        }
        d_max_pooling[filter_dim * 784 + max_i * 28 + max_j] = 1;
        d_max_layer[filter_dim * 196 + (id / 2) * 14 + (j / 2)] = cur_max;
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  d_dense_input[id] = d_max_layer[id];
  d_dense_input[id + 64] = d_max_layer[id + 64];
  d_dense_input[id + 64 * 2] = d_max_layer[id + 64 * 2];
  d_dense_input[id + 64 * 3] = d_max_layer[id + 64 * 3];
  d_dense_input[id + 64 * 4] = d_max_layer[id + 64 * 4];
  d_dense_input[id + 64 * 5] = d_max_layer[id + 64 * 5];
  d_dense_input[id + 64 * 6] = d_max_layer[id + 64 * 6];
  d_dense_input[id + 64 * 7] = d_max_layer[id + 64 * 7];
  d_dense_input[id + 64 * 8] = d_max_layer[id + 64 * 8];
  d_dense_input[id + 64 * 9] = d_max_layer[id + 64 * 9];
  d_dense_input[id + 64 * 10] = d_max_layer[id + 64 * 10];
  d_dense_input[id + 64 * 11] = d_max_layer[id + 64 * 11];
  d_dense_input[id + 64 * 12] = d_max_layer[id + 64 * 12];
  d_dense_input[id + 64 * 13] = d_max_layer[id + 64 * 13];
  d_dense_input[id + 64 * 14] = d_max_layer[id + 64 * 14];
  if ((id + 64 * 15) < 980) {
    d_dense_input[id + 64 * 15] = d_max_layer[id + 64 * 15];
  }
  // Dense Layer
  d_dense_sum[id] = 0;
  d_dense_sigmoid[id] = 0;
  if ((id + 64) < 120) {
    d_dense_sum[id + 64] = 0;
    d_dense_sigmoid[id + 64] = 0;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  for (int j = 0; j < 980; j++) {
    d_dense_sum[id] += d_dense_w[j * 120 + id] * d_dense_input[j];
  }
  d_dense_sum[id] += d_dense_b[id];
  d_dense_sigmoid[id] = sigmoid(d_dense_sum[id]);
  if ((id + 64) < 120) {
    for (int j = 0; j < 980; j++) {
      d_dense_sum[id + 64] += d_dense_w[j * 120 + id + 64] * d_dense_input[j];
    }
    d_dense_sum[id + 64] += d_dense_b[id + 64];
    d_dense_sigmoid[id + 64] = sigmoid(d_dense_sum[id + 64]);
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  // Dense Layer 2

  if (id < 10) {
    d_dense_sum2[id] = 0;
    for (int j = 0; j < 120; j++) {
      d_dense_sum2[id] += d_dense_w2[j * 10 + id] * d_dense_sigmoid[j];
    }
    d_dense_sum2[id] += d_dense_b2[id];
  }

  // Softmax Output (remove softmax_den function)
  float den = 0;
  for (int i = 0; i < 10; i++) {
    den += exp(d_dense_sum2[i]);
  }
  if (id < 10) {
    d_dense_softmax[id] = exp(d_dense_sum2[id]) / den;
  }
}
// end of function

// the backward_pass kernel
__kernel void backward_pass(
    __global int *d_img, __global int *d_vector_y, __global float *d_conv_w,
    __global float *d_conv_b, __global float *d_dense_w,
    __global float *d_dense_b, __global float *d_dense_w2,
    __global float *d_dense_b2, __global float *d_conv_layer,
    __global float *d_sig_layer, __global char *d_max_pooling,
    __global float *d_max_layer, __global float *d_dense_input,
    __global float *d_dense_sum, __global float *d_dense_sigmoid,
    __global float *d_dense_sum2, __global float *d_dense_softmax,
    __global float *d_dw2, __global float *d_db2, __global float *d_dw1,
    __global float *d_db1, __global float *d_dw_max, __global float *d_dw_conv,
    __global float *d_db_conv, __global float *delta4, __global float *delta3,
    __global float *delta2) {

  int g_size = get_global_size(0);
  // Get the index of the current element
  // 0 ~ 1023, 1024 work items in totoal in one work group)
  int id = get_global_id(0);
  // printf("id: %d\n", id);

  // Do the operation in work group 0 and work item 0 only
  // 3) backward_pass(dense_softmax, vector_y, img);
  //................................(1)...................................... No
  // changes in speed
  if (id == 0) {
    for (int i = 0; i < 10; i++) {
      delta4[i] = d_dense_softmax[i] -
                  d_vector_y[i]; // Derivative of Softmax + Cross entropy
      d_db2[i] = delta4[i];      // Bias Changes
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  // if(id < 10){
  //   delta4[id] = d_dense_softmax[id] - d_vector_y[id]; // Derivative of
  //   Softmax + Cross entropy
  // }
  // barrier(CLK_GLOBAL_MEM_FENCE);
  // if(id < 10){
  //   d_db2[id] = delta4[id];      // Bias Changes
  // }

  // Calculate Weight Changes for Dense Layer 2
  //....................................(2)...........................................No
  // changes in speed
  if (id == 0) {
    for (int i = 0; i < 120; i++) {
      for (int j = 0; j < 10; j++) {
        d_dw2[i * 10 + j] = d_dense_sigmoid[i] * delta4[j];
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  // if(id < 120){
  //   for (int j = 0; j < 10; j++) {
  //     d_dw2[id * 10 + j] = d_dense_sigmoid[id] * delta4[j];
  //   }
  // }
  // barrier(CLK_GLOBAL_MEM_FENCE);

  //....................................(3)...........................................Accuracy
  // changes no time speed up
  // Delta 3

  // if(id < 120){
  //   delta3[id] = 0;
  //   for (int j = 0; j < 10; j++){
  //     delta3[id] += d_dense_w2[id * 10 + j] * delta4[j];
  //   }
  //   delta3[id] *= d_sigmoid(d_dense_sum[id]);
  // }
  // barrier(CLK_GLOBAL_MEM_FENCE);

  // if(id < 120){
  //   d_db1[id] = delta3[id]; // Bias Weight change
  // }
  // barrier(CLK_GLOBAL_MEM_FENCE);

  if (id == 0) {
    for (int i = 0; i < 120; i++) {
      delta3[i] = 0;
      for (int j = 0; j < 10; j++) {
        delta3[i] += d_dense_w2[i * 10 + j] * delta4[j];
      }
      delta3[i] *= d_sigmoid(d_dense_sum[i]);
    }
    for (int i = 0; i < 120; i++)
      d_db1[i] = delta3[i]; // Bias Weight change
  }

  barrier(CLK_GLOBAL_MEM_FENCE);

  //....................................(4)...........................................
  // Accuracy changes
  // Calculate Weight Changes for Dense Layer 1
  // if (id == 0){
  //   for (int i = 0; i < 980; i++) {
  //     for (int j = 0; j < 120; j++) {
  //       d_dw1[i * 120 + j] = d_dense_input[i] * delta3[j];
  //     }
  //   }
  // }

  if (id < 980) {
    for (int j = 0; j < 120; j++) {
      d_dw1[id * 120 + j] = d_dense_input[id] * delta3[j];
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  //....................................(5)...........................................
  // ok 23.33 --> 19.14
  // Delta2
  // if (id == 0) {
  //   for (int i = 0; i < 980; i++) {
  //     delta2[i] = 0;
  //     for (int j = 0; j < 120; j++) {
  //       delta2[i] += d_dense_w[i * 120 + j] * delta3[j];
  //     }
  //     delta2[i] *= d_sigmoid(d_dense_input[i]);
  //   }
  // }

  if (id < 980) {
    delta2[id] = 0;
    for (int j = 0; j < 120; j++) {
      delta2[id] += d_dense_w[id * 120 + j] * delta3[j];
    }
    delta2[id] *= d_sigmoid(d_dense_input[id]);
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  //....................................(6)...........................................
  // ok 23.33
  // Calc back-propagated max layer dw_max
  if (id == 0) {
    int k = 0;
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
      for (int i = 0; i < 28; i += 2) {
        for (int j = 0; j < 28; j += 2) {
          for (int l = 0; l < 2; l++) {
            for (int m = 0; m < 2; m++) {
              if (d_max_pooling[filter_dim * 784 + (i + l) * 28 + (j + m)] == 1)
                d_dw_max[filter_dim * 784 + i * 28 + j] = delta2[k];
              else
                d_dw_max[filter_dim * 784 + i * 28 + j] = 0;
            }
          }
          k++;
        }
      }
    }
  }

  /* if (id < 28) {
    int k = 0;
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
      for (int j = 0; j < 28; j += 2) {
        k = (784 * filter_dim + id * 28 + j);
        for (int l = 0; l < 2; l++) {
          for (int m = 0; m < 2; m++) {
            if (d_max_pooling[filter_dim * 784 + (id + l) * 28 + (j + m)] == 1)
              d_dw_max[filter_dim * 784 + id * 28 + j] = delta2[k];
            else
              d_dw_max[filter_dim * 784 + id * 28 + j] = 0;
          }
        }
      }
    }
  } */
  barrier(CLK_GLOBAL_MEM_FENCE);

  //....................................(7)...........................................
  // ok
  // Calc Conv Bias Changes
  // if (id == 0) {
  //   for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
  //     for (int i = 0; i < 28; i++) {
  //       for (int j = 0; j < 28; j++) {
  //         d_db_conv[filter_dim * 784 + i * 28 + j] =
  //             d_dw_max[filter_dim * 784 + i * 28 + j];
  //       }
  //     }
  //   }
  // }

  d_db_conv[id] = d_dw_max[id];
  d_db_conv[id + g_size] = d_dw_max[id + g_size];
  d_db_conv[id + g_size * 2] = d_dw_max[id + g_size * 2];
  if (id + g_size * 3 < DENSE_W_SIZE) {
    d_db_conv[id + g_size * 3] = d_dw_max[id + g_size * 3];
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  //....................................(8)...........................................
  // ok with no changes
  // Set Conv Layer Weight changes to 0
  // if (id == 0) {
  //   for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
  //     for (int i = 0; i < 5; i++) {
  //       for (int j = 0; j < 5; j++) {
  //         d_dw_conv[filter_dim * 49 + i * 7 + j] = 0;
  //       }
  //     }
  //   }
  // }

  if (id < 5 * 5 * 5) {
    d_dw_conv[id] = 0;
  }
  barrier(CLK_GLOBAL_MEM_FENCE);

  //....................................(9)...........................................
  // 27.26 --> 23.64
  // Calculate Weight Changes for Conv Layer
  // if (id == 0) {
  //   for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
  //     for (int i = 0; i < 28; i++) {
  //       for (int j = 0; j < 28; j++) {
  //         float cur_val = d_dw_max[filter_dim * 784 + i * 28 + j];
  //         for (int k = 0; k < 5; k++) {
  //           for (int l = 0; l < 5; l++) {
  //             d_dw_conv[filter_dim * 49 + k * 7 + l] +=
  //                 ((i + k - 2) >= 0 && (j + l - 2) >= 0) ? (d_img[(i + k - 2)
  //                 * 32 + (j + l - 2)] * cur_val) : 0;
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  if (id < 28) {
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
      for (int j = 0; j < 28; j++) {
        float cur_val = d_dw_max[filter_dim * 784 + id * 28 + j];
        for (int k = 0; k < 5; k++) {
          for (int l = 0; l < 5; l++) {
            d_dw_conv[filter_dim * 49 + k * 7 + l] +=
                ((id + k - 2) >= 0 && (j + l - 2) >= 0)
                    ? (d_img[(id + k - 2) * 32 + (j + l - 2)] * cur_val)
                    : 0;
          }
        }
      }
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}

// the update_weight kernel
__kernel void update_weight(__global float *d_conv_w, __global float *d_conv_b,
                            __global float *d_dense_w,
                            __global float *d_dense_b,
                            __global float *d_dense_w2,
                            __global float *d_dense_b2, __global float *d_dw2,
                            __global float *d_db2, __global float *d_dw1,
                            __global float *d_db1, __global float *d_dw_conv,
                            __global float *d_db_conv) {

  int g_size = get_global_size(0);
  // Get the index of the current element
  // 0 ~ 1023, 1024 work items in totoal in one work group)
  int id = get_global_id(0);
  // printf("id: %d\n", id);

  // 4) update_weights(); 33.23s --> 28.66s
  float p_eta = eta;
  // dense_b
  if (id < DENSE_B_SIZE) {
    d_dense_b[id] -= p_eta * d_db1[id];
  }
  // dense_b2
  else if (id < DENSE_B2_SIZE + DENSE_B_SIZE) {
    d_dense_b2[id - DENSE_B_SIZE] -=
        DENSE_B_SIZE * p_eta * d_db2[id - DENSE_B_SIZE];
  }
  // dense_w2
  else if (id < DENSE_W2_SIZE + DENSE_B2_SIZE + DENSE_B_SIZE) {
    d_dense_w2[id - (DENSE_B2_SIZE + DENSE_B_SIZE)] -=
        p_eta * d_dw2[id - (DENSE_B2_SIZE + DENSE_B_SIZE)];
  }
  // conv_w
  else if (id < CONV_W_SIZE + DENSE_W2_SIZE + DENSE_B2_SIZE + DENSE_B_SIZE) {
    d_conv_w[id - (DENSE_W2_SIZE + DENSE_B2_SIZE + DENSE_B_SIZE)] -=
        p_eta * d_dw_conv[id - (DENSE_W2_SIZE + DENSE_B2_SIZE + DENSE_B_SIZE)];
  }
  // conv_b
  else if (id < CONV_B_SIZE + CONV_W_SIZE + DENSE_W2_SIZE + DENSE_B2_SIZE +
                    DENSE_B_SIZE) {
    d_conv_b[id -
             (CONV_W_SIZE + DENSE_W2_SIZE + DENSE_B2_SIZE + DENSE_B_SIZE)] -=
        p_eta * d_db_conv[id - (CONV_W_SIZE + DENSE_W2_SIZE + DENSE_B2_SIZE +
                                DENSE_B_SIZE)];
  }
  // dense_w
  int i = 0;
  for (; i + g_size < DENSE_W_SIZE; i = i + g_size) {
    d_dense_w[id + i] -= p_eta * d_dw1[id + i];
  }
  if (id + i < DENSE_W_SIZE) {
    d_dense_w[id + i] -= p_eta * d_dw1[id + i];
  }
}
