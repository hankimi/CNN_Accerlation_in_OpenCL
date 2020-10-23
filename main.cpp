#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)   // Maximum size for the kernel .cl source code size

#define CONV_W_SIZE 245
#define CONV_B_SIZE 3920
#define DENSE_W_SIZE 117600
#define DENSE_B_SIZE 120
#define DENSE_W2_SIZE 1200
#define DENSE_B2_SIZE 10

#define CONV_LAYER_SIZE 3920
#define SIG_LAYER_SIZE 3920
#define MAX_POOLING_SIZE 3920
#define MAX_LAYER_SIZE 980
#define DENSE_INPUT_SIZE 980
#define DENSE_SUM_SIZE 120
#define DENSE_SIGMOID_SIZE 120
#define DENSE_SUM2_SIZE 10
#define DENSE_SOFTMAX_SIZE 10
#define DW2_SIZE 1200
#define DB2_SIZE 10
#define DW1_SIZE 117600
#define DB1_SIZE 120
#define DW_MAX_SIZE 3920
#define DW_CONV_SIZE 245
#define DB_CONV_SIZE 3920

#define DELTA4_SIZE 10
#define DELTA3_SIZE 120
#define DELTA2_SIZE 980

const int EPOCH_NUM = 50;
const int batch_size = 200;
const int IMG_SIZE = 1024;
const int VECTOR_Y_SIZE = 10;
const int filter_size = 7;
const float eta = 0.01;

int data_train[60000][784];
int data_test[10000][784];
int label_train[60000];
int label_test[10000];

float conv_w[5][7][7];
float conv_b[5][28][28];
float conv_layer[5][28][28];
float sig_layer[5][28][28];
char max_pooling[5][28][28];
float max_layer[5][14][14];

float dense_input[980];
float dense_w[980][120];
float dense_b[120];
float dense_sum[120];
float dense_sigmoid[120];
float dense_w2[120][10];
float dense_b2[10];
float dense_sum2[10];
float dense_softmax[10];

float dw2[120][10];
float db2[10];
float dw1[980][120];
float db1[120];

float dw_max[5][28][28];
float dw_conv[5][7][7];
float db_conv[5][28][28];

/* ************************************************************ */
/* Helper functions */
float sigmoid(float x) {
    if (x > 500) x = 500;
    if (x < -500) x = -500;
    return 1 / (1 + exp(-x));
}
float d_sigmoid(float x) {
    float sig = sigmoid(x);
    return sig * (1 - sig);
}
float softmax_den(float *x, int len) {
    float val = 0;
    for (int i = 0; i < len; i++) {
        val += exp(x[i]);
    }
    return val;
}

/* ************************************************************ */
/* Forward Pass */
void forward_pass(int img[][32]) {
    // Convolution Operation + Sigmoid Activation
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                max_pooling[filter_dim][i][j] = 0;

                conv_layer[filter_dim][i][j] = 0;
                sig_layer[filter_dim][i][j] = 0;
                for (int k = 0; k < filter_size; k++) {
                    for (int l = 0; l < filter_size; l++) {
                        conv_layer[filter_dim][i][j] = img[-2 + i + k][-2 + j + l] * conv_w[filter_dim][k][l];
                    }
                }
                sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
            }
        }
    }

    // MAX Pooling (max_pooling, max_layer)
    float cur_max = 0;
    int max_i = 0, max_j = 0;
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
        for (int i = 0; i < 28; i += 2) {
            for (int j = 0; j < 28; j += 2) {
                max_i = i;
                max_j = j;
                cur_max = sig_layer[filter_dim][i][j];
                for (int k = 0; k < 2; k++) {
                    for (int l = 0; l < 2; l++) {
                        if (sig_layer[filter_dim][i + k][j + l] > cur_max) {
                            max_i = i + k;
                            max_j = j + l;
                            cur_max = sig_layer[filter_dim][max_i][max_j];
                        }
                    }
                }
                max_pooling[filter_dim][max_i][max_j] = 1;
                max_layer[filter_dim][i / 2][j / 2] = cur_max;
            }
        }
    }

    int k = 0;
    for (int filter_dim = 0; filter_dim < 5; filter_dim++) {
        for (int i = 0; i < 14; i++) {
            for (int j = 0; j < 14; j++) {
                dense_input[k] = max_layer[filter_dim][i][j];
                k++;
            }
        }
    }

    // Dense Layer
    for (int i = 0; i < 120; i++) {
        dense_sum[i] = 0;
        dense_sigmoid[i] = 0;
        for (int j = 0; j < 980; j++) {
            dense_sum[i] += dense_w[j][i] * dense_input[j];
        }
        dense_sum[i] += dense_b[i];
        dense_sigmoid[i] = sigmoid(dense_sum[i]);
    }

    // Dense Layer 2
    for (int i = 0; i < 10; i++) {
        dense_sum2[i] = 0;
        for (int j = 0; j < 120; j++) {
            dense_sum2[i] += dense_w2[j][i] * dense_sigmoid[j];
        }
        dense_sum2[i] += dense_b2[i];
    }

    // Softmax Output
    float den = softmax_den(dense_sum2, 10);
    for (int i = 0; i < 10; i++) {
        dense_softmax[i] = exp(dense_sum2[i]) / den;
    }
}

/* ************************************************************ */

void read_train_data() {
    ifstream csvread;
    csvread.open("/cad2/ece1718s/mnist_train.csv", ios::in);
    if (csvread) {
        string s;
        int data_pt = 0;
        while (getline(csvread, s)) {
            stringstream ss(s);
            int pxl = 0;
            while (ss.good()) {
                string substr;
                getline(ss, substr, ',');
                if (pxl == 0) {
                    label_train[data_pt] = stoi(substr);
                } else {
                    data_train[data_pt][pxl - 1] = stoi(substr);
                }
                pxl++;
            }
            data_pt++;
        }
        csvread.close();
    } else {
        cerr << "Unable to read train data!" << endl;
        exit(EXIT_FAILURE);
    }
}
void read_test_data() {
    ifstream csvread;
    csvread.open("/cad2/ece1718s/mnist_test.csv", ios::in);
    if (csvread) {
        string s;
        int data_pt = 0;
        while (getline(csvread, s)) {
            stringstream ss(s);
            int pxl = 0;
            while (ss.good()) {
                string substr;
                getline(ss, substr, ',');
                if (pxl == 0) {
                    label_test[data_pt] = stoi(substr);
                } else {
                    data_test[data_pt][pxl - 1] = stoi(substr);
                }
                pxl++;
            }
            data_pt++;
        }
        csvread.close();
    } else {
        cerr << "Unable to read test data!" << endl;
        exit(EXIT_FAILURE);
    }
}

void give_img(int *vec, int img[][32]) {
    int k = 0;
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            if (i < 2 || j < 2) {
                img[i][j] = 0;
            } else {
                img[i][j] = vec[k];
                k++;
            }
        }
    }
}

void give_y(int y, int *vector_y) {
    for (int i = 0; i < 10; i++) vector_y[i] = 0;
    vector_y[y] = 1;
}
int give_prediction() {
    float max_val = dense_softmax[0];
    int max_pos = 0;
    for (int i = 1; i < 10; i++) {
        if (dense_softmax[i] > max_val) {
            max_val = dense_softmax[i];
            max_pos = i;
        }
    }

    return max_pos;
}

int main(void) {
    read_test_data();
    read_train_data();
    //initialise_weights();

    int epoch = EPOCH_NUM;
    int num = 0;

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("training_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    printf("kernel code loaded from file!\n");

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1,
                         &device_id, &ret_num_devices);
    if (ret == CL_SUCCESS) {
        cout << "GetDeviceID success!\n";
    } else {
        cout << "GetDeviceID failed with ret = " << ret << "\n";
    }

    // Create an OpenCL context
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    if (ret == CL_SUCCESS) {
        cout << "CreateContext success!\n";
    } else {
        cout << "CreateContext failed with ret = " << ret << "\n";
    }

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    if (ret == CL_SUCCESS) {
        cout << "CreateCommandQueue success!\n";
    } else {
        cout << "CreateCommandQueue failed with ret = " << ret << "\n";
    }

    // Create memory buffers on the device for each vector
    cl_mem d_img = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                  IMG_SIZE * sizeof(int), NULL, &ret);
    cl_mem d_vector_y = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       VECTOR_Y_SIZE * sizeof(int), NULL, &ret);
    cl_mem d_conv_w = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     CONV_W_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_conv_b = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     CONV_B_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_w = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      DENSE_W_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_b = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      DENSE_B_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_w2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       DENSE_W2_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_b2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                       DENSE_B2_SIZE * sizeof(float), NULL, &ret);

    cl_mem d_conv_layer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         CONV_LAYER_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_sig_layer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        SIG_LAYER_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_max_pooling = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          MAX_POOLING_SIZE * sizeof(char), NULL, &ret);
    cl_mem d_max_layer = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        MAX_LAYER_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_input = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          DENSE_INPUT_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_sum = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        DENSE_SUM_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_sigmoid = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            DENSE_SIGMOID_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_sum2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                         DENSE_SUM2_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dense_softmax = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                            DENSE_SOFTMAX_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dw2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  DW2_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_db2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  DB2_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dw1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  DW1_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_db1 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                  DB1_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dw_max = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     DW_MAX_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_dw_conv = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      DW_CONV_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_db_conv = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      DB_CONV_SIZE * sizeof(float), NULL, &ret);

    cl_mem d_delta4 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     DELTA4_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_delta3 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     DELTA3_SIZE * sizeof(float), NULL, &ret);
    cl_mem d_delta2 = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                     DELTA2_SIZE * sizeof(float), NULL, &ret);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
    if (ret == CL_SUCCESS) {
        cout << "Program created successfully!\n";
    } else {
        cout << "Program creation failed with ret = " << ret << "\n";
    }

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret == CL_SUCCESS) {
        cout << "Program built successfully!\n";
    } else {
        cout << "Program build failed!\n";
        if (ret == CL_BUILD_PROGRAM_FAILURE) {
            // Determine the size of the log
            size_t log_size;
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            // Allocate memory for the log
            char *log = (char *)malloc(log_size);
            // Get the log
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
            // Print the log
            printf("%s\n", log);
        }
    }
    // Create the initialize_weight kernel
    cl_kernel initialize_weight_kernel = clCreateKernel(program, "initialize_weight", &ret);
    if (ret == CL_SUCCESS) {
        cout << "Initialize weight kernel created successfully!\n";
    } else {
        cout << "Initialize weight kernel creation failed with ret = " << ret << endl;
    }
    // Set the arguments of the initialize_weight_kernel
    ret = clSetKernelArg(initialize_weight_kernel, 0, sizeof(cl_mem), (void *)&d_conv_w);
    ret = clSetKernelArg(initialize_weight_kernel, 1, sizeof(cl_mem), (void *)&d_conv_b);
    ret = clSetKernelArg(initialize_weight_kernel, 2, sizeof(cl_mem), (void *)&d_dense_w);
    ret = clSetKernelArg(initialize_weight_kernel, 3, sizeof(cl_mem), (void *)&d_dense_b);
    ret = clSetKernelArg(initialize_weight_kernel, 4, sizeof(cl_mem), (void *)&d_dense_w2);
    ret = clSetKernelArg(initialize_weight_kernel, 5, sizeof(cl_mem), (void *)&d_dense_b2);

    // Create the forward_pass kernel
    cl_kernel forward_pass_kernel = clCreateKernel(program, "forward_pass", &ret);
    if (ret == CL_SUCCESS) {
        cout << "forward_pass kernel created successfully!\n";
    } else {
        cout << "forward_pass kernel creation failed with ret = " << ret << endl;
    }
    // Set the arguments of the forward_pass_kernel
    ret = clSetKernelArg(forward_pass_kernel, 0, sizeof(cl_mem), (void *)&d_img);
    ret = clSetKernelArg(forward_pass_kernel, 1, sizeof(cl_mem), (void *)&d_vector_y);
    ret = clSetKernelArg(forward_pass_kernel, 2, sizeof(cl_mem), (void *)&d_conv_w);
    ret = clSetKernelArg(forward_pass_kernel, 3, sizeof(cl_mem), (void *)&d_conv_b);
    ret = clSetKernelArg(forward_pass_kernel, 4, sizeof(cl_mem), (void *)&d_dense_w);
    ret = clSetKernelArg(forward_pass_kernel, 5, sizeof(cl_mem), (void *)&d_dense_b);
    ret = clSetKernelArg(forward_pass_kernel, 6, sizeof(cl_mem), (void *)&d_dense_w2);
    ret = clSetKernelArg(forward_pass_kernel, 7, sizeof(cl_mem), (void *)&d_dense_b2);

    ret = clSetKernelArg(forward_pass_kernel, 8, sizeof(cl_mem), (void *)&d_conv_layer);
    ret = clSetKernelArg(forward_pass_kernel, 9, sizeof(cl_mem), (void *)&d_sig_layer);
    ret = clSetKernelArg(forward_pass_kernel, 10, sizeof(cl_mem), (void *)&d_max_pooling);
    ret = clSetKernelArg(forward_pass_kernel, 11, sizeof(cl_mem), (void *)&d_max_layer);
    ret = clSetKernelArg(forward_pass_kernel, 12, sizeof(cl_mem), (void *)&d_dense_input);
    ret = clSetKernelArg(forward_pass_kernel, 13, sizeof(cl_mem), (void *)&d_dense_sum);
    ret = clSetKernelArg(forward_pass_kernel, 14, sizeof(cl_mem), (void *)&d_dense_sigmoid);
    ret = clSetKernelArg(forward_pass_kernel, 15, sizeof(cl_mem), (void *)&d_dense_sum2);
    ret = clSetKernelArg(forward_pass_kernel, 16, sizeof(cl_mem), (void *)&d_dense_softmax);
    ret = clSetKernelArg(forward_pass_kernel, 17, sizeof(cl_mem), (void *)&d_dw2);
    ret = clSetKernelArg(forward_pass_kernel, 18, sizeof(cl_mem), (void *)&d_db2);
    ret = clSetKernelArg(forward_pass_kernel, 19, sizeof(cl_mem), (void *)&d_dw1);
    ret = clSetKernelArg(forward_pass_kernel, 20, sizeof(cl_mem), (void *)&d_db1);
    ret = clSetKernelArg(forward_pass_kernel, 21, sizeof(cl_mem), (void *)&d_dw_max);
    ret = clSetKernelArg(forward_pass_kernel, 22, sizeof(cl_mem), (void *)&d_dw_conv);
    ret = clSetKernelArg(forward_pass_kernel, 23, sizeof(cl_mem), (void *)&d_db_conv);
    ret = clSetKernelArg(forward_pass_kernel, 24, sizeof(cl_mem), (void *)&d_delta4);
    ret = clSetKernelArg(forward_pass_kernel, 25, sizeof(cl_mem), (void *)&d_delta3);
    ret = clSetKernelArg(forward_pass_kernel, 26, sizeof(cl_mem), (void *)&d_delta2);

    // Create the backward_pass kernel
    cl_kernel backward_pass_kernel = clCreateKernel(program, "backward_pass", &ret);
    if (ret == CL_SUCCESS) {
        cout << "backward_pass kernel created successfully!\n";
    } else {
        cout << "backward_pass kernel creation failed with ret = " << ret << endl;
    }
    // Set the arguments of the backward_pass_kernel
    ret = clSetKernelArg(backward_pass_kernel, 0, sizeof(cl_mem), (void *)&d_img);
    ret = clSetKernelArg(backward_pass_kernel, 1, sizeof(cl_mem), (void *)&d_vector_y);
    ret = clSetKernelArg(backward_pass_kernel, 2, sizeof(cl_mem), (void *)&d_conv_w);
    ret = clSetKernelArg(backward_pass_kernel, 3, sizeof(cl_mem), (void *)&d_conv_b);
    ret = clSetKernelArg(backward_pass_kernel, 4, sizeof(cl_mem), (void *)&d_dense_w);
    ret = clSetKernelArg(backward_pass_kernel, 5, sizeof(cl_mem), (void *)&d_dense_b);
    ret = clSetKernelArg(backward_pass_kernel, 6, sizeof(cl_mem), (void *)&d_dense_w2);
    ret = clSetKernelArg(backward_pass_kernel, 7, sizeof(cl_mem), (void *)&d_dense_b2);

    ret = clSetKernelArg(backward_pass_kernel, 8, sizeof(cl_mem), (void *)&d_conv_layer);
    ret = clSetKernelArg(backward_pass_kernel, 9, sizeof(cl_mem), (void *)&d_sig_layer);
    ret = clSetKernelArg(backward_pass_kernel, 10, sizeof(cl_mem), (void *)&d_max_pooling);
    ret = clSetKernelArg(backward_pass_kernel, 11, sizeof(cl_mem), (void *)&d_max_layer);
    ret = clSetKernelArg(backward_pass_kernel, 12, sizeof(cl_mem), (void *)&d_dense_input);
    ret = clSetKernelArg(backward_pass_kernel, 13, sizeof(cl_mem), (void *)&d_dense_sum);
    ret = clSetKernelArg(backward_pass_kernel, 14, sizeof(cl_mem), (void *)&d_dense_sigmoid);
    ret = clSetKernelArg(backward_pass_kernel, 15, sizeof(cl_mem), (void *)&d_dense_sum2);
    ret = clSetKernelArg(backward_pass_kernel, 16, sizeof(cl_mem), (void *)&d_dense_softmax);
    ret = clSetKernelArg(backward_pass_kernel, 17, sizeof(cl_mem), (void *)&d_dw2);
    ret = clSetKernelArg(backward_pass_kernel, 18, sizeof(cl_mem), (void *)&d_db2);
    ret = clSetKernelArg(backward_pass_kernel, 19, sizeof(cl_mem), (void *)&d_dw1);
    ret = clSetKernelArg(backward_pass_kernel, 20, sizeof(cl_mem), (void *)&d_db1);
    ret = clSetKernelArg(backward_pass_kernel, 21, sizeof(cl_mem), (void *)&d_dw_max);
    ret = clSetKernelArg(backward_pass_kernel, 22, sizeof(cl_mem), (void *)&d_dw_conv);
    ret = clSetKernelArg(backward_pass_kernel, 23, sizeof(cl_mem), (void *)&d_db_conv);
    ret = clSetKernelArg(backward_pass_kernel, 24, sizeof(cl_mem), (void *)&d_delta4);
    ret = clSetKernelArg(backward_pass_kernel, 25, sizeof(cl_mem), (void *)&d_delta3);
    ret = clSetKernelArg(backward_pass_kernel, 26, sizeof(cl_mem), (void *)&d_delta2);

    // Create the update_weight kernel
    cl_kernel update_weight_kernel = clCreateKernel(program, "update_weight", &ret);
    if (ret == CL_SUCCESS) {
        cout << "update_weight kernel created successfully!\n";
    } else {
        cout << "update_weight kernel creation failed with ret = " << ret << endl;
    }
    // Set the arguments of the update_weight_kernel
    ret = clSetKernelArg(update_weight_kernel, 0, sizeof(cl_mem), (void *)&d_conv_w);
    ret = clSetKernelArg(update_weight_kernel, 1, sizeof(cl_mem), (void *)&d_conv_b);
    ret = clSetKernelArg(update_weight_kernel, 2, sizeof(cl_mem), (void *)&d_dense_w);
    ret = clSetKernelArg(update_weight_kernel, 3, sizeof(cl_mem), (void *)&d_dense_b);
    ret = clSetKernelArg(update_weight_kernel, 4, sizeof(cl_mem), (void *)&d_dense_w2);
    ret = clSetKernelArg(update_weight_kernel, 5, sizeof(cl_mem), (void *)&d_dense_b2);

    ret = clSetKernelArg(update_weight_kernel, 6, sizeof(cl_mem), (void *)&d_dw2);
    ret = clSetKernelArg(update_weight_kernel, 7, sizeof(cl_mem), (void *)&d_db2);
    ret = clSetKernelArg(update_weight_kernel, 8, sizeof(cl_mem), (void *)&d_dw1);
    ret = clSetKernelArg(update_weight_kernel, 9, sizeof(cl_mem), (void *)&d_db1);
    ret = clSetKernelArg(update_weight_kernel, 10, sizeof(cl_mem), (void *)&d_dw_conv);
    ret = clSetKernelArg(update_weight_kernel, 11, sizeof(cl_mem), (void *)&d_db_conv);

    size_t global_item_size, local_item_size;
    // Execute the initialize_weight kernel
    global_item_size = 16384;   // Process a total of 1024 work items
    local_item_size = 1024;     // Process in one work group with size of 1024
    ret = clEnqueueNDRangeKernel(command_queue, initialize_weight_kernel, 1, NULL,
                                 &global_item_size, &local_item_size, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        cout << "initialize_weight kernel execution failed with ret = " << ret << endl;
        if (ret == -5) {
            cout << "Out of resources!\n";
        }
    }

    // execute kernels to starts training
    cout << "Start Training...\n";
    for (int i = 0; i < epoch; i++) {
        for (int j = 0; j < batch_size; j++) {
            num = rand() % 60000;
            int img[32][32];
            int vector_y[10];
            give_y(label_train[num], vector_y);
            give_img(data_train[num], img);
            // Copy host memory to device memory
            ret = clEnqueueWriteBuffer(command_queue, d_img, CL_TRUE, 0,
                                       IMG_SIZE * sizeof(int), img, 0, NULL, NULL);
            ret = clEnqueueWriteBuffer(command_queue, d_vector_y, CL_TRUE, 0,
                                       VECTOR_Y_SIZE * sizeof(int), vector_y, 0, NULL, NULL);
            // Execute the forward_pass kernel
            global_item_size = 64;   // Process a total of 1024 work items
            local_item_size = 64;    // Process in one work group with size of 1024
            ret = clEnqueueNDRangeKernel(command_queue, forward_pass_kernel, 1, NULL,
                                         &global_item_size, &local_item_size, 0, NULL, NULL);
            if (ret != CL_SUCCESS) {
                cout << "forward_pass kernel execution failed with ret = " << ret << endl;
                if (ret == -5) {
                    cout << "Out of resources!\n";
                }
            }
            // Execute the backward_pass kernel
            global_item_size = 1024;   // Process a total of 1024 work items
            local_item_size = 1024;    // Process in one work group with size of 1024
            ret = clEnqueueNDRangeKernel(command_queue, backward_pass_kernel, 1, NULL,
                                         &global_item_size, &local_item_size, 0, NULL, NULL);
            if (ret != CL_SUCCESS) {
                cout << "backward_pass kernel execution failed with ret = " << ret << endl;
                if (ret == -5) {
                    cout << "Out of resources!\n";
                }
            }
            // Execute the update_weight kernel
            global_item_size = 16384;   // Process a total of 1024 work items
            local_item_size = 1024;     // Process in one work group with size of 1024
            ret = clEnqueueNDRangeKernel(command_queue, update_weight_kernel, 1, NULL,
                                         &global_item_size, &local_item_size, 0, NULL, NULL);
            if (ret != CL_SUCCESS) {
                cout << "update_weight kernel execution failed with ret = " << ret << endl;
                if (ret == -5) {
                    cout << "Out of resources!\n";
                }
            }
        }
        cout << "Epoch " << i << " done." << endl;
    }

    // Copy from device memory to host memory
    ret = clEnqueueReadBuffer(command_queue, d_conv_w, CL_TRUE, 0,
                              CONV_W_SIZE * sizeof(float), conv_w, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, d_conv_b, CL_TRUE, 0,
                              CONV_B_SIZE * sizeof(float), conv_b, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, d_dense_w, CL_TRUE, 0,
                              DENSE_W_SIZE * sizeof(float), dense_w, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, d_dense_b, CL_TRUE, 0,
                              DENSE_B_SIZE * sizeof(float), dense_b, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, d_dense_w2, CL_TRUE, 0,
                              DENSE_W2_SIZE * sizeof(float), dense_w2, 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, d_dense_b2, CL_TRUE, 0,
                              DENSE_B2_SIZE * sizeof(float), dense_b2, 0, NULL, NULL);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(initialize_weight_kernel);
    ret = clReleaseKernel(forward_pass_kernel);
    ret = clReleaseKernel(backward_pass_kernel);
    ret = clReleaseKernel(update_weight_kernel);
    ret = clReleaseProgram(program);

    ret = clReleaseMemObject(d_img);
    ret = clReleaseMemObject(d_vector_y);
    ret = clReleaseMemObject(d_conv_w);
    ret = clReleaseMemObject(d_conv_b);
    ret = clReleaseMemObject(d_dense_w);
    ret = clReleaseMemObject(d_dense_b);
    ret = clReleaseMemObject(d_dense_w2);
    ret = clReleaseMemObject(d_dense_b2);

    ret = clReleaseMemObject(d_conv_layer);
    ret = clReleaseMemObject(d_sig_layer);
    ret = clReleaseMemObject(d_max_pooling);
    ret = clReleaseMemObject(d_max_layer);
    ret = clReleaseMemObject(d_dense_input);
    ret = clReleaseMemObject(d_dense_sum);
    ret = clReleaseMemObject(d_dense_sigmoid);
    ret = clReleaseMemObject(d_dense_sum2);
    ret = clReleaseMemObject(d_dense_softmax);
    ret = clReleaseMemObject(d_dw2);
    ret = clReleaseMemObject(d_db2);
    ret = clReleaseMemObject(d_dw1);
    ret = clReleaseMemObject(d_db1);
    ret = clReleaseMemObject(d_dw_max);
    ret = clReleaseMemObject(d_dw_conv);
    ret = clReleaseMemObject(d_db_conv);
    ret = clReleaseMemObject(d_delta4);
    ret = clReleaseMemObject(d_delta3);
    ret = clReleaseMemObject(d_delta2);

    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // test
    int val_len = 600;
    int cor = 0;
    int confusion_mat[10][10];
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) confusion_mat[i][j] = 0;
    }

    cout << "Start Testing..." << endl;
    for (int i = 0; i < val_len; i++) {
        int img[32][32];
        give_img(data_test[i], img);
        forward_pass(img);
        int pre = give_prediction();
        confusion_mat[label_test[i]][pre]++;
        if (pre == label_test[i]) cor++;
    }
    float accu = float(cor) / val_len;
    cout << "Accuracy: " << accu << endl;

    cout << "   0 1 2 3 4 5 6 7 8 9" << endl;
    for (int i = 0; i < 10; i++) {
        cout << i << ": ";
        for (int j = 0; j < 10; j++) {
            cout << confusion_mat[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
