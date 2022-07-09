

#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <queue>

#include<string>
#include<vector>

#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>


#define BLOCK_LEN		(512)
#define FFT_OUT_SIZE    (BLOCK_LEN / 2 + 1)
#define STATE_SIZE      (512)

#define DTLN_MODEL_A "dtln_aec_1.tflite"
#define DTLN_MODEL_B "dtln_aec_2.tflite"

#define SAMEPLERATE  (16000)

#define BLOCK_SHIFT  (128)

struct trg_engine {
    float mic_buffer[BLOCK_LEN] = { 0 };
    float out_buffer[BLOCK_LEN] = { 0 };
    float states_a[STATE_SIZE] = { 0 };
    float states_b[STATE_SIZE] = { 0 };
    float lpb_buffer[BLOCK_LEN]= {0};

    TfLiteTensor* input_details_a[3], * input_details_b[3];
    const TfLiteTensor* output_details_a[2], * output_details_b[2];
    TfLiteInterpreter* interpreter_a, * interpreter_b;
    TfLiteModel* model_a, * model_b;
};





#endif 



