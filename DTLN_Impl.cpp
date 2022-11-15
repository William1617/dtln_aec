
 
#include "DTLN_defs.h"
#include "pocketfft_hdronly.h"
#include "AudioFile.h"

using namespace std;

typedef complex<double> cpx_type;


void ExportWAV(
        const std::string & Filename, 
		const std::vector<float>& Data, 
		unsigned SampleRate) {
    AudioFile<float>::AudioBuffer Buffer;
	Buffer.resize(1);

	Buffer[0] = Data;
	size_t BufSz = Data.size();

	AudioFile<float> File;
	File.setAudioBuffer(Buffer);
	File.setAudioBufferSize(1, (int)BufSz);
	File.setNumSamplesPerChannel((int)BufSz);
	File.setNumChannels(1);
	File.setBitDepth(16);
	File.setSampleRate(SAMEPLERATE);
	File.save(Filename, AudioFileFormat::Wave);		
}



void DTLNAEC() {


    trg_engine* m_pEngine;

    m_pEngine = new trg_engine;

	// load model
    m_pEngine->model_a = TfLiteModelCreateFromFile(DTLNModelNameA);
    m_pEngine->model_b = TfLiteModelCreateFromFile(DTLNModelNameB);

    // Build the interpreter
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 1);

    // Create the interpreter.
    m_pEngine->interpreter_a = TfLiteInterpreterCreate(m_pEngine->model_a, options);
    if (m_pEngine->interpreter_a == nullptr) {
        printf("Failed to create interpreter a\n");
        return ;
    }
	m_pEngine->interpreter_b = TfLiteInterpreterCreate(m_pEngine->model_b, options);
    if (m_pEngine->interpreter_b == nullptr) {
        printf("Failed to create interpreter b\n");
        return ;
    }

    // Allocate tensor buffers.
    if (TfLiteInterpreterAllocateTensors(m_pEngine->interpreter_a) != kTfLiteOk) {
        printf("Failed to allocate tensors a!\n");
        return;
    }
	if (TfLiteInterpreterAllocateTensors(m_pEngine->interpreter_b) != kTfLiteOk) {
        printf("Failed to allocate tensors b!\n");
        return ;
    }
    
    //input wav data of first model
    m_pEngine->input_details_a[0] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter_a, 0);
    //input state data of first model
    m_pEngine->input_details_a[1] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter_a, 1);
    //lpbinput
     m_pEngine->input_details_a[2] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter_a, 2);
    //output wav data of first model
    m_pEngine->output_details_a[0] = TfLiteInterpreterGetOutputTensor(m_pEngine->interpreter_a, 0);
    //output state data of first model
    m_pEngine->output_details_a[1] = TfLiteInterpreterGetOutputTensor(m_pEngine->interpreter_a, 1);

    //input wav data of second model
    m_pEngine->input_details_b[0] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter_b, 0);
    //input state data of second model
    m_pEngine->input_details_b[1] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter_b, 1);
    //lpb input
    m_pEngine->input_details_b[2] = TfLiteInterpreterGetInputTensor(m_pEngine->interpreter_b, 2);
    //output wav data of second model
    m_pEngine->output_details_b[0] = TfLiteInterpreterGetOutputTensor(m_pEngine->interpreter_b, 0);
    //output state data of second model
    m_pEngine->output_details_b[1] = TfLiteInterpreterGetOutputTensor(m_pEngine->interpreter_b, 1);
	float f32_output[BLOCK_LEN];
    std::vector<float>  testaecdata; //vector used to store enhanced data in a wav file
    AudioFile<float> inputmicfile;
    AudioFile<float> inputlpbfile;
    std::string micfile="./wav/mic.wav";
    std::string lpbfile="./wav/lpb.wav";
    inputmicfile.load(micfile);
    inputlpbfile.load(lpbfile);
    int audiolen=inputfile.getNumSamplesPerChannel();
    int process_num=audiolen/BLOCK_SHIFT;
    //for BLOCK_LEN input samples,do process_num infer
    for(int i=0;i<process_num;i++)
    {
        memmove(m_pEngine->mic_buffer, m_pEngine->mic_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
        memmove(m_pEngine->lpb_buffer, m_pEngine->lpb_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
        for(int n=0;n<BLOCK_SHIFT;n++){
                m_pEngine->mic_buffer[n+BLOCK_LEN-BLOCK_SHIFT]=inputmicfile.samples[0][n+i*BLOCK_SHIFT];
                m_pEngine->lpb_buffer[n+BLOCK_LEN-BLOCK_SHIFT]=inputlpbfile.samples[0][n+i*BLOCK_SHIFT];
            } 
        DTLNAECInfer(m_pEngine);
        for(int j=0;j<BLOCK_SHIFT;j++){
            testaecdata.push_back(m_pEngine->out_buffer[j]);    //for one forward process save first BLOCK_SHIFT model output samples
        }
    }
    ExportWAV("aectest.wav",testaecdata,SAMEPLERATE);


 }
 
void DTLNAECInfer(trg_engine* m_pEngine) {

    float in_mag[BLOCK_LEN / 2 + 1] = { 0 };
    float in_phase[BLOCK_LEN / 2 + 1] = { 0 };
    float estimated_block[BLOCK_LEN];
    
    float lpb_mag[FFT_OUT_SIZE]={0};
    float lpb_phase[FFT_OUT_SIZE]={0};
    double fft_in[BLOCK_LEN];
    std::vector<cpx_type> fft_res(BLOCK_LEN);
    double lpb_in[BLOCK_LEN];
    std::vector<cpx_type> lpb_res(BLOCK_LEN);

    std::vector<size_t> shape;
    shape.push_back(BLOCK_LEN);
    std::vector<size_t> axes;
    axes.push_back(0);
    std::vector<ptrdiff_t> stridel, strideo;
    strideo.push_back(sizeof(cpx_type));
    stridel.push_back(sizeof(double));
    for (int i = 0; i < BLOCK_LEN; i++){
        fft_in[i] = m_pEngine->mic_buffer[i];
	}
    for(int j =0;j<BLOCK_LEN;j++){
        lpb_in[j]= m_pEngine->lpb_buffer[j];
    }
    
    //apply FFT to input wav data
    pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, fft_in, fft_res.data(), 1.0);
	__calc_mag_phase(fft_res, in_mag, in_phase, FFT_OUT_SIZE);
    pocketfft::r2c(shape, stridel, strideo, axes, pocketfft::FORWARD, lpb_in,lpb_res.data(), 1.0);
	__calc_mag_phase(lpb_res, lpb_mag, lpb_phase, FFT_OUT_SIZE);
  
    //the data input of first model is the magnitude of input wav data
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[0], in_mag, FFT_OUT_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[2], lpb_mag, FFT_OUT_SIZE * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_a[1], m_pEngine->states_a, STATE_SIZE * sizeof(float));

    if (TfLiteInterpreterInvoke(m_pEngine->interpreter_a) != kTfLiteOk) {
        printf("Error invoking detection model\n");
    }

    float out_mask[FFT_OUT_SIZE];
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[0], out_mask, FFT_OUT_SIZE * sizeof(float));
    //the putput state of current block will become the input state of next block
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_a[1], m_pEngine->states_a, STATE_SIZE * sizeof(float));


    //ifft(FFT(input wav data)*out_mask)
	for (int i = 0; i < FFT_OUT_SIZE; i++) {
        fft_res[i] = cpx_type(in_mag[i] * out_mask[i] * cosf(in_phase[i]), 
						in_mag[i] * out_mask[i] * sinf(in_phase[i]));
	}

    pocketfft::c2r(shape, 
				strideo, 
				stridel, 
				axes, 
				pocketfft::BACKWARD, 
				fft_res.data(), 
				fft_in, 1.0);   
     
    for (int i = 0; i < BLOCK_LEN; i++)
        estimated_block[i] = fft_in[i] / BLOCK_LEN;   

    //the output data of first model will becomde the input data of second model
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_b[0], estimated_block, BLOCK_LEN * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_b[2], m_pEngine->lpb_buffer, BLOCK_LEN * sizeof(float));
    TfLiteTensorCopyFromBuffer(m_pEngine->input_details_b[1], m_pEngine->states_b, STATE_SIZE * sizeof(float));

    if (TfLiteInterpreterInvoke(m_pEngine->interpreter_b) != kTfLiteOk) {
        printf("Error invoking detection model");
    }

    float out_block[BLOCK_LEN];
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_b[0], out_block, BLOCK_LEN * sizeof(float));
    //the putput state of current block will become the input state of next block(same as first model)
    TfLiteTensorCopyToBuffer(m_pEngine->output_details_b[1],m_pEngine->states_b,STATE_SIZE*sizeof(float));

    //apply overlap_add
    memmove(m_pEngine->out_buffer, m_pEngine->out_buffer + BLOCK_SHIFT, (BLOCK_LEN - BLOCK_SHIFT) * sizeof(float));
    memset(m_pEngine->out_buffer + (BLOCK_LEN - BLOCK_SHIFT), 0, BLOCK_SHIFT * sizeof(float));
    for (int i = 0; i < BLOCK_LEN; i++)
        m_pEngine->out_buffer[i] += out_block[i];

}
 
void __calc_mag_phase(std::vector<cpx_type> fft_res, 
				float* in_mag, 
				float* in_phase, 
				int count) {
    for (int i = 0; i < count; i++) {
        in_mag[i] = sqrtf(fft_res[i].real() * fft_res[i].real() + fft_res[i].imag() * fft_res[i].imag());
        in_phase[i] = atan2f(fft_res[i].imag(), fft_res[i].real());
    }
}



