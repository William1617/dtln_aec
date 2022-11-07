import numpy as np
import time
import onnxruntime
import os
import soundfile as sf
block_len=512
block_shift=128
model_1=onnxruntime.InferenceSession('dtln_aec_1.onnx')
model_2=onnxruntime.InferenceSession('dtln_aec_2.onnx')
model_input_name1= [inp.name for inp in model_1.get_inputs()]

model_input_name2= [inp.name for inp in model_2.get_inputs()]

directory  = "./testdata/testmic/"
file_paths=[]
for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.
testlist=file_paths
lpbdir= "./testdata/testlpb/"
outpath= "./testdata/testmodel/"
for path in testlist:
    model_input1 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in model_1.get_inputs()}
    model_input2 = {
            inp.name: np.zeros(
                [dim if isinstance(dim, int) else 1 for dim in inp.shape],
                dtype=np.float32)
            for inp in model_2.get_inputs()}
    S=path.split('/') 
    wavname=S[-1]
    micaudio,rate =sf.read(path)
    lpbaudio,rate2=sf.read(lpbdir+wavname)
    in_buffer = np.zeros((block_len)).astype('float32')
    lpb_buffer=np.zeros((block_len)).astype('float32')
    out_buffer = np.zeros((block_len)).astype('float32')
# calculate number of blocks
    min_len=min(lpbaudio.shape[0],micaudio.shape[0])
    num_blocks = (min_len - (block_len-block_shift)) // block_shift
    out_file = np.zeros(min_len)
    for idx in range(num_blocks):
        in_buffer[:-block_shift] = in_buffer[block_shift:]
        in_buffer[-block_shift:] = micaudio[idx*block_shift:(idx*block_shift)+block_shift]
        lpb_buffer[:-block_shift] = lpb_buffer[block_shift:]
        lpb_buffer[-block_shift:] = lpbaudio[idx*block_shift:(idx*block_shift)+block_shift]
    # calculate fft of input block
        in_block_fft = np.fft.rfft(in_buffer)
        in_mag = np.abs(in_block_fft)
        in_phase = np.angle(in_block_fft)
        in_lpb_fft=np.fft.rfft(lpb_buffer)
        lpb_mag=np.abs(in_lpb_fft)
    # reshape magnitude to input dimensions
        in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
        lpb_mag=np.reshape(lpb_mag, (1,1,-1)).astype('float32')
        model_input1[model_input_name1[0]]=in_mag
        model_input1[model_input_name1[1]]=lpb_mag
        modelout1=model_1.run(None,model_input1)
        out_mask=modelout1[0]
        model_input1[model_input_name1[2]]=modelout1[1]
        estimated_complex = np.squeeze(out_mask)  * np.exp(1j * in_phase)
        estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
        estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
        lpb_frame=np.reshape(lpb_buffer,(1,1,-1)).astype('float32')
        model_input2[model_input_name2[0]]=estimated_block
        model_input2[model_input_name2[1]]=lpb_frame
        modelout2=model_2.run(None,model_input2)
        out_block=np.squeeze(modelout2[0])
        model_input2[model_input_name2[2]]=modelout2[1]
        out_buffer[:-block_shift] = out_buffer[block_shift:]
        out_buffer[-block_shift:] = np.zeros((block_shift))
        out_buffer  += np.squeeze(out_block)
    # write block to output file
        out_file[idx*block_shift:(idx*block_shift)+block_shift] = out_buffer[:block_shift]
    out_file=out_file.astype('float32')
    sf.write(outpath+wavname, out_file,rate)