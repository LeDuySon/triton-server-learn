import argparse
import numpy as np
import sys
import queue
import uuid

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


def sync_send(triton_client, values, batch_size, sequence_id,
              model_name, model_version):

    count = 1
    result_list = []
    for value in values:
        # Create the tensor for INPUT
        inputs = []
        inputs.append(grpcclient.InferInput('lstm_input', [1, 1, 1], "FP32"))
        # Initialize the data
        inputs[0].set_data_from_numpy(np.array([[[value]]]).astype("float32"))
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('dense'))
        # Issue the synchronous sequence inference.
        result = triton_client.infer(model_name=model_name,
                                     inputs=inputs,
                                     outputs=outputs,
                                     sequence_id=sequence_id,
                                     sequence_start=int((count == 1)),
                                     sequence_end=int((count == len(values))))
        
        result_list.append(int(result.as_numpy('dense')[0][0]))
        count = count + 1
    return result_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument(
        '-u',
        '--url',
        type=str,
        required=False,
        default='localhost:8001',
        help='Inference server URL and it gRPC port. Default is localhost:8001.'
    )
    parser.add_argument('-d',
                        '--dyna',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Assume dynamic sequence model')
    parser.add_argument('-o',
                        '--offset',
                        type=int,
                        required=False,
                        default=0,
                        help='Add offset to sequence ID used')

    FLAGS = parser.parse_args()

    try:
        triton_client = grpcclient.InferenceServerClient(url=FLAGS.url,
                                                         verbose=FLAGS.verbose)
    except Exception as e:
        print("context creation failed: " + str(e))
        sys.exit()

    
    model_name = "sequencer_prod"
    sequence_id = 15
    model_version = ""
    batch_size = 1

    try:
        for i in range(10000):
            values = [0, 0, 0, 1, 0]
            if(i % 2 != 0):
                values[0] = 1
                
            result_list = sync_send(triton_client, values, batch_size,
                    sequence_id+i, model_name, model_version)
            
            print("Index: ", i)
            print("Value: ")
            print(values)
            print("Results: ")
            print(result_list)

    except InferenceServerException as error:
        print(error)
        sys.exit(1)
    
