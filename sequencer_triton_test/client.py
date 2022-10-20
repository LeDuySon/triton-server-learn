import argparse
import numpy as np
import sys

import tritonclient.grpc as grpcclient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')
    parser.add_argument('-s',
                        '--ssl',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable SSL encrypted channel to the server')
    parser.add_argument('-t',
                        '--client-timeout',
                        type=float,
                        required=False,
                        default=None,
                        help='Client timeout in seconds. Default is None.')
    parser.add_argument(
        '-r',
        '--root-certificates',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded root certificates. Default is None.')
    parser.add_argument(
        '-p',
        '--private-key',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded private key. Default is None.')
    parser.add_argument(
        '-x',
        '--certificate-chain',
        type=str,
        required=False,
        default=None,
        help='File holding PEM-encoded certicate chain. Default is None.')
    parser.add_argument(
        '-C',
        '--grpc-compression-algorithm',
        type=str,
        required=False,
        default=None,
        help=
        'The compression algorithm to be used when sending request to server. Default is None.'
    )

    FLAGS = parser.parse_args()
    try:
        triton_client = grpcclient.InferenceServerClient(
            url=FLAGS.url,
            verbose=FLAGS.verbose,
            ssl=FLAGS.ssl,
            root_certificates=FLAGS.root_certificates,
            private_key=FLAGS.private_key,
            certificate_chain=FLAGS.certificate_chain)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    model_name = "sequencer_test"

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput('lstm_input', [1, 1, 1], "FP32"))

    # Create the data for the two input tensors. Initialize the first
    # to unique integers and the second to all ones.
    input0_data = np.array([[[0]]]).astype("float32")

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data)

    outputs.append(grpcclient.InferRequestedOutput('dense'))

    # Test with outputs
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        client_timeout=FLAGS.client_timeout,
        headers={'test': '1'},
        compression_algorithm=FLAGS.grpc_compression_algorithm)

    # statistics = triton_client.get_inference_statistics(model_name=model_name)
    # print(statistics)
    # if len(statistics.model_stats) != 1:
    #     print("FAILED: Inference Statistics")
    #     sys.exit(1)

    # Get the output arrays from the results
    output0_data = results.as_numpy('dense')

    print(output0_data)