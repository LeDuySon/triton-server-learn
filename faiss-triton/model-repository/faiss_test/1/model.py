import json
import os

import numpy as np
import faiss
# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        self.output_name = "output"
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, self.output_name)

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        
        # load indexer 
        self.index_path = os.path.join("/models", args['model_name'], 'index.bin')
        print(self.index_path, flush=True)
        
        self.faiss_index = self.load_index(self.index_path)
        self.k = 4 # num nearest

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        output0_dtype = self.output0_dtype

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            num_index_vector = self.faiss_index.ntotal 
            print("Num vector in index: ", num_index_vector)
            
            # Get INPUT0
            in_0 = pb_utils.get_input_tensor_by_name(request, "inp_vector")
            
            inp_vector = in_0.as_numpy()
            print("Input vector shape: ", inp_vector.shape, flush = True)
            
            results = self.get_k_nearest(inp_vector)
            
            self.add_vector(inp_vector)
            print(results, flush = True)
            
            out_tensor_0 = pb_utils.Tensor(self.output_name,
                                           results.astype(output0_dtype))

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occured"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...', flush=True)
        print('Save index to disk ...', flush = True)
        
        faiss_cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
        faiss.write_index(faiss_cpu_index, self.index_path)
        
        print("Done!", flush=True)

    def load_index(self, index_path):
        res = faiss.StandardGpuResources()  # use a single GPU
        index_flat = faiss.read_index(index_path)
        
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        return gpu_index_flat
    
    def get_k_nearest(self, vector):
        dist, ins = self.faiss_index.search(vector, self.k)
        
        return ins
    
    def add_vector(self, vector):
        self.faiss_index.add(vector)
        

        
