# triton-server-learn

## Triton license plate
- Learn triton through pet project: OCR license plate
- Triton Concept:
    - Triton ensemble model
    ![alt text](https://raw.githubusercontent.com/LeDuySon/triton-server-learn/master/images/licenseplate.png)
    - Platform: tensorrt
    - Backend: Python Runtime

## Faiss triton
- Test faiss GPU on triton server -> benchmark performance 

## Sequencer triton 
- Test Sequencer batching on triton server
    - Utilizing for time-series model 
   - Refer: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#sequence-batcher

- Deploy prometheus + grafana + alertmanger for monitoring Triton server 
    - Alermanager: https://prometheus.io/docs/tutorials/alerting_based_on_metrics/ 