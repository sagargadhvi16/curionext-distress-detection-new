##Feature Extraction Optimization

## Objective
Profile and optimize the audio feature extraction pipeline to meet real-time constraints (<100 ms per audio clip).

### Performance Results

Benchmark configuration:
- Sample rate: 16 kHz
- Audio duration: 3 seconds
- Hardware: CPU(local machine)

| Pipeline                           | Time (ms)     |
|------------------------------------|-------------=-|
 Handcrafted feature aggregation     | ~75–85 ms     |
 YAMNet embedding (CPU)              | ~800–1200 ms  |

## Key Observations
- Profiling individual feature functions overestimates runtime due to repeated computation.
- Aggregated feature extraction avoids redundancy and reflects actual pipeline performance.
- YAMNet is a deep model with higher expected latency and is profiled separately.

## Conclusion
The handcrafted audio feature pipeline meets real-time requirements.  
YAMNet is retained for richer semantic understanding and handled as a separate, non–real-time component.

### Notes
Standalone profiling of individual feature functions overestimates runtime due to repeated spectral and pitch computation. The aggregated pipeline avoids redundant calculations and represents actual runtime behavior.
