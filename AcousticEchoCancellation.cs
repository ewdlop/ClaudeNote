using AcousticEchoCancellation;
using NAudio.Dsp;
using NAudio.Wave;
using System;
using System.Linq;
using System.Numerics;

//not working I feel, or maybe it does work, but just for a nonlinear system
namespace AcousticEchoCancellation
{
    /// <summary>
    /// Base interface for all AEC algorithms
    /// </summary>
    public interface IAECAlgorithm
    {
        float ProcessSample(float micInput, float speakerReference);
        void Reset();
        string AlgorithmName { get; }
    }

    /// <summary>
    /// Performance metrics for AEC evaluation
    /// </summary>
    public class AECMetrics
    {
        private readonly int windowSize;
        private readonly Queue<float> inputPower;
        private readonly Queue<float> outputPower;
        private readonly Queue<float> errorBuffer;

        public float ERLE { get; private set; } // Echo Return Loss Enhancement
        public float ERL { get; private set; }  // Echo Return Loss
        public float PESQ { get; private set; } // Perceptual Evaluation of Speech Quality (simplified)
        public float ConvergenceTime { get; private set; }

        public AECMetrics(int windowSize = 1000)
        {
            this.windowSize = windowSize;
            inputPower = new Queue<float>(windowSize);
            outputPower = new Queue<float>(windowSize);
            errorBuffer = new Queue<float>(windowSize);
        }

        public void Update(float input, float output, float reference)
        {
            // Update buffers
            if (inputPower.Count >= windowSize) inputPower.Dequeue();
            if (outputPower.Count >= windowSize) outputPower.Dequeue();
            if (errorBuffer.Count >= windowSize) errorBuffer.Dequeue();

            inputPower.Enqueue(input * input);
            outputPower.Enqueue(output * output);
            errorBuffer.Enqueue(Math.Abs(output));

            // Calculate ERLE (in dB)
            float avgInputPower = inputPower.Average();
            float avgOutputPower = outputPower.Average();

            if (avgOutputPower > 0 && avgInputPower > 0)
            {
                ERLE = 10 * (float)Math.Log10(avgInputPower / avgOutputPower);
            }

            // Calculate ERL
            if (reference != 0 && input != 0)
            {
                ERL = 20 * (float)Math.Log10(Math.Abs(reference / input));
            }

            // Simplified PESQ estimation (0-5 scale)
            float snr = avgInputPower > 0 ? avgOutputPower / avgInputPower : 0;
            PESQ = Math.Max(1, Math.Min(5, 5 - 4 * snr));

            // Estimate convergence time (simplified)
            if (errorBuffer.Count > 10)
            {
                var recentErrors = errorBuffer.Skip(errorBuffer.Count - 10).ToArray();
                float errorVariance = CalculateVariance(recentErrors);
                if (errorVariance < 0.001f)
                {
                    ConvergenceTime = errorBuffer.Count / 16000f; // Assuming 16kHz sample rate
                }
            }
        }

        private float CalculateVariance(float[] data)
        {
            float mean = data.Average();
            float sumSquares = data.Sum(x => (x - mean) * (x - mean));
            return sumSquares / data.Length;
        }

        public void PrintMetrics()
        {
            Console.WriteLine($"=== AEC Performance Metrics ===");
            Console.WriteLine($"ERLE: {ERLE:F2} dB");
            Console.WriteLine($"ERL: {ERL:F2} dB");
            Console.WriteLine($"PESQ: {PESQ:F2}/5.0");
            Console.WriteLine($"Convergence Time: {ConvergenceTime:F3} seconds");
        }
    }

    /// <summary>
    /// Double-talk detector for improved AEC performance
    /// </summary>
    public class DoubleTalkDetector
    {
        private readonly int windowSize;
        private readonly float threshold;
        private readonly Queue<float> correlationBuffer;
        private bool isDoubleTalk;

        public bool IsDoubleTalk => isDoubleTalk;

        public DoubleTalkDetector(int windowSize = 100, float threshold = 0.7f)
        {
            this.windowSize = windowSize;
            this.threshold = threshold;
            correlationBuffer = new Queue<float>(windowSize);
        }

        public bool Detect(float[] micInput, float[] reference)
        {
            if (micInput.Length != reference.Length || micInput.Length == 0)
                return false;

            // Calculate normalized cross-correlation
            float correlation = CalculateCrossCorrelation(micInput, reference);

            // Update buffer
            if (correlationBuffer.Count >= windowSize)
                correlationBuffer.Dequeue();
            correlationBuffer.Enqueue(correlation);

            // Detect double-talk based on correlation drop
            float avgCorrelation = correlationBuffer.Average();
            isDoubleTalk = avgCorrelation < threshold;

            return isDoubleTalk;
        }

        private float CalculateCrossCorrelation(float[] x, float[] y)
        {
            float sumXY = 0, sumX2 = 0, sumY2 = 0;

            for (int i = 0; i < x.Length; i++)
            {
                sumXY += x[i] * y[i];
                sumX2 += x[i] * x[i];
                sumY2 += y[i] * y[i];
            }

            float denominator = (float)Math.Sqrt(sumX2 * sumY2);
            return denominator > 0 ? sumXY / denominator : 0;
        }
    }

    /// <summary>
    /// Complete AEC system with all algorithms and metrics
    /// </summary>
    public class ComprehensiveAECSystem
    {
        private WaveInEvent waveIn;
        private WaveOutEvent waveOut;
        private BufferedWaveProvider outputBuffer;
        private AECProcessor aecProcessor;
        private AECMetrics metrics;
        private DoubleTalkDetector dtDetector;
        private Timer metricsTimer;
        private bool isProcessing;

        public void RunBenchmark()
        {
            Console.WriteLine("Starting AEC Algorithm Benchmark...\n");

            // Test signal parameters
            int sampleRate = 16000;
            int duration = 5; // seconds
            int totalSamples = sampleRate * duration;

            // Generate test signals
            float[] echoSignal = GenerateTestSignal(totalSamples, 440); // 440Hz tone
            float[] nearEndSignal = GenerateTestSignal(totalSamples, 880); // 880Hz tone
            float[] mixedSignal = new float[totalSamples];

            // Create echo path (simplified room impulse response)
            float[] echoPath = GenerateEchoPath(512);

            // Mix signals with echo
            for (int i = 0; i < totalSamples; i++)
            {
                float echo = 0;
                for (int j = 0; j < Math.Min(echoPath.Length, i); j++)
                {
                    echo += echoPath[j] * echoSignal[i - j];
                }
                mixedSignal[i] = nearEndSignal[i] * 0.3f + echo * 0.7f; // Mix near-end and echo
            }

            // Test each algorithm
            var algorithms = new[]
            {
                AECProcessor.Algorithm.LMS,
                AECProcessor.Algorithm.NLMS,
                AECProcessor.Algorithm.RLS,
                AECProcessor.Algorithm.FDAF,
                AECProcessor.Algorithm.APA,
                AECProcessor.Algorithm.Kalman
            };

            foreach (var algo in algorithms)
            {
                aecProcessor = new AECProcessor();
                aecProcessor.SetAlgorithm(algo);
                metrics = new AECMetrics();

                Console.WriteLine($"\nTesting {aecProcessor.GetCurrentAlgorithmName()}...");

                var startTime = DateTime.Now;

                // Process samples
                for (int i = 0; i < totalSamples; i++)
                {
                    float output = aecProcessor.ProcessSample(mixedSignal[i], echoSignal[i]);
                    metrics.Update(mixedSignal[i], output, echoSignal[i]);
                }

                var processingTime = (DateTime.Now - startTime).TotalSeconds;

                // Print results
                metrics.PrintMetrics();
                Console.WriteLine($"Processing Time: {processingTime:F3} seconds");
                Console.WriteLine($"Real-time Factor: {duration / processingTime:F2}x");
            }

            Console.WriteLine("\nBenchmark Complete!");
        }

        private float[] GenerateTestSignal(int samples, float frequency)
        {
            float[] signal = new float[samples];
            float sampleRate = 16000;

            for (int i = 0; i < samples; i++)
            {
                signal[i] = (float)Math.Sin(2 * Math.PI * frequency * i / sampleRate);
            }

            return signal;
        }

        private float[] GenerateEchoPath(int length)
        {
            float[] path = new float[length];
            Random rand = new Random(42); // Fixed seed for reproducibility

            // Simple exponentially decaying echo path
            for (int i = 0; i < length; i++)
            {
                path[i] = (float)(Math.Exp(-i / 50.0) * (0.5 + 0.5 * rand.NextDouble()));
            }

            return path;
        }
    }

    /// <summary>
    /// Main program entry point
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Advanced AEC Implementation");
            Console.WriteLine("==========================\n");
            Console.WriteLine("1. Real-time AEC with algorithm switching");
            Console.WriteLine("2. Benchmark all algorithms");
            Console.WriteLine("3. File-based AEC processing");
            Console.Write("\nSelect option: ");

            var option = Console.ReadLine();

            switch (option)
            {
                case "1":
                    var realTimeAEC = new RealTimeAEC();
                    Console.Write("Enter test audio file path (or press Enter to skip): ");
                    var audioPath = Console.ReadLine();
                    realTimeAEC.Start(string.IsNullOrWhiteSpace(audioPath) ? null : audioPath);
                    Console.ReadKey();
                    realTimeAEC.Stop();
                    break;

                case "2":
                    var benchmark = new ComprehensiveAECSystem();
                    benchmark.RunBenchmark();
                    break;

                case "3":
                    Console.Write("Enter input file path: ");
                    var inputFile = Console.ReadLine();
                    Console.Write("Enter reference file path: ");
                    var referenceFile = Console.ReadLine();
                    ProcessFiles(inputFile, referenceFile);
                    break;

                default:
                    Console.WriteLine("Invalid option");
                    break;
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }

        static void ProcessFiles(string inputFile, string referenceFile)
        {
            // File processing implementation
            using var inputReader = new AudioFileReader(inputFile);
            using var referenceReader = new AudioFileReader(referenceFile);

            var processor = new AECProcessor();
            processor.SetAlgorithm(AECProcessor.Algorithm.NLMS);

            var outputFile = Path.GetFileNameWithoutExtension(inputFile) + "_processed.wav";

            using var writer = new WaveFileWriter(outputFile, inputReader.WaveFormat);

            float[] inputBuffer = new float[1024];
            float[] referenceBuffer = new float[1024];

            while (true)
            {
                int inputRead = inputReader.Read(inputBuffer, 0, inputBuffer.Length);
                int refRead = referenceReader.Read(referenceBuffer, 0, referenceBuffer.Length);

                if (inputRead == 0) break;

                for (int i = 0; i < inputRead; i++)
                {
                    float reference = i < refRead ? referenceBuffer[i] : 0;
                    inputBuffer[i] = processor.ProcessSample(inputBuffer[i], reference);
                }

                writer.WriteSamples(inputBuffer, 0, inputRead);
            }

            Console.WriteLine($"Processed file saved as: {outputFile}");
        }
    }
}

/// <summary>
/// Least Mean Squares (LMS) Algorithm
/// Simple but effective for stationary signals
/// </summary>
public class LMSAlgorithm : IAECAlgorithm
{
    private readonly int filterLength;
    private readonly float stepSize;
    private readonly float[] weights;
    private readonly float[] referenceBuffer;
    private int bufferIndex;

    public string AlgorithmName => "LMS (Least Mean Squares)";

    public LMSAlgorithm(int filterLength = 256, float stepSize = 0.01f)
    {
        this.filterLength = filterLength;
        this.stepSize = stepSize;
        weights = new float[filterLength];
        referenceBuffer = new float[filterLength];
        bufferIndex = 0;
    }

    public float ProcessSample(float micInput, float speakerReference)
    {
        // Update reference buffer
        referenceBuffer[bufferIndex] = speakerReference;

        // Calculate filter output (estimated echo)
        float estimatedEcho = 0;
        for (int i = 0; i < filterLength; i++)
        {
            int idx = (bufferIndex - i + filterLength) % filterLength;
            estimatedEcho += weights[i] * referenceBuffer[idx];
        }

        // Calculate error
        float error = micInput - estimatedEcho;

        // Update weights using LMS rule: w(n+1) = w(n) + μ * e(n) * x(n)
        for (int i = 0; i < filterLength; i++)
        {
            int idx = (bufferIndex - i + filterLength) % filterLength;
            weights[i] += stepSize * error * referenceBuffer[idx];
        }

        // Update circular buffer index
        bufferIndex = (bufferIndex + 1) % filterLength;

        return error;
    }

    public void Reset()
    {
        Array.Clear(weights, 0, filterLength);
        Array.Clear(referenceBuffer, 0, filterLength);
        bufferIndex = 0;
    }
}

/// <summary>
/// Normalized Least Mean Squares (NLMS) Algorithm
/// Improved version of LMS with normalized step size
/// </summary>
public class NLMSAlgorithm : IAECAlgorithm
{
    private readonly int filterLength;
    private readonly float stepSize;
    private readonly float regularization;
    private readonly float[] weights;
    private readonly float[] referenceBuffer;
    private int bufferIndex;
    private float powerEstimate;
    private readonly float forgetFactor;

    public string AlgorithmName => "NLMS (Normalized LMS)";

    public NLMSAlgorithm(int filterLength = 512, float stepSize = 0.5f, float regularization = 0.001f)
    {
        this.filterLength = filterLength;
        this.stepSize = stepSize;
        this.regularization = regularization;
        this.forgetFactor = 0.999f;
        weights = new float[filterLength];
        referenceBuffer = new float[filterLength];
        bufferIndex = 0;
        powerEstimate = 0;
    }

    public float ProcessSample(float micInput, float speakerReference)
    {
        // Update reference buffer
        referenceBuffer[bufferIndex] = speakerReference;

        // Update power estimate with exponential averaging
        powerEstimate = forgetFactor * powerEstimate + (1 - forgetFactor) * speakerReference * speakerReference;

        // Calculate filter output
        float estimatedEcho = 0;
        float instantPower = 0;
        for (int i = 0; i < filterLength; i++)
        {
            int idx = (bufferIndex - i + filterLength) % filterLength;
            float sample = referenceBuffer[idx];
            estimatedEcho += weights[i] * sample;
            instantPower += sample * sample;
        }

        // Calculate error
        float error = micInput - estimatedEcho;

        // Normalized step size
        float normalizedStep = stepSize / (instantPower + regularization);

        // Update weights using NLMS rule
        for (int i = 0; i < filterLength; i++)
        {
            int idx = (bufferIndex - i + filterLength) % filterLength;
            weights[i] += normalizedStep * error * referenceBuffer[idx];
        }

        // Update circular buffer index
        bufferIndex = (bufferIndex + 1) % filterLength;

        return error;
    }

    public void Reset()
    {
        Array.Clear(weights, 0, filterLength);
        Array.Clear(referenceBuffer, 0, filterLength);
        bufferIndex = 0;
        powerEstimate = 0;
    }
}

/// <summary>
/// Recursive Least Squares (RLS) Algorithm
/// Faster convergence but more computationally intensive
/// </summary>
public class RLSAlgorithm : IAECAlgorithm
{
    private readonly int filterLength;
    private readonly float forgettingFactor;
    private readonly float delta;
    private readonly float[] weights;
    private readonly float[] referenceBuffer;
    private readonly float[,] P; // Inverse correlation matrix
    private readonly float[] k; // Gain vector
    private int bufferIndex;

    public string AlgorithmName => "RLS (Recursive Least Squares)";

    public RLSAlgorithm(int filterLength = 128, float forgettingFactor = 0.999f, float delta = 0.1f)
    {
        this.filterLength = filterLength;
        this.forgettingFactor = forgettingFactor;
        this.delta = delta;

        weights = new float[filterLength];
        referenceBuffer = new float[filterLength];
        P = new float[filterLength, filterLength];
        k = new float[filterLength];

        // Initialize P matrix
        for (int i = 0; i < filterLength; i++)
        {
            P[i, i] = 1.0f / delta;
        }

        bufferIndex = 0;
    }

    public float ProcessSample(float micInput, float speakerReference)
    {
        // Update reference buffer
        referenceBuffer[bufferIndex] = speakerReference;

        // Get current input vector
        float[] x = new float[filterLength];
        for (int i = 0; i < filterLength; i++)
        {
            x[i] = referenceBuffer[(bufferIndex - i + filterLength) % filterLength];
        }

        // Calculate filter output
        float estimatedEcho = 0;
        for (int i = 0; i < filterLength; i++)
        {
            estimatedEcho += weights[i] * x[i];
        }

        // Calculate error
        float error = micInput - estimatedEcho;

        // Calculate gain vector k = P*x / (λ + x'*P*x)
        float[] Px = new float[filterLength];
        float xPx = 0;

        for (int i = 0; i < filterLength; i++)
        {
            Px[i] = 0;
            for (int j = 0; j < filterLength; j++)
            {
                Px[i] += P[i, j] * x[j];
            }
            xPx += x[i] * Px[i];
        }

        float denominator = forgettingFactor + xPx;

        for (int i = 0; i < filterLength; i++)
        {
            k[i] = Px[i] / denominator;
        }

        // Update weights
        for (int i = 0; i < filterLength; i++)
        {
            weights[i] += k[i] * error;
        }

        // Update P matrix: P = (P - k*x'*P) / λ
        for (int i = 0; i < filterLength; i++)
        {
            for (int j = 0; j < filterLength; j++)
            {
                P[i, j] = (P[i, j] - k[i] * Px[j]) / forgettingFactor;
            }
        }

        // Update circular buffer index
        bufferIndex = (bufferIndex + 1) % filterLength;

        return error;
    }

    public void Reset()
    {
        Array.Clear(weights, 0, filterLength);
        Array.Clear(referenceBuffer, 0, filterLength);
        Array.Clear(P, 0, P.Length);
        Array.Clear(k, 0, filterLength);

        // Reinitialize P matrix
        for (int i = 0; i < filterLength; i++)
        {
            P[i, i] = 1.0f / delta;
        }

        bufferIndex = 0;
    }
}

/// <summary>
/// Frequency Domain Adaptive Filter (FDAF)
/// Efficient for long filters using FFT
/// </summary>
public class FDAFAlgorithm : IAECAlgorithm
{
    private readonly int blockSize;
    private readonly int filterLength;
    private readonly float stepSize;
    private readonly float regularization;
    private readonly System.Numerics.Complex[] weights;
    private readonly float[] inputBuffer;
    private readonly float[] referenceBuffer;
    private readonly System.Numerics.Complex[] inputFFT;
    private readonly System.Numerics.Complex[] referenceFFT;
    private readonly System.Numerics.Complex[] errorFFT;
    private int bufferIndex;

    public string AlgorithmName => "FDAF (Frequency Domain Adaptive Filter)";

    public FDAFAlgorithm(int blockSize = 256, float stepSize = 0.1f, float regularization = 0.001f)
    {
        this.blockSize = blockSize;
        this.filterLength = blockSize * 2;
        this.stepSize = stepSize;
        this.regularization = regularization;

        weights = new System.Numerics.Complex[filterLength];
        inputBuffer = new float[filterLength];
        referenceBuffer = new float[filterLength];
        inputFFT = new System.Numerics.Complex[filterLength];
        referenceFFT = new System.Numerics.Complex[filterLength];
        errorFFT = new System.Numerics.Complex[filterLength];
        bufferIndex = 0;
    }

    public float ProcessSample(float micInput, float speakerReference)
    {
        // Buffer input samples
        inputBuffer[bufferIndex] = micInput;
        referenceBuffer[bufferIndex] = speakerReference;

        float output = micInput; // Default passthrough

        // Process when we have a full block
        if ((bufferIndex + 1) % blockSize == 0)
        {
            ProcessBlock();

            // Get the last processed sample as output
            int idx = (bufferIndex - blockSize + 1 + filterLength) % filterLength;
            output = inputBuffer[idx];
        }

        bufferIndex = (bufferIndex + 1) % filterLength;
        return output;
    }

    private void ProcessBlock()
    {
        // Prepare data for FFT
        for (int i = 0; i < filterLength; i++)
        {
            referenceFFT[i] = new System.Numerics.Complex(referenceBuffer[i], 0);
        }

        // Forward FFT of reference
        FastFourierTransform(referenceFFT, true);

        // Calculate filter output in frequency domain
        System.Numerics.Complex[] outputFFT = new System.Numerics.Complex[filterLength];
        for (int i = 0; i < filterLength; i++)
        {
            outputFFT[i] = weights[i] * referenceFFT[i];
        }

        // Inverse FFT to get time domain output
        FastFourierTransform(outputFFT, false);

        // Calculate error and update weights
        for (int i = 0; i < blockSize; i++)
        {
            int idx = (bufferIndex - blockSize + 1 + i + filterLength) % filterLength;
            float estimatedEcho = (float)outputFFT[i].Real / filterLength;
            float error = inputBuffer[idx] - estimatedEcho;
            errorFFT[i] = new System.Numerics.Complex(error, 0);

            // Store cleaned signal back
            inputBuffer[idx] = error;
        }

        // Zero-pad error for FFT
        for (int i = blockSize; i < filterLength; i++)
        {
            errorFFT[i] = System.Numerics.Complex.Zero;
        }

        // Forward FFT of error
        FastFourierTransform(errorFFT, true);

        // Update weights in frequency domain
        for (int i = 0; i < filterLength; i++)
        {
            System.Numerics.Complex conj = System.Numerics.Complex.Conjugate(referenceFFT[i]);
            float power = (float)(referenceFFT[i].Real * referenceFFT[i].Real +
                                referenceFFT[i].Imaginary * referenceFFT[i].Imaginary);
            float normalizedStep = stepSize / (power + regularization);
            weights[i] += normalizedStep * errorFFT[i] * conj;
        }
    }

    private void FastFourierTransform(System.Numerics.Complex[] data, bool forward)
    {
        int n = data.Length;
        if (n <= 1) return;

        // Bit reversal
        int j = 0;
        for (int i = 1; i < n - 1; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }
            j ^= bit;

            if (i < j)
            {
                System.Numerics.Complex temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }

        // FFT computation
        for (int len = 2; len <= n; len <<= 1)
        {
            double angle = 2 * Math.PI / len * (forward ? -1 : 1);
            System.Numerics.Complex wlen = new System.Numerics.Complex(Math.Cos(angle), Math.Sin(angle));

            for (int i = 0; i < n; i += len)
            {
                System.Numerics.Complex w = System.Numerics.Complex.One;
                for (int k = 0; k < len / 2; k++)
                {
                    System.Numerics.Complex u = data[i + k];
                    System.Numerics.Complex v = data[i + k + len / 2] * w;
                    data[i + k] = u + v;
                    data[i + k + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
    }

    public void Reset()
    {
        Array.Clear(weights, 0, filterLength);
        Array.Clear(inputBuffer, 0, filterLength);
        Array.Clear(referenceBuffer, 0, filterLength);
        bufferIndex = 0;
    }
}

/// <summary>
/// Affine Projection Algorithm (APA)
/// Better convergence than NLMS for correlated signals
/// </summary>
public class APAAlgorithm : IAECAlgorithm
{
    private readonly int filterLength;
    private readonly int projectionOrder;
    private readonly float stepSize;
    private readonly float regularization;
    private readonly float[] weights;
    private readonly float[,] X; // Reference matrix
    private readonly float[] d; // Desired signal vector
    private int bufferIndex;

    public string AlgorithmName => "APA (Affine Projection Algorithm)";

    public APAAlgorithm(int filterLength = 256, int projectionOrder = 4, float stepSize = 0.5f)
    {
        this.filterLength = filterLength;
        this.projectionOrder = projectionOrder;
        this.stepSize = stepSize;
        this.regularization = 0.001f;

        weights = new float[filterLength];
        X = new float[filterLength, projectionOrder];
        d = new float[projectionOrder];
        bufferIndex = 0;
    }

    public float ProcessSample(float micInput, float speakerReference)
    {
        // Update reference matrix (shift and add new sample)
        for (int i = filterLength - 1; i > 0; i--)
        {
            for (int j = 0; j < projectionOrder; j++)
            {
                X[i, j] = X[i - 1, j];
            }
        }

        // Add new reference sample to first row
        X[0, 0] = speakerReference;
        for (int j = 1; j < projectionOrder; j++)
        {
            X[0, j] = 0; // Zero-pad for now (simplified)
        }

        // Update desired signal vector
        for (int i = projectionOrder - 1; i > 0; i--)
        {
            d[i] = d[i - 1];
        }
        d[0] = micInput;

        // Calculate output vector Y = X^T * w
        float[] y = new float[projectionOrder];
        for (int j = 0; j < projectionOrder; j++)
        {
            y[j] = 0;
            for (int i = 0; i < filterLength; i++)
            {
                y[j] += X[i, j] * weights[i];
            }
        }

        // Calculate error vector e = d - y
        float[] e = new float[projectionOrder];
        for (int j = 0; j < projectionOrder; j++)
        {
            e[j] = d[j] - y[j];
        }

        // Calculate X^T * X + δI (regularized autocorrelation)
        float[,] R = new float[projectionOrder, projectionOrder];
        for (int i = 0; i < projectionOrder; i++)
        {
            for (int j = 0; j < projectionOrder; j++)
            {
                R[i, j] = 0;
                for (int k = 0; k < filterLength; k++)
                {
                    R[i, j] += X[k, i] * X[k, j];
                }
                if (i == j) R[i, j] += regularization;
            }
        }

        // Solve for gain: g = (X^T * X + δI)^-1 * e
        float[] g = SolveLinearSystem(R, e);

        // Update weights: w = w + μ * X * g
        for (int i = 0; i < filterLength; i++)
        {
            float update = 0;
            for (int j = 0; j < projectionOrder; j++)
            {
                update += X[i, j] * g[j];
            }
            weights[i] += stepSize * update;
        }

        bufferIndex = (bufferIndex + 1) % filterLength;
        return e[0]; // Return current error
    }

    private float[] SolveLinearSystem(float[,] A, float[] b)
    {
        int n = b.Length;
        float[] x = new float[n];
        float[,] L = new float[n, n];

        // Cholesky decomposition (simplified - assumes positive definite)
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                float sum = A[i, j];
                for (int k = 0; k < j; k++)
                {
                    sum -= L[i, k] * L[j, k];
                }

                if (i == j)
                {
                    L[i, j] = (float)Math.Sqrt(Math.Max(sum, 1e-10));
                }
                else
                {
                    L[i, j] = sum / L[j, j];
                }
            }
        }

        // Forward substitution
        float[] y = new float[n];
        for (int i = 0; i < n; i++)
        {
            float sum = b[i];
            for (int j = 0; j < i; j++)
            {
                sum -= L[i, j] * y[j];
            }
            y[i] = sum / L[i, i];
        }

        // Back substitution
        for (int i = n - 1; i >= 0; i--)
        {
            float sum = y[i];
            for (int j = i + 1; j < n; j++)
            {
                sum -= L[j, i] * x[j];
            }
            x[i] = sum / L[i, i];
        }

        return x;
    }

    public void Reset()
    {
        Array.Clear(weights, 0, filterLength);
        Array.Clear(X, 0, X.Length);
        Array.Clear(d, 0, projectionOrder);
        bufferIndex = 0;
    }
}

/// <summary>
/// Kalman Filter based AEC
/// Optimal for systems with known noise characteristics
/// </summary>
public class KalmanAECAlgorithm : IAECAlgorithm
{
    private readonly int stateSize;
    private readonly float[] state;
    private readonly float[,] P; // Error covariance
    private readonly float Q; // Process noise
    private readonly float R; // Measurement noise
    private readonly float[] referenceBuffer;
    private int bufferIndex;

    public string AlgorithmName => "Kalman Filter AEC";

    public KalmanAECAlgorithm(int stateSize = 128, float processNoise = 0.001f, float measurementNoise = 0.1f)
    {
        this.stateSize = stateSize;
        this.Q = processNoise;
        this.R = measurementNoise;

        state = new float[stateSize];
        P = new float[stateSize, stateSize];
        referenceBuffer = new float[stateSize];

        // Initialize covariance matrix
        for (int i = 0; i < stateSize; i++)
        {
            P[i, i] = 1.0f;
        }

        bufferIndex = 0;
    }

    public float ProcessSample(float micInput, float speakerReference)
    {
        // Update reference buffer
        referenceBuffer[bufferIndex] = speakerReference;

        // Measurement update (observation model)
        float[] H = new float[stateSize];
        for (int i = 0; i < stateSize; i++)
        {
            H[i] = referenceBuffer[(bufferIndex - i + stateSize) % stateSize];
        }

        // Calculate predicted measurement
        float y_pred = 0;
        for (int i = 0; i < stateSize; i++)
        {
            y_pred += H[i] * state[i];
        }

        // Innovation (measurement residual)
        float innovation = micInput - y_pred;

        // Innovation covariance
        float S = R;
        for (int i = 0; i < stateSize; i++)
        {
            for (int j = 0; j < stateSize; j++)
            {
                S += H[i] * P[i, j] * H[j];
            }
        }

        // Kalman gain
        float[] K = new float[stateSize];
        for (int i = 0; i < stateSize; i++)
        {
            float sum = 0;
            for (int j = 0; j < stateSize; j++)
            {
                sum += P[i, j] * H[j];
            }
            K[i] = sum / S;
        }

        // State update
        for (int i = 0; i < stateSize; i++)
        {
            state[i] += K[i] * innovation;
        }

        // Covariance update
        float[,] P_new = new float[stateSize, stateSize];
        for (int i = 0; i < stateSize; i++)
        {
            for (int j = 0; j < stateSize; j++)
            {
                P_new[i, j] = P[i, j] - K[i] * H[j] * S;
                if (i == j) P_new[i, j] += Q; // Add process noise
            }
        }
        Array.Copy(P_new, P, P.Length);

        bufferIndex = (bufferIndex + 1) % stateSize;

        // Return echo-cancelled output
        return innovation;
    }

    public void Reset()
    {
        Array.Clear(state, 0, stateSize);
        Array.Clear(P, 0, P.Length);
        Array.Clear(referenceBuffer, 0, stateSize);

        // Reinitialize covariance
        for (int i = 0; i < stateSize; i++)
        {
            P[i, i] = 1.0f;
        }

        bufferIndex = 0;
    }
}

/// <summary>
/// Main AEC processor that can switch between algorithms
/// </summary>
public class AECProcessor
{
    private IAECAlgorithm currentAlgorithm;
    private readonly LMSAlgorithm lms;
    private readonly NLMSAlgorithm nlms;
    private readonly RLSAlgorithm rls;
    private readonly FDAFAlgorithm fdaf;
    private readonly APAAlgorithm apa;
    private readonly KalmanAECAlgorithm kalman;

    public enum Algorithm
    {
        LMS,
        NLMS,
        RLS,
        FDAF,
        APA,
        Kalman
    }

    public AECProcessor()
    {
        lms = new LMSAlgorithm();
        nlms = new NLMSAlgorithm();
        rls = new RLSAlgorithm();
        fdaf = new FDAFAlgorithm();
        apa = new APAAlgorithm();
        kalman = new KalmanAECAlgorithm();

        currentAlgorithm = nlms; // Default to NLMS
    }

    public void SetAlgorithm(Algorithm algo)
    {
        currentAlgorithm = algo switch
        {
            Algorithm.LMS => lms,
            Algorithm.NLMS => nlms,
            Algorithm.RLS => rls,
            Algorithm.FDAF => fdaf,
            Algorithm.APA => apa,
            Algorithm.Kalman => kalman,
            _ => nlms
        };

        Console.WriteLine($"Switched to {currentAlgorithm.AlgorithmName}");
    }

    public float ProcessSample(float micInput, float speakerReference)
    {
        return currentAlgorithm.ProcessSample(micInput, speakerReference);
    }

    public void Reset()
    {
        currentAlgorithm.Reset();
    }

    public string GetCurrentAlgorithmName()
    {
        return currentAlgorithm.AlgorithmName;
    }
}

/// <summary>
/// Real-time AEC application with algorithm selection
/// </summary>
public class RealTimeAEC
{
    private WaveInEvent waveIn;
    private WaveOutEvent waveOut;
    private BufferedWaveProvider outputBuffer;
    private AECProcessor aecProcessor;
    private AudioFileReader audioFile;
    private float[] referenceBuffer;
    private int referenceIndex;
    private bool isProcessing;

    public void Start(string testAudioFile = null)
    {
        aecProcessor = new AECProcessor();

        // Setup microphone
        waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(16000, 16, 1), // 16kHz for better real-time performance
            BufferMilliseconds = 10,
            NumberOfBuffers = 3,
            DeviceNumber= 1 // Default microphone
        };

        // Setup output
        outputBuffer = new BufferedWaveProvider(waveIn.WaveFormat)
        {
            BufferLength = waveIn.WaveFormat.SampleRate * 2,
            DiscardOnBufferOverflow = true
        };

        waveOut = new WaveOutEvent
        {
            DesiredLatency = 50
        };

        // Load test audio if provided
        if (!string.IsNullOrEmpty(testAudioFile))
        {
            audioFile = new AudioFileReader(testAudioFile);
            referenceBuffer = new float[audioFile.Length / 4];
            audioFile.Read(referenceBuffer, 0, referenceBuffer.Length);
            referenceIndex = 0;
        }

        waveIn.DataAvailable += OnDataAvailable;

        waveOut.Init(outputBuffer);
        waveOut.Play();
        waveIn.StartRecording();

        isProcessing = true;

        Console.WriteLine("AEC Started. Commands:");
        Console.WriteLine("1-6: Switch algorithm (LMS/NLMS/RLS/FDAF/APA/Kalman)");
        Console.WriteLine("Q: Quit");

        // Handle user input
        Task.Run(() =>
        {
            while (isProcessing)
            {
                var key = Console.ReadKey(true).KeyChar;
                switch (key)
                {
                    case '1': aecProcessor.SetAlgorithm(AECProcessor.Algorithm.LMS); break;
                    case '2': aecProcessor.SetAlgorithm(AECProcessor.Algorithm.NLMS); break;
                    case '3': aecProcessor.SetAlgorithm(AECProcessor.Algorithm.RLS); break;
                    case '4': aecProcessor.SetAlgorithm(AECProcessor.Algorithm.FDAF); break;
                    case '5': aecProcessor.SetAlgorithm(AECProcessor.Algorithm.APA); break;
                    case '6': aecProcessor.SetAlgorithm(AECProcessor.Algorithm.Kalman); break;
                    case 'q':
                    case 'Q':
                        Stop();
                        break;
                    default:
                        break;
                }
            }
        }).GetAwaiter().GetResult();
    }

    private void OnDataAvailable(object sender, WaveInEventArgs e)
    {
        if (!isProcessing) return;

        // Convert to float samples
        float[] samples = new float[e.BytesRecorded / 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short sample = BitConverter.ToInt16(e.Buffer, i * 2);
            samples[i] = sample / 32768f;
        }

        // Process through AEC
        float[] processed = new float[samples.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            // Get reference signal (from file or generate)
            float reference = GetReferenceSample();

            // Apply AEC
            processed[i] = aecProcessor.ProcessSample(samples[i], reference);
        }

        // Convert back to bytes
        byte[] outputBytes = new byte[processed.Length * 2];
        for (int i = 0; i < processed.Length; i++)
        {
            short sample = (short)(processed[i] * 32767f);
            BitConverter.GetBytes(sample).CopyTo(outputBytes, i * 2);
        }

        outputBuffer.AddSamples(outputBytes, 0, outputBytes.Length);
    }

    private float GetReferenceSample()
    {
        if (referenceBuffer != null && referenceBuffer.Length > 0)
        {
            float sample = referenceBuffer[referenceIndex];
            referenceIndex = (referenceIndex + 1) % referenceBuffer.Length;
            return sample;
        }
        return 0;
    }

    public void Stop()
    {
        isProcessing = false;

        waveIn?.StopRecording();
        waveIn?.Dispose();

        waveOut?.Stop();
        waveOut?.Dispose();

        audioFile?.Dispose();

        Console.WriteLine("AEC Stopped.");
    }
}
