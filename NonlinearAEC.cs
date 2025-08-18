using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using NAudio.Wave;

//do not work
namespace NonlinearAEC
{
    /// <summary>
    /// Volterra Filter for nonlinear echo cancellation
    /// Handles polynomial nonlinearities up to specified order
    /// </summary>
    public class VolterraFilter
    {
        private readonly int memoryLength;
        private readonly int nonlinearOrder;
        private readonly float stepSize;
        private readonly float regularization;

        // Linear kernel
        private float[] h1;

        // Quadratic kernel (2nd order)
        private float[,] h2;

        // Cubic kernel (3rd order)
        private float[,,] h3;

        private readonly float[] referenceBuffer;
        private int bufferIndex;

        public string AlgorithmName => "Volterra Filter (Nonlinear)";

        public VolterraFilter(int memoryLength = 128, int nonlinearOrder = 3, float stepSize = 0.01f)
        {
            this.memoryLength = memoryLength;
            this.nonlinearOrder = Math.Min(nonlinearOrder, 3); // Limit to 3rd order for practicality
            this.stepSize = stepSize;
            this.regularization = 0.001f;

            // Initialize kernels
            h1 = new float[memoryLength];

            if (nonlinearOrder >= 2)
                h2 = new float[memoryLength, memoryLength];

            if (nonlinearOrder >= 3)
                h3 = new float[memoryLength, memoryLength, memoryLength];

            referenceBuffer = new float[memoryLength];
            bufferIndex = 0;
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            // Update reference buffer
            referenceBuffer[bufferIndex] = speakerReference;

            // Calculate Volterra series output
            float y = 0;

            // Linear term
            for (int i = 0; i < memoryLength; i++)
            {
                int idx1 = (bufferIndex - i + memoryLength) % memoryLength;
                y += h1[i] * referenceBuffer[idx1];
            }

            // Quadratic term
            if (nonlinearOrder >= 2 && h2 != null)
            {
                for (int i = 0; i < memoryLength; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        int idx1 = (bufferIndex - i + memoryLength) % memoryLength;
                        int idx2 = (bufferIndex - j + memoryLength) % memoryLength;
                        y += h2[i, j] * referenceBuffer[idx1] * referenceBuffer[idx2];
                    }
                }
            }

            // Cubic term
            if (nonlinearOrder >= 3 && h3 != null)
            {
                for (int i = 0; i < memoryLength / 2; i++) // Reduced for computational efficiency
                {
                    for (int j = 0; j <= i; j++)
                    {
                        for (int k = 0; k <= j; k++)
                        {
                            int idx1 = (bufferIndex - i + memoryLength) % memoryLength;
                            int idx2 = (bufferIndex - j + memoryLength) % memoryLength;
                            int idx3 = (bufferIndex - k + memoryLength) % memoryLength;
                            y += h3[i, j, k] * referenceBuffer[idx1] * referenceBuffer[idx2] * referenceBuffer[idx3];
                        }
                    }
                }
            }

            // Calculate error
            float error = micInput - y;

            // Update kernels using gradient descent
            UpdateKernels(error);

            // Update buffer index
            bufferIndex = (bufferIndex + 1) % memoryLength;

            return error;
        }

        private void UpdateKernels(float error)
        {
            float adaptiveStep = stepSize * error;

            // Update linear kernel
            for (int i = 0; i < memoryLength; i++)
            {
                int idx = (bufferIndex - i + memoryLength) % memoryLength;
                h1[i] += adaptiveStep * referenceBuffer[idx];
            }

            // Update quadratic kernel
            if (nonlinearOrder >= 2 && h2 != null)
            {
                float quadraticStep = adaptiveStep * 0.5f; // Reduced step for higher orders
                for (int i = 0; i < memoryLength; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        int idx1 = (bufferIndex - i + memoryLength) % memoryLength;
                        int idx2 = (bufferIndex - j + memoryLength) % memoryLength;
                        h2[i, j] += quadraticStep * referenceBuffer[idx1] * referenceBuffer[idx2];
                    }
                }
            }

            // Update cubic kernel
            if (nonlinearOrder >= 3 && h3 != null)
            {
                float cubicStep = adaptiveStep * 0.25f; // Further reduced for stability
                for (int i = 0; i < memoryLength / 2; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        for (int k = 0; k <= j; k++)
                        {
                            int idx1 = (bufferIndex - i + memoryLength) % memoryLength;
                            int idx2 = (bufferIndex - j + memoryLength) % memoryLength;
                            int idx3 = (bufferIndex - k + memoryLength) % memoryLength;
                            h3[i, j, k] += cubicStep * referenceBuffer[idx1] * referenceBuffer[idx2] * referenceBuffer[idx3];
                        }
                    }
                }
            }
        }

        public void Reset()
        {
            Array.Clear(h1, 0, h1.Length);
            if (h2 != null) Array.Clear(h2, 0, h2.Length);
            if (h3 != null) Array.Clear(h3, 0, h3.Length);
            Array.Clear(referenceBuffer, 0, referenceBuffer.Length);
            bufferIndex = 0;
        }
    }

    /// <summary>
    /// Hammerstein Model - Nonlinear static function followed by linear filter
    /// Efficient for memoryless nonlinearities
    /// </summary>
    public class HammersteinAEC
    {
        private readonly int filterLength;
        private readonly int polynomialOrder;
        private readonly float stepSize;

        private float[] polynomialCoeffs;
        private float[] linearFilter;
        private float[] referenceBuffer;
        private float[] nonlinearBuffer;
        private int bufferIndex;

        public string AlgorithmName => "Hammerstein Model AEC";

        public HammersteinAEC(int filterLength = 256, int polynomialOrder = 5, float stepSize = 0.01f)
        {
            this.filterLength = filterLength;
            this.polynomialOrder = polynomialOrder;
            this.stepSize = stepSize;

            polynomialCoeffs = new float[polynomialOrder + 1];
            polynomialCoeffs[1] = 1.0f; // Initialize with linear pass-through

            linearFilter = new float[filterLength];
            referenceBuffer = new float[filterLength];
            nonlinearBuffer = new float[filterLength];
            bufferIndex = 0;
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            // Apply nonlinear function (polynomial)
            float nonlinearOutput = ApplyPolynomial(speakerReference);

            // Store in nonlinear buffer
            nonlinearBuffer[bufferIndex] = nonlinearOutput;
            referenceBuffer[bufferIndex] = speakerReference;

            // Apply linear filter
            float filterOutput = 0;
            for (int i = 0; i < filterLength; i++)
            {
                int idx = (bufferIndex - i + filterLength) % filterLength;
                filterOutput += linearFilter[i] * nonlinearBuffer[idx];
            }

            // Calculate error
            float error = micInput - filterOutput;

            // Update both polynomial and filter coefficients
            UpdateCoefficients(error, speakerReference);

            bufferIndex = (bufferIndex + 1) % filterLength;

            return error;
        }

        private float ApplyPolynomial(float x)
        {
            float result = polynomialCoeffs[0];
            float xPower = x;

            for (int i = 1; i <= polynomialOrder; i++)
            {
                result += polynomialCoeffs[i] * xPower;
                xPower *= x;
            }

            // Soft clipping to prevent instability
            return Math.Max(-2.0f, Math.Min(2.0f, result));
        }

        private void UpdateCoefficients(float error, float currentReference)
        {
            // Update linear filter coefficients
            for (int i = 0; i < filterLength; i++)
            {
                int idx = (bufferIndex - i + filterLength) % filterLength;
                linearFilter[i] += stepSize * error * nonlinearBuffer[idx];
            }

            // Update polynomial coefficients
            float xPower = 1.0f;
            for (int i = 0; i <= polynomialOrder; i++)
            {
                // Gradient of polynomial with respect to coefficients
                float gradient = 0;
                for (int j = 0; j < filterLength; j++)
                {
                    int idx = (bufferIndex - j + filterLength) % filterLength;
                    gradient += linearFilter[j] * xPower;
                }

                polynomialCoeffs[i] += stepSize * error * gradient * 0.1f; // Reduced rate for polynomial
                xPower *= currentReference;
            }
        }

        public void Reset()
        {
            Array.Clear(polynomialCoeffs, 0, polynomialCoeffs.Length);
            polynomialCoeffs[1] = 1.0f;
            Array.Clear(linearFilter, 0, linearFilter.Length);
            Array.Clear(referenceBuffer, 0, referenceBuffer.Length);
            Array.Clear(nonlinearBuffer, 0, nonlinearBuffer.Length);
            bufferIndex = 0;
        }
    }

    /// <summary>
    /// Wiener-Hammerstein Model - Linear filter -> Nonlinear -> Linear filter
    /// More general than Hammerstein, handles pre and post filtering
    /// </summary>
    public class WienerHammersteinAEC
    {
        private readonly int preFilterLength;
        private readonly int postFilterLength;
        private readonly int polynomialOrder;
        private readonly float stepSize;

        private float[] preFilter;
        private float[] postFilter;
        private float[] polynomialCoeffs;
        private float[] inputBuffer;
        private float[] preFilteredBuffer;
        private float[] nonlinearBuffer;
        private int bufferIndex;

        public string AlgorithmName => "Wiener-Hammerstein Model AEC";

        public WienerHammersteinAEC(int preFilterLength = 128, int postFilterLength = 128,
                                    int polynomialOrder = 3, float stepSize = 0.005f)
        {
            this.preFilterLength = preFilterLength;
            this.postFilterLength = postFilterLength;
            this.polynomialOrder = polynomialOrder;
            this.stepSize = stepSize;

            preFilter = new float[preFilterLength];
            postFilter = new float[postFilterLength];
            polynomialCoeffs = new float[polynomialOrder + 1];
            polynomialCoeffs[1] = 1.0f; // Linear initialization

            int maxLength = Math.Max(preFilterLength, postFilterLength);
            inputBuffer = new float[maxLength];
            preFilteredBuffer = new float[maxLength];
            nonlinearBuffer = new float[maxLength];
            bufferIndex = 0;
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            inputBuffer[bufferIndex] = speakerReference;

            // Pre-filter stage
            float preFiltered = 0;
            for (int i = 0; i < preFilterLength; i++)
            {
                int idx = (bufferIndex - i + inputBuffer.Length) % inputBuffer.Length;
                preFiltered += preFilter[i] * inputBuffer[idx];
            }
            preFilteredBuffer[bufferIndex] = preFiltered;

            // Nonlinear stage
            float nonlinearOutput = ApplyPolynomial(preFiltered);
            nonlinearBuffer[bufferIndex] = nonlinearOutput;

            // Post-filter stage
            float output = 0;
            for (int i = 0; i < postFilterLength; i++)
            {
                int idx = (bufferIndex - i + nonlinearBuffer.Length) % nonlinearBuffer.Length;
                output += postFilter[i] * nonlinearBuffer[idx];
            }

            // Calculate error
            float error = micInput - output;

            // Update all stages
            UpdateAllStages(error);

            bufferIndex = (bufferIndex + 1) % inputBuffer.Length;

            return error;
        }

        private float ApplyPolynomial(float x)
        {
            float result = polynomialCoeffs[0];
            float xPower = x;

            for (int i = 1; i <= polynomialOrder; i++)
            {
                result += polynomialCoeffs[i] * xPower;
                xPower *= x;
            }

            return Math.Max(-2.0f, Math.Min(2.0f, result));
        }

        private void UpdateAllStages(float error)
        {
            // Update post-filter (easiest gradient)
            for (int i = 0; i < postFilterLength; i++)
            {
                int idx = (bufferIndex - i + nonlinearBuffer.Length) % nonlinearBuffer.Length;
                postFilter[i] += stepSize * error * nonlinearBuffer[idx];
            }

            // Update polynomial coefficients
            float polynomialGradient = 0;
            for (int i = 0; i < postFilterLength; i++)
            {
                polynomialGradient += postFilter[i];
            }

            float xPower = 1.0f;
            float preFilteredValue = preFilteredBuffer[bufferIndex];
            for (int i = 0; i <= polynomialOrder; i++)
            {
                polynomialCoeffs[i] += stepSize * error * polynomialGradient * xPower * 0.1f;
                xPower *= preFilteredValue;
            }

            // Update pre-filter (most complex gradient through nonlinearity)
            for (int i = 0; i < preFilterLength; i++)
            {
                int idx = (bufferIndex - i + inputBuffer.Length) % inputBuffer.Length;
                float polynomialDerivative = CalculatePolynomialDerivative(preFilteredBuffer[bufferIndex]);
                float gradient = polynomialGradient * polynomialDerivative * inputBuffer[idx];
                preFilter[i] += stepSize * error * gradient * 0.05f; // Reduced for stability
            }
        }

        private float CalculatePolynomialDerivative(float x)
        {
            float derivative = 0;
            float xPower = 1.0f;

            for (int i = 1; i <= polynomialOrder; i++)
            {
                derivative += i * polynomialCoeffs[i] * xPower;
                xPower *= x;
            }

            return derivative;
        }

        public void Reset()
        {
            Array.Clear(preFilter, 0, preFilter.Length);
            Array.Clear(postFilter, 0, postFilter.Length);
            Array.Clear(polynomialCoeffs, 0, polynomialCoeffs.Length);
            polynomialCoeffs[1] = 1.0f;
            Array.Clear(inputBuffer, 0, inputBuffer.Length);
            Array.Clear(preFilteredBuffer, 0, preFilteredBuffer.Length);
            Array.Clear(nonlinearBuffer, 0, nonlinearBuffer.Length);
            bufferIndex = 0;
        }
    }

    /// <summary>
    /// Neural Network based AEC - Handles complex nonlinearities
    /// Simple feedforward network with backpropagation
    /// </summary>
    public class NeuralNetworkAEC
    {
        private readonly int inputSize;
        private readonly int hiddenSize;
        private readonly int outputSize;
        private readonly float learningRate;

        // Network weights
        private float[,] weightsInputHidden;
        private float[,] weightsHiddenOutput;
        private float[] biasHidden;
        private float[] biasOutput;

        // Activations and buffers
        private float[] inputBuffer;
        private float[] hiddenActivations;
        private float outputActivation;
        private int bufferIndex;

        public string AlgorithmName => "Neural Network AEC";

        public NeuralNetworkAEC(int inputSize = 64, int hiddenSize = 32, float learningRate = 0.001f)
        {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = 1;
            this.learningRate = learningRate;

            // Initialize weights with Xavier initialization
            Random rand = new Random(42);
            float inputScale = (float)Math.Sqrt(2.0 / inputSize);
            float hiddenScale = (float)Math.Sqrt(2.0 / hiddenSize);

            weightsInputHidden = new float[inputSize, hiddenSize];
            weightsHiddenOutput = new float[hiddenSize, outputSize];
            biasHidden = new float[hiddenSize];
            biasOutput = new float[outputSize];

            // Random initialization
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    weightsInputHidden[i, j] = (float)(rand.NextDouble() * 2 - 1) * inputScale;
                }
            }

            for (int i = 0; i < hiddenSize; i++)
            {
                for (int j = 0; j < outputSize; j++)
                {
                    weightsHiddenOutput[i, j] = (float)(rand.NextDouble() * 2 - 1) * hiddenScale;
                }
            }

            inputBuffer = new float[inputSize];
            hiddenActivations = new float[hiddenSize];
            bufferIndex = 0;
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            // Shift input buffer and add new sample
            for (int i = inputSize - 1; i > 0; i--)
            {
                inputBuffer[i] = inputBuffer[i - 1];
            }
            inputBuffer[0] = speakerReference;

            // Forward pass
            outputActivation = ForwardPass(inputBuffer);

            // Calculate error
            float error = micInput - outputActivation;

            // Backward pass (backpropagation)
            BackwardPass(error);

            return error;
        }

        private float ForwardPass(float[] input)
        {
            // Hidden layer
            for (int j = 0; j < hiddenSize; j++)
            {
                float sum = biasHidden[j];
                for (int i = 0; i < inputSize; i++)
                {
                    sum += input[i] * weightsInputHidden[i, j];
                }
                hiddenActivations[j] = ReLU(sum);
            }

            // Output layer
            float output = biasOutput[0];
            for (int i = 0; i < hiddenSize; i++)
            {
                output += hiddenActivations[i] * weightsHiddenOutput[i, 0];
            }

            return Tanh(output); // Bounded output
        }

        private void BackwardPass(float error)
        {
            // Output layer gradients
            float outputGradient = error * TanhDerivative(outputActivation);

            // Update output weights
            for (int i = 0; i < hiddenSize; i++)
            {
                weightsHiddenOutput[i, 0] += learningRate * outputGradient * hiddenActivations[i];
            }
            biasOutput[0] += learningRate * outputGradient;

            // Hidden layer gradients
            float[] hiddenGradients = new float[hiddenSize];
            for (int j = 0; j < hiddenSize; j++)
            {
                float gradient = outputGradient * weightsHiddenOutput[j, 0];
                hiddenGradients[j] = gradient * ReLUDerivative(hiddenActivations[j]);
            }

            // Update hidden weights
            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    weightsInputHidden[i, j] += learningRate * hiddenGradients[j] * inputBuffer[i];
                }
            }

            for (int j = 0; j < hiddenSize; j++)
            {
                biasHidden[j] += learningRate * hiddenGradients[j];
            }
        }

        private float ReLU(float x) => Math.Max(0, x);
        private float ReLUDerivative(float x) => x > 0 ? 1 : 0;
        private float Tanh(float x) => (float)Math.Tanh(x);
        private float TanhDerivative(float x) => 1 - x * x;

        public void Reset()
        {
            Array.Clear(inputBuffer, 0, inputBuffer.Length);
            Array.Clear(hiddenActivations, 0, hiddenActivations.Length);
            outputActivation = 0;
            bufferIndex = 0;
        }
    }

    /// <summary>
    /// Functional Link Adaptive Filter (FLAF)
    /// Expands input space with nonlinear basis functions
    /// </summary>
    public class FunctionalLinkAEC
    {
        private readonly int memoryLength;
        private readonly int expansionOrder;
        private readonly float stepSize;

        private float[] weights;
        private float[] referenceBuffer;
        private float[] expandedFeatures;
        private int bufferIndex;

        public string AlgorithmName => "Functional Link Adaptive Filter";

        public FunctionalLinkAEC(int memoryLength = 64, int expansionOrder = 3, float stepSize = 0.01f)
        {
            this.memoryLength = memoryLength;
            this.expansionOrder = expansionOrder;
            this.stepSize = stepSize;

            // Calculate expanded feature size
            int expandedSize = memoryLength * (1 + expansionOrder * 3); // Linear + sin/cos + powers
            weights = new float[expandedSize];
            expandedFeatures = new float[expandedSize];
            referenceBuffer = new float[memoryLength];
            bufferIndex = 0;
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            referenceBuffer[bufferIndex] = speakerReference;

            // Generate expanded features
            GenerateExpandedFeatures();

            // Calculate output
            float output = 0;
            for (int i = 0; i < expandedFeatures.Length; i++)
            {
                output += weights[i] * expandedFeatures[i];
            }

            // Calculate error
            float error = micInput - output;

            // Update weights
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += stepSize * error * expandedFeatures[i];
            }

            bufferIndex = (bufferIndex + 1) % memoryLength;

            return error;
        }

        private void GenerateExpandedFeatures()
        {
            int featureIndex = 0;

            for (int i = 0; i < memoryLength; i++)
            {
                int idx = (bufferIndex - i + memoryLength) % memoryLength;
                float x = referenceBuffer[idx];

                // Linear term
                expandedFeatures[featureIndex++] = x;

                // Trigonometric expansions
                expandedFeatures[featureIndex++] = (float)Math.Sin(Math.PI * x);
                expandedFeatures[featureIndex++] = (float)Math.Cos(Math.PI * x);

                // Polynomial expansions
                for (int order = 2; order <= expansionOrder; order++)
                {
                    expandedFeatures[featureIndex++] = (float)Math.Pow(x, order);
                }

                // Cross-terms (simplified - just with previous sample)
                if (i > 0)
                {
                    int prevIdx = (bufferIndex - i + 1 + memoryLength) % memoryLength;
                    expandedFeatures[featureIndex++] = x * referenceBuffer[prevIdx];
                    expandedFeatures[featureIndex++] = x * x * referenceBuffer[prevIdx];
                }
            }
        }

        public void Reset()
        {
            Array.Clear(weights, 0, weights.Length);
            Array.Clear(referenceBuffer, 0, referenceBuffer.Length);
            Array.Clear(expandedFeatures, 0, expandedFeatures.Length);
            bufferIndex = 0;
        }
    }

    /// <summary>
    /// Kernel Adaptive Filter - Nonlinear processing in kernel space
    /// Uses Gaussian kernel for implicit nonlinear mapping
    /// </summary>
    public class KernelAdaptiveFilter
    {
        private readonly int dictionarySize;
        private readonly float kernelWidth;
        private readonly float stepSize;
        private readonly float regularization;

        private List<float[]> dictionary;
        private List<float> weights;
        private float[] currentInput;
        private readonly int inputDimension;

        public string AlgorithmName => "Kernel Adaptive Filter (KLMS)";

        public KernelAdaptiveFilter(int inputDimension = 32, int dictionarySize = 100,
                                   float kernelWidth = 1.0f, float stepSize = 0.1f)
        {
            this.inputDimension = inputDimension;
            this.dictionarySize = dictionarySize;
            this.kernelWidth = kernelWidth;
            this.stepSize = stepSize;
            this.regularization = 0.001f;

            dictionary = new List<float[]>();
            weights = new List<float>();
            currentInput = new float[inputDimension];
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            // Shift input vector
            for (int i = inputDimension - 1; i > 0; i--)
            {
                currentInput[i] = currentInput[i - 1];
            }
            currentInput[0] = speakerReference;

            // Calculate output using kernel expansion
            float output = 0;
            for (int i = 0; i < dictionary.Count; i++)
            {
                float kernel = GaussianKernel(currentInput, dictionary[i]);
                output += weights[i] * kernel;
            }

            // Calculate error
            float error = micInput - output;

            // Update weights
            for (int i = 0; i < weights.Count; i++)
            {
                float kernel = GaussianKernel(currentInput, dictionary[i]);
                weights[i] += stepSize * error * kernel;
            }

            // Add to dictionary if needed (novelty criterion)
            if (ShouldAddToDictionary(currentInput))
            {
                AddToDictionary(currentInput.ToArray(), stepSize * error);
            }

            return error;
        }

        private float GaussianKernel(float[] x1, float[] x2)
        {
            float squaredDistance = 0;
            for (int i = 0; i < inputDimension; i++)
            {
                float diff = x1[i] - x2[i];
                squaredDistance += diff * diff;
            }
            return (float)Math.Exp(-squaredDistance / (2 * kernelWidth * kernelWidth));
        }

        private bool ShouldAddToDictionary(float[] input)
        {
            if (dictionary.Count >= dictionarySize)
                return false;

            // Novelty criterion - add if sufficiently different from existing entries
            foreach (var entry in dictionary)
            {
                if (GaussianKernel(input, entry) > 0.95f) // Similarity threshold
                    return false;
            }

            return true;
        }

        private void AddToDictionary(float[] input, float weight)
        {
            dictionary.Add(input);
            weights.Add(weight);

            // Prune if exceeding size limit
            if (dictionary.Count > dictionarySize)
            {
                // Remove least significant entry
                int minIdx = 0;
                float minWeight = Math.Abs(weights[0]);
                for (int i = 1; i < weights.Count; i++)
                {
                    if (Math.Abs(weights[i]) < minWeight)
                    {
                        minWeight = Math.Abs(weights[i]);
                        minIdx = i;
                    }
                }
                dictionary.RemoveAt(minIdx);
                weights.RemoveAt(minIdx);
            }
        }

        public void Reset()
        {
            dictionary.Clear();
            weights.Clear();
            Array.Clear(currentInput, 0, currentInput.Length);
        }
    }

    /// <summary>
    /// Adaptive Nonlinear Processor with multiple algorithms
    /// </summary>
    public class NonlinearAECProcessor
    {
        public enum Algorithm
        {
            Volterra,
            Hammerstein,
            WienerHammerstein,
            NeuralNetwork,
            FunctionalLink,
            KernelAdaptive
        }

        private VolterraFilter volterra;
        private HammersteinAEC hammerstein;
        private WienerHammersteinAEC wienerHammerstein;
        private NeuralNetworkAEC neuralNetwork;
        private FunctionalLinkAEC functionalLink;
        private KernelAdaptiveFilter kernelFilter;

        private Algorithm currentAlgorithm;
        private object currentFilter;

        public NonlinearAECProcessor()
        {
            InitializeAlgorithms();
            SetAlgorithm(Algorithm.Hammerstein); // Default
        }

        private void InitializeAlgorithms()
        {
            volterra = new VolterraFilter();
            hammerstein = new HammersteinAEC();
            wienerHammerstein = new WienerHammersteinAEC();
            neuralNetwork = new NeuralNetworkAEC();
            functionalLink = new FunctionalLinkAEC();
            kernelFilter = new KernelAdaptiveFilter();
        }

        public void SetAlgorithm(Algorithm algo)
        {
            currentAlgorithm = algo;
            currentFilter = algo switch
            {
                Algorithm.Volterra => volterra,
                Algorithm.Hammerstein => hammerstein,
                Algorithm.WienerHammerstein => wienerHammerstein,
                Algorithm.NeuralNetwork => neuralNetwork,
                Algorithm.FunctionalLink => functionalLink,
                Algorithm.KernelAdaptive => kernelFilter,
                _ => hammerstein
            };

            Console.WriteLine($"Switched to {GetAlgorithmName()}");
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            return currentAlgorithm switch
            {
                Algorithm.Volterra => volterra.ProcessSample(micInput, speakerReference),
                Algorithm.Hammerstein => hammerstein.ProcessSample(micInput, speakerReference),
                Algorithm.WienerHammerstein => wienerHammerstein.ProcessSample(micInput, speakerReference),
                Algorithm.NeuralNetwork => neuralNetwork.ProcessSample(micInput, speakerReference),
                Algorithm.FunctionalLink => functionalLink.ProcessSample(micInput, speakerReference),
                Algorithm.KernelAdaptive => kernelFilter.ProcessSample(micInput, speakerReference),
                _ => micInput
            };
        }

        public string GetAlgorithmName()
        {
            return currentAlgorithm switch
            {
                Algorithm.Volterra => volterra.AlgorithmName,
                Algorithm.Hammerstein => hammerstein.AlgorithmName,
                Algorithm.WienerHammerstein => wienerHammerstein.AlgorithmName,
                Algorithm.NeuralNetwork => neuralNetwork.AlgorithmName,
                Algorithm.FunctionalLink => functionalLink.AlgorithmName,
                Algorithm.KernelAdaptive => kernelFilter.AlgorithmName,
                _ => "Unknown"
            };
        }

        public void Reset()
        {
            volterra?.Reset();
            hammerstein?.Reset();
            wienerHammerstein?.Reset();
            neuralNetwork?.Reset();
            functionalLink?.Reset();
            kernelFilter?.Reset();
        }
    }

    /// <summary>
    /// Cascade of linear and nonlinear filters for robust echo cancellation
    /// </summary>
    public class CascadeNonlinearAEC
    {
        private readonly NLMSLinearStage linearStage;
        private readonly HammersteinAEC nonlinearStage;
        private readonly ResidualSuppressor residualSuppressor;
        private readonly float crossoverThreshold;

        public string AlgorithmName => "Cascade Linear-Nonlinear AEC";

        public CascadeNonlinearAEC(float crossoverThreshold = 0.1f)
        {
            this.crossoverThreshold = crossoverThreshold;
            linearStage = new NLMSLinearStage(512, 0.5f);
            nonlinearStage = new HammersteinAEC(256, 5);
            residualSuppressor = new ResidualSuppressor();
        }

        public float ProcessSample(float micInput, float speakerReference)
        {
            // First stage: Linear echo cancellation
            float linearOutput = linearStage.ProcessSample(micInput, speakerReference);

            // Check if nonlinear processing is needed
            float linearError = Math.Abs(linearOutput);

            float output;
            if (linearError > crossoverThreshold)
            {
                // Second stage: Nonlinear echo cancellation on residual
                output = nonlinearStage.ProcessSample(linearOutput, speakerReference);
            }
            else
            {
                output = linearOutput;
            }

            // Third stage: Residual echo suppression
            output = residualSuppressor.Process(output, speakerReference);

            return output;
        }

        public void Reset()
        {
            linearStage.Reset();
            nonlinearStage.Reset();
            residualSuppressor.Reset();
        }

        // Simple NLMS for linear stage
        private class NLMSLinearStage
        {
            private readonly float[] weights;
            private readonly float[] buffer;
            private readonly float stepSize;
            private int index;

            public NLMSLinearStage(int length, float stepSize)
            {
                weights = new float[length];
                buffer = new float[length];
                this.stepSize = stepSize;
            }

            public float ProcessSample(float mic, float reference)
            {
                buffer[index] = reference;

                float estimate = 0;
                float power = 0.001f;

                for (int i = 0; i < weights.Length; i++)
                {
                    int idx = (index - i + buffer.Length) % buffer.Length;
                    estimate += weights[i] * buffer[idx];
                    power += buffer[idx] * buffer[idx];
                }

                float error = mic - estimate;
                float normalizedStep = stepSize / power;

                for (int i = 0; i < weights.Length; i++)
                {
                    int idx = (index - i + buffer.Length) % buffer.Length;
                    weights[i] += normalizedStep * error * buffer[idx];
                }

                index = (index + 1) % buffer.Length;
                return error;
            }

            public void Reset()
            {
                Array.Clear(weights, 0, weights.Length);
                Array.Clear(buffer, 0, buffer.Length);
                index = 0;
            }
        }

        // Spectral residual suppressor
        private class ResidualSuppressor
        {
            private readonly float[] history;
            private readonly float suppressionFactor;
            private int historyIndex;

            public ResidualSuppressor(int historySize = 100, float suppressionFactor = 0.5f)
            {
                history = new float[historySize];
                this.suppressionFactor = suppressionFactor;
            }

            public float Process(float input, float reference)
            {
                history[historyIndex] = Math.Abs(input);
                historyIndex = (historyIndex + 1) % history.Length;

                float avgMagnitude = history.Average();
                float refMagnitude = Math.Abs(reference);

                // Suppress if residual correlates with reference
                if (refMagnitude > 0.1f && avgMagnitude > 0.05f)
                {
                    float correlation = Math.Min(1.0f, avgMagnitude / refMagnitude);
                    return input * (1 - suppressionFactor * correlation);
                }

                return input;
            }

            public void Reset()
            {
                Array.Clear(history, 0, history.Length);
                historyIndex = 0;
            }
        }
    }

    /// <summary>
    /// Performance metrics for nonlinear AEC evaluation
    /// </summary>
    public class NonlinearAECMetrics
    {
        private readonly int windowSize;
        private readonly Queue<float> inputSamples;
        private readonly Queue<float> outputSamples;
        private readonly Queue<float> referenceSamples;

        public float THD { get; private set; } // Total Harmonic Distortion
        public float ERLE { get; private set; } // Echo Return Loss Enhancement
        public float NonlinearityIndex { get; private set; }
        public float ConvergenceRate { get; private set; }

        public NonlinearAECMetrics(int windowSize = 2048)
        {
            this.windowSize = windowSize;
            inputSamples = new Queue<float>(windowSize);
            outputSamples = new Queue<float>(windowSize);
            referenceSamples = new Queue<float>(windowSize);
        }

        public void Update(float input, float output, float reference)
        {
            // Update buffers
            if (inputSamples.Count >= windowSize)
            {
                inputSamples.Dequeue();
                outputSamples.Dequeue();
                referenceSamples.Dequeue();
            }

            inputSamples.Enqueue(input);
            outputSamples.Enqueue(output);
            referenceSamples.Enqueue(reference);

            if (inputSamples.Count >= windowSize)
            {
                CalculateMetrics();
            }
        }

        private void CalculateMetrics()
        {
            // Calculate ERLE
            float inputPower = inputSamples.Select(x => x * x).Average();
            float outputPower = outputSamples.Select(x => x * x).Average();

            if (outputPower > 0 && inputPower > 0)
            {
                ERLE = 10 * (float)Math.Log10(inputPower / outputPower);
            }

            // Calculate THD using FFT
            Complex[] fftInput = new Complex[windowSize];
            Complex[] fftOutput = new Complex[windowSize];

            var inputArray = inputSamples.ToArray();
            var outputArray = outputSamples.ToArray();

            for (int i = 0; i < windowSize; i++)
            {
                fftInput[i] = new Complex(inputArray[i], 0);
                fftOutput[i] = new Complex(outputArray[i], 0);
            }

            FFT(fftInput);
            FFT(fftOutput);

            // Find fundamental frequency
            int fundamentalBin = FindPeakBin(fftInput);
            float fundamentalPower = (float)fftOutput[fundamentalBin].Magnitude;

            // Calculate harmonic powers
            float harmonicPower = 0;
            for (int harmonic = 2; harmonic <= 5; harmonic++)
            {
                int bin = fundamentalBin * harmonic;
                if (bin < windowSize / 2)
                {
                    harmonicPower += (float)Math.Pow(fftOutput[bin].Magnitude, 2);
                }
            }

            THD = harmonicPower > 0 ?
                  100 * (float)Math.Sqrt(harmonicPower) / fundamentalPower : 0;

            // Calculate nonlinearity index
            CalculateNonlinearityIndex();

            // Estimate convergence rate
            EstimateConvergenceRate();
        }

        private void CalculateNonlinearityIndex()
        {
            var input = inputSamples.ToArray();
            var output = outputSamples.ToArray();

            // Calculate correlation for different polynomial orders
            float linearCorr = CalculateCorrelation(input, output, 1);
            float quadraticCorr = CalculateCorrelation(input, output, 2);
            float cubicCorr = CalculateCorrelation(input, output, 3);

            // Nonlinearity index based on higher-order correlation
            NonlinearityIndex = (quadraticCorr + cubicCorr) / (2 * Math.Max(linearCorr, 0.001f));
        }

        private float CalculateCorrelation(float[] x, float[] y, int order)
        {
            float[] xPower = new float[x.Length];
            for (int i = 0; i < x.Length; i++)
            {
                xPower[i] = (float)Math.Pow(x[i], order);
            }

            float meanX = xPower.Average();
            float meanY = y.Average();

            float covariance = 0;
            float varX = 0;
            float varY = 0;

            for (int i = 0; i < x.Length; i++)
            {
                float dx = xPower[i] - meanX;
                float dy = y[i] - meanY;
                covariance += dx * dy;
                varX += dx * dx;
                varY += dy * dy;
            }

            return (float)(covariance / Math.Sqrt(varX * varY + 1e-10));
        }

        private void EstimateConvergenceRate()
        {
            var errors = outputSamples.ToArray();
            int halfWindow = windowSize / 2;

            float firstHalfPower = errors.Take(halfWindow).Select(x => x * x).Average();
            float secondHalfPower = errors.Skip(halfWindow).Select(x => x * x).Average();

            if (firstHalfPower > 0)
            {
                ConvergenceRate = (firstHalfPower - secondHalfPower) / firstHalfPower;
            }
        }

        private void FFT(Complex[] data)
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
                    Complex temp = data[i];
                    data[i] = data[j];
                    data[j] = temp;
                }
            }

            // FFT
            for (int len = 2; len <= n; len <<= 1)
            {
                double angle = -2 * Math.PI / len;
                Complex wlen = new Complex(Math.Cos(angle), Math.Sin(angle));

                for (int i = 0; i < n; i += len)
                {
                    Complex w = Complex.One;
                    for (int k = 0; k < len / 2; k++)
                    {
                        Complex u = data[i + k];
                        Complex v = data[i + k + len / 2] * w;
                        data[i + k] = u + v;
                        data[i + k + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
        }

        private int FindPeakBin(Complex[] fft)
        {
            int peakBin = 0;
            double maxMagnitude = 0;

            for (int i = 1; i < fft.Length / 2; i++)
            {
                double magnitude = fft[i].Magnitude;
                if (magnitude > maxMagnitude)
                {
                    maxMagnitude = magnitude;
                    peakBin = i;
                }
            }

            return peakBin;
        }

        public void PrintMetrics()
        {
            Console.WriteLine("=== Nonlinear AEC Metrics ===");
            Console.WriteLine($"ERLE: {ERLE:F2} dB");
            Console.WriteLine($"THD: {THD:F2}%");
            Console.WriteLine($"Nonlinearity Index: {NonlinearityIndex:F3}");
            Console.WriteLine($"Convergence Rate: {ConvergenceRate:F3}");
        }
    }

    /// <summary>
    /// Real-time nonlinear AEC application
    /// </summary>
    public class RealTimeNonlinearAEC
    {
        private WaveInEvent waveIn;
        private WaveOutEvent waveOut;
        private BufferedWaveProvider outputBuffer;
        private NonlinearAECProcessor processor;
        private NonlinearAECMetrics metrics;
        private AudioFileReader testSignal;
        private float[] referenceBuffer;
        private int referenceIndex;
        private bool isProcessing;
        private System.Timers.Timer metricsTimer;

        public void Start(string testAudioPath = null)
        {
            processor = new NonlinearAECProcessor();
            metrics = new NonlinearAECMetrics();

            // Setup audio
            waveIn = new WaveInEvent
            {
                WaveFormat = new WaveFormat(16000, 16, 1),
                BufferMilliseconds = 10,
                DeviceNumber = 1
            };

            outputBuffer = new BufferedWaveProvider(waveIn.WaveFormat)
            {
                BufferLength = waveIn.WaveFormat.SampleRate,
                DiscardOnBufferOverflow = true
            };

            waveOut = new WaveOutEvent { DesiredLatency = 50 };

            // Load test signal if provided
            if (!string.IsNullOrEmpty(testAudioPath))
            {
                testSignal = new AudioFileReader(testAudioPath);
                referenceBuffer = new float[testSignal.Length / sizeof(float)];
                testSignal.Read(referenceBuffer, 0, referenceBuffer.Length);

                // Add nonlinear distortion to simulate speaker nonlinearity
                for (int i = 0; i < referenceBuffer.Length; i++)
                {
                    float x = referenceBuffer[i];
                    referenceBuffer[i] = x + 0.2f * x * x + 0.1f * x * x * x; // Add harmonics
                    referenceBuffer[i] = Math.Max(-1, Math.Min(1, referenceBuffer[i])); // Clip
                }
            }

            waveIn.DataAvailable += OnDataAvailable;

            // Setup metrics display timer
            metricsTimer = new System.Timers.Timer(2000); // Every 2 seconds
            metricsTimer.Elapsed += (s, e) => metrics.PrintMetrics();
            metricsTimer.Start();

            waveOut.Init(outputBuffer);
            waveOut.Play();
            waveIn.StartRecording();

            isProcessing = true;

            Console.WriteLine("Nonlinear AEC Started");
            Console.WriteLine("Commands:");
            Console.WriteLine("1: Volterra Filter");
            Console.WriteLine("2: Hammerstein Model");
            Console.WriteLine("3: Wiener-Hammerstein Model");
            Console.WriteLine("4: Neural Network");
            Console.WriteLine("5: Functional Link");
            Console.WriteLine("6: Kernel Adaptive Filter");
            Console.WriteLine("Q: Quit\n");

            HandleUserInput();
        }

        private void OnDataAvailable(object sender, WaveInEventArgs e)
        {
            if (!isProcessing) return;

            float[] samples = new float[e.BytesRecorded / 2];
            for (int i = 0; i < samples.Length; i++)
            {
                short sample = BitConverter.ToInt16(e.Buffer, i * 2);
                samples[i] = sample / 32768f;
            }

            float[] processed = new float[samples.Length];
            for (int i = 0; i < samples.Length; i++)
            {
                float reference = GetReferenceSample();
                processed[i] = processor.ProcessSample(samples[i], reference);
                metrics.Update(samples[i], processed[i], reference);
            }

            byte[] outputBytes = new byte[processed.Length * 2];
            for (int i = 0; i < processed.Length; i++)
            {
                short sample = (short)(processed[i] * 32767f);
                BitConverter.GetBytes(sample).CopyTo(outputBytes, i * 2);
            }

            outputBuffer.AddSamples(e.Buffer, 0, e.Buffer.Length);
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

        private void HandleUserInput()
        {
            Task.Run(() =>
            {
                while (isProcessing)
                {
                    var key = Console.ReadKey(true).KeyChar;
                    switch (key)
                    {
                        case '1':
                            processor.SetAlgorithm(NonlinearAECProcessor.Algorithm.Volterra);
                            break;
                        case '2':
                            processor.SetAlgorithm(NonlinearAECProcessor.Algorithm.Hammerstein);
                            break;
                        case '3':
                            processor.SetAlgorithm(NonlinearAECProcessor.Algorithm.WienerHammerstein);
                            break;
                        case '4':
                            processor.SetAlgorithm(NonlinearAECProcessor.Algorithm.NeuralNetwork);
                            break;
                        case '5':
                            processor.SetAlgorithm(NonlinearAECProcessor.Algorithm.FunctionalLink);
                            break;
                        case '6':
                            processor.SetAlgorithm(NonlinearAECProcessor.Algorithm.KernelAdaptive);
                            break;
                        case 'q':
                        case 'Q':
                            Stop();
                            break;
                    }
                }
            });
        }

        public void Stop()
        {
            isProcessing = false;

            metricsTimer?.Stop();
            metricsTimer?.Dispose();

            waveIn?.StopRecording();
            waveIn?.Dispose();

            waveOut?.Stop();
            waveOut?.Dispose();

            testSignal?.Dispose();

            Console.WriteLine("Nonlinear AEC Stopped");
        }
    }

    /// <summary>
    /// Main program
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Nonlinear AEC Implementation");
            Console.WriteLine("============================\n");

            var app = new RealTimeNonlinearAEC();

            Console.Write("Enter test audio file (or press Enter for mic only): ");
            string audioPath = Console.ReadLine();

            app.Start(string.IsNullOrWhiteSpace(audioPath) ? null : audioPath);

            Console.ReadLine(); // Wait for user to quit
            app.Stop();
        }
    }
}
