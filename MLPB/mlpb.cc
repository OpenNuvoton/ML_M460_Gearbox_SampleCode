/*
 * Copyright (C) 2021 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Description: Keyword spotting example code using MFCC feature extraction
 * and neural network.
 */

#include "mlpb.h"
#include <float.h>
#include <arm_math.h>


MLPB::MLPB(float *ptrioBuffer, uint32_t nElements)
{
    if (this->_InitModel())
    {
        numFeatures = nElements;

        /* this->inputBuffer = std::move(std::vector<float>(
                                             ptrioBuffer,
                                             ptrioBuffer + nElements));
          */
        ptrInputBuffer = ptrioBuffer;

        this->InitMlpb();
    }
}

void MLPB::ExtractFeatures()
{
    // for feature process step, for some application don't need.
    this->inputBuffer = std::move(std::vector<float>(
                                      ptrInputBuffer,
                                      ptrInputBuffer + numFeatures));

    //for(int i = 0; i < 16; i++){
    //
    //   printf("%d: %f\r\n",i, this->inputBuffer[i]);
    //
    //}
}

void MLPB::CalculateGearBoxWindow(const uint16_t FeatureElements, const uint16_t win_size, const float max_val, const float min_val)
{

    const uint16_t SensorNumber =     FeatureElements / win_size;


    std::vector<float> row(300, 0);
    std::vector<std::vector<float>> vecBuffer(4, row) ;

    for (int i = 0; i < FeatureElements; i++)
    {
        vecBuffer[i % SensorNumber][i / SensorNumber] = this->inputBuffer[i];
        //printf("%d",v[i][j]);

    }

    const uint8_t featureType = 4;
    featureBuffer.resize(SensorNumber * featureType);  //allocate size

    for (int i = 0; i < SensorNumber; i++)
    {

        int tmpIdx = i;
        featureBuffer[tmpIdx] = absSum(&vecBuffer[i][0], win_size);

        tmpIdx = 1 * SensorNumber + i;
        arm_std_f32(vecBuffer[i].data(), win_size, &featureBuffer[tmpIdx]);

        tmpIdx = 2 * SensorNumber + i;
        uint32_t  pIdx;
        arm_max_f32(vecBuffer[i].data(), win_size, &featureBuffer[tmpIdx], &pIdx);

        tmpIdx = 3 * SensorNumber + i;
        featureBuffer[tmpIdx] = kurtosis(&vecBuffer[i][0], win_size);

    }

#ifndef DNN
    normalize(&featureBuffer[0], featureBuffer.size(), max_val, min_val);
#endif

    //debug
    //for(int i = 0; i < featureBuffer.size(); i++){
    //
    //   printf("%d: %f\r\n",i, featureBuffer[i]);
    //
    //}
}

void MLPB::InitMlpb()
{
    if (!model->IsInited())
    {
        printf("Warning: model has not been initialised\r\n");
        model->Init();
    }


    numOutClasses = model->GetOutputShape()->data[1];  // Output shape should be [1, numOutClasses].

    // Following are for debug purposes.
    printf("Initialising network anomaly..\r\n");
    printf("numOutDim: %d\r\n", numFeatures);

    //mfcc =  std::unique_ptr<MFCC>(new MFCC(numMfccFeatures, frameLen));
#ifdef RAW_DATA
    numFeaturesModelIn = 16;
    output = std::vector<float>(numFeaturesModelIn, 0.0);
#else
    output = std::vector<float>(numFeatures, 0.0);
#endif
}

void MLPB::Classify()
{
    // Copy the features data into the TfLite input tensor.
    float *inTensorData = tflite::GetTensorData<float>(model->GetInputTensor());
#ifdef RAW_DATA
    memcpy(inTensorData, this->featureBuffer.data(), numFeaturesModelIn * sizeof(float));  //float
#else
    memcpy(inTensorData, this->inputBuffer.data(), numFeatures * sizeof(float));  //float
#endif

    // Run inference on this data.
    model->RunInference();

    // Get output from the TfLite tensor.
    float *outTensorData = tflite::GetTensorData<float>(model->GetOutputTensor());
#ifdef RAW_DATA
    memcpy(output.data(), outTensorData, numOutClasses * sizeof(float));  //float
#else
    memcpy(output.data(), outTensorData, numFeatures * sizeof(float));  //float
#endif
}

int MLPB::GetDnnResult()
{
    float32_t  pResult;
    uint32_t   pIdx;
    arm_max_f32(output.data(), numOutClasses, &pResult, &pIdx);
    return pIdx;
}

void MLPB::EncoderDecoder()
{
    // Copy the features data into the TfLite input tensor.
    float *inTensorData = tflite::GetTensorData<float>(model->GetInputTensor());

#ifdef RAW_DATA
    memcpy(inTensorData, this->featureBuffer.data(), numFeaturesModelIn * sizeof(float));  //float
#else
    memcpy(inTensorData, this->inputBuffer.data(), numFeatures * sizeof(float));  //float
#endif

    // Run inference on this data.
    model->RunInference();

    // Get output from the TfLite tensor.
    float *outTensorData = tflite::GetTensorData<float>(model->GetOutputTensor());
#ifdef RAW_DATA
    memcpy(output.data(), outTensorData, numFeaturesModelIn * sizeof(float));  //float
#else
    memcpy(output.data(), outTensorData, numFeatures * sizeof(float));  //float
#endif
}

int MLPB::GetMaeResult(const float Threshold)
{
    std::vector<float> result;
    float mae = 0;
    int normal = 0;

#ifdef RAW_DATA
    std::transform(output.begin(), output.end(), featureBuffer.begin(), std::back_inserter(result), std::minus<float>());
#else
    std::transform(output.begin(), output.end(), inputBuffer.begin(), std::back_inserter(result), std::minus<float>());
#endif

    //  for(int i = 0; i < output.size(); i++){
    //
    //     printf("output  %d: %f\r\n",i, output[i]);
    //
    //  }
    //
    //  for(int i = 0; i < result.size(); i++){
    //
    //     printf("result  %d: %f\r\n",i, result[i]);
    //
    //  }

    for (uint8_t i = 0; i < result.size(); i++)
    {
        result[i] = abs(result[i]);
        //printf("absValue: %f\r\n", result[i]);
        mae += result[i];
    }

    mae /= result.size();
    printf("mean average error: %f\n", mae);
    normal = (mae < Threshold) ? 1 : 0;


    return normal;
}

int MLPB::GetTopClass(const std::vector<float> &prediction)
{
    int maxInd = 0;
    float maxVal = FLT_MIN;

    for (int i = 0; i < numOutClasses; i++)
    {
        if (maxVal < prediction[i])
        {
            maxVal = prediction[i];
            maxInd = i;
        }
    }

    return maxInd;
}

/*
The statistic computed here is the adjusted Fisher-Pearson standardized
moment coefficient G2, computed directly from the second and fourth
central moment.
*/
float kurtosis(float *datavec, uint16_t len)
{

    int i;
    float mean_x = 0;
    float adj = 0;
    float numerator = 0;
    float denominator = 0;

    arm_mean_f32(datavec, len, &mean_x);

    // Calculate sum((x-mean(x))^4) and sum((x-mean(x))^2)
    double r = 0;
    double q = 0;

    for (i = 0; i < len; i++)
    {
        r += pow(datavec[i] - mean_x,  4);
        q += pow(datavec[i] - mean_x,  2);
    }

    adj = 3 * pow((len - 1), 2) / ((len - 2) * (len - 3));
    numerator = len * (len + 1) * (len - 1) * r;
    denominator = (len - 2) * (len - 3) * pow(q, 2);


    float k = (numerator / denominator) - adj;
    return k;
}

float absSum(float *datavec, uint16_t len)
{

    int i;
    float sum = 0;
    float bufvec[len];

    arm_abs_f32(datavec, bufvec, len);

    for (i = 0; i < len; i++)
    {
        sum += bufvec[i];
    }

    return sum;

}

void normalize(float *featurevec, int len, float max, float min)
{

    for (int i = 0; i < len; i++)
    {
        featurevec[i] = (featurevec[i] - min) / (max - min);
    }
}

