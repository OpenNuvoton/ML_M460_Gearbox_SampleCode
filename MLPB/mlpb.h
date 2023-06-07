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

#ifndef __MLPB_H__
#define __MLPB_H__

#define RAW_DATA
#define DNN

#include <vector>
#include <arm_math.h>

#include "Model.h"

class MLPB
{

public:
    MLPB(float *ptrAudioBuffer, uint32_t nElements);

    ~MLPB() = default;

    void ExtractFeatures();
    void Classify();
    int GetDnnResult();
    void EncoderDecoder();
    int  GetMaeResult(const float Threshold);
    int  GetTopClass(const std::vector<float> &prediction);
    void CalculateGearBoxWindow(const uint16_t, const uint16_t, const float, const float);

    //std::vector<int16_t> inputBuffer;
    float *ptrInputBuffer;
    std::vector<float> inputBuffer;
    std::vector<float> featureBuffer;
    std::vector<float> output;
    //std::vector<float> transFetBuffer;

    int numOutClasses;
    int numFeatures;

    int numFeaturesModelIn;

protected:
    /** @brief Initialises the model */
    bool _InitModel();
    void InitMlpb();

    Model *model;
    //DnnModel* dnnmodel;
};


/**
 * @brief   Gets the pointer to the model data
 * @return  a uint8_t pointer
 **/
float kurtosis(float *datavec, uint16_t len);

float absSum(float *datavec, uint16_t len);

void normalize(float *datavec, int len, float max, float min);



#endif /* __MLPB_H__ */
