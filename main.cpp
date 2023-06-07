/**************************************************************************//**
 * @file     main.c
 * @version  V3.00
 * @brief    This is an tflu_gearbox_anomaly.
 *
 * @copyright SPDX-License-Identifier: Apache-2.0
 * @copyright Copyright (C) 2021 Nuvoton Technology Corp. All rights reserved.
 ******************************************************************************/

#include <stdio.h>
#include <string.h>
#include "NuMicro.h"
#include "config.h"
#include <vector>
using namespace std;

//#include <arm_math.h>
#include "MLPB/mlpb.h"
#ifdef RAW_DATA
    #include "gearbox_raw_test_data.h"
#else
    #include "gearbox_test_data.h"
#endif
#include "BufAttributes.h"


//MLPB
MLPB *mlpb;

void SYS_Init(void);
void I2C2_Init(void);



void SYS_Init(void)
{
    /* Set PF multi-function pins for XT1_OUT(PF.2) and XT1_IN(PF.3) */
    SET_XT1_OUT_PF2();
    SET_XT1_IN_PF3();

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init System Clock                                                                                       */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Enable HIRC and HXT clock */
    CLK_EnableXtalRC(CLK_PWRCTL_HIRCEN_Msk | CLK_PWRCTL_HXTEN_Msk);

    /* Wait for HIRC and HXT clock ready */
    CLK_WaitClockReady(CLK_STATUS_HIRCSTB_Msk | CLK_STATUS_HXTSTB_Msk);

    /* Set PCLK0 and PCLK1 to HCLK/2 */
    CLK->PCLKDIV = (CLK_PCLKDIV_APB0DIV_DIV2 | CLK_PCLKDIV_APB1DIV_DIV2);

    /* Set core clock to 200MHz */
    CLK_SetCoreClock(FREQ_200MHZ);

    /* Enable all GPIO clock */
    CLK->AHBCLK0 |= CLK_AHBCLK0_GPACKEN_Msk | CLK_AHBCLK0_GPBCKEN_Msk | CLK_AHBCLK0_GPCCKEN_Msk | CLK_AHBCLK0_GPDCKEN_Msk |
                    CLK_AHBCLK0_GPECKEN_Msk | CLK_AHBCLK0_GPFCKEN_Msk | CLK_AHBCLK0_GPGCKEN_Msk | CLK_AHBCLK0_GPHCKEN_Msk;
    CLK->AHBCLK1 |= CLK_AHBCLK1_GPICKEN_Msk | CLK_AHBCLK1_GPJCKEN_Msk;

    /* Enable UART0 module clock */
    CLK_EnableModuleClock(UART0_MODULE);

    /* Select UART0 module clock source as HIRC and UART0 module clock divider as 1 */
    CLK_SetModuleClock(UART0_MODULE, CLK_CLKSEL1_UART0SEL_HIRC, CLK_CLKDIV0_UART0(1));

    /* Enable I2S0 module clock */
    CLK_EnableModuleClock(I2S0_MODULE);

    /* Enable I2C2 module clock */
    CLK_EnableModuleClock(I2C2_MODULE);

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init I/O Multi-function                                                                                 */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Set multi-function pins for UART0 RXD and TXD */
    SET_UART0_RXD_PB12();
    SET_UART0_TXD_PB13();

    /* Set multi-function pins for I2S0 */
    SET_I2S0_BCLK_PI6();
    SET_I2S0_MCLK_PI7();
    SET_I2S0_DI_PI8();
    SET_I2S0_DO_PI9();
    SET_I2S0_LRCK_PI10();

    /* Enable I2S0 clock pin (PI6) schmitt trigger */
    PI->SMTEN |= GPIO_SMTEN_SMTEN6_Msk;

    /* Set I2C2 multi-function pins */
    SET_I2C2_SDA_PD0();
    SET_I2C2_SCL_PD1();

    /* Enable I2C2 clock pin (PD1) schmitt trigger */
    PD->SMTEN |= GPIO_SMTEN_SMTEN1_Msk;
}

/* Init I2C interface */
void I2C2_Init(void)
{
    /* Open I2C2 and set clock to 100k */
    I2C_Open(I2C2, 100000);

    /* Get I2C2 Bus Clock */
    printf("I2C clock %d Hz\n", I2C_GetBusClockFreq(I2C2));
}



/* Model's parameters is exposed by the following functions */
/**
 * @brief   Gets the Threshold to distinguish
 * @return  a float
 **/
const float GetThreshold();

const float GetMaxValTrain();

const float GetMinValTrain();

const uint8_t GetFrameLenSample();


/*---------------------------------------------------------------------------------------------------------*/
/*  Main Function                                                                                          */
/*---------------------------------------------------------------------------------------------------------*/
int32_t main(void)
{
    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Init System, peripheral clock and multi-function I/O */
    SYS_Init();

    /* Init UART to 115200-8n1 for print message */
    UART_Open(UART0, 115200);

    printf("+-----------------------------------------------------------------------+\n");
    printf("|                          tflu_gearbox_anomaly                         |\n");
    printf("+-----------------------------------------------------------------------+\n");


    /* Init I2C2 to access codec */
    I2C2_Init();

    /* Set PD3 low to enable phone jack on NuMaker board. */
    SYS->GPD_MFP0 &= ~(SYS_GPD_MFP0_PD3MFP_Msk);
    GPIO_SetMode(PD, BIT3, GPIO_MODE_OUTPUT);
    PD3 = 0;

    /* Select source from HXT(12MHz) */
    CLK_SetModuleClock(I2S0_MODULE, CLK_CLKSEL3_I2S0SEL_HXT, 0);

    /* Set MCLK and enable MCLK */
    I2S_EnableMCLK(I2S0, 12000000);

    /* Enable I2S Rx function */
    I2S_ENABLE_RXDMA(I2S0);
    I2S_ENABLE_RX(I2S0);

    /* Enable I2S Tx function */
    I2S_ENABLE_TXDMA(I2S0);
    I2S_ENABLE_TX(I2S0);

    printf("\nThis sample code run gearbox anomaly detaction\n");


#ifdef RAW_DATA
    uint16_t FeatureElements = 4; //a1~a4 sensors
    const uint16_t win_size = 300;
#else
    const uint32_t FeatureElements = 16; //4 sensors * 4 features
#endif

    const uint16_t TestNum = sizeof(y_test) / sizeof(uint8_t);
    static float Threshold = GetThreshold();


    uint16_t correct_acc = 0;
    const char outputClass[2][1] = {0, 1};

#ifdef RAW_DATA
    FeatureElements *= win_size;  // 1 input has win_size * featureNumber
#endif

    float X_Single_Input[FeatureElements];
    memcpy(X_Single_Input, &X_test[0], FeatureElements * sizeof(float));



    MLPB mlpb(X_Single_Input, FeatureElements);


    uint8_t normal_or_anomaly; // record true or false of each time

    for (uint16_t i = 0; i < TestNum; i++)
    {

#ifdef RAW_DATA
        memcpy(X_Single_Input, &X_test[i * FeatureElements], FeatureElements * sizeof(float));
#else
        memcpy(X_Single_Input, &X_test[i * FeatureElements], FeatureElements * sizeof(float));
#endif



        //printf("XXXXX %d: %f\r\n",i, X_test[i][0]);

        mlpb.ExtractFeatures();

#ifdef RAW_DATA
        mlpb.CalculateGearBoxWindow(FeatureElements, win_size, GetMaxValTrain(), GetMinValTrain());
#endif

#ifdef DNN
        mlpb.Classify();
        normal_or_anomaly = mlpb.GetDnnResult();
#else
        mlpb.EncoderDecoder();
        normal_or_anomaly = mlpb.GetMaeResult(Threshold);
#endif

        //comment out this for clean debug window
        printf("%d: inference:%d, answer:%d\r\n", i, normal_or_anomaly, y_test[i]);

        if (normal_or_anomaly == y_test[i])
        {
            correct_acc++;
        }
    }

    printf("The total %d test data's accuracy is %f\r\n", TestNum, correct_acc / (float)TestNum);


    while (1)
    {

    }
}
