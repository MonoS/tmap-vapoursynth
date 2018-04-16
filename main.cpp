#include <stdlib.h>
#include "VapourSynth.h"
#include "VSHelper.h"
#include <stdint.h>
#include <immintrin.h>
//#include "iaca/iacaMarks.h"

#define static_buf
//#undef __AVX__
#define __AVX__

#ifdef static_buf
#include <map>
#include <windows.h>
#include <string.h>

std::map<DWORD, float*> buffers;
#endif // static_buf

float const LDR_nits = 100.f;

typedef struct
{
    VSNodeRef *node;
    const VSVideoInfo *vi;

    float source_peak;
    float desat;

    bool lin;
    bool show_satmask;
    bool show_clipped;

    float exposure_bias;
    float tm;
    float w;
    float tm_ldr_value;
    float ldr_value_mult;
} tmData;


static void VS_CC tmInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi)
{
    tmData *d = (tmData *) * instanceData;
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef  *VS_CC tmGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    tmData *d = (tmData *) * instanceData;

    if (activationReason == arInitial)
    {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    }
    else if (activationReason == arAllFramesReady)
    {
        const VSFrameRef *src = vsapi->getFrameFilter(n, d->node, frameCtx);
		VSFrameRef *dst = vsapi->copyFrame(src, core);

		int height = vsapi->getFrameHeight(src, 0);
		int width  = vsapi->getFrameWidth(src, 0);
        int stride = vsapi->getStride(dst, 0) / sizeof(float);
        #ifdef static_buf
		DWORD tid = GetCurrentThreadId();

		if (buffers.find(tid) == buffers.end()) {
			buffers[tid] = (float*)calloc(3*width*height, sizeof(float));
		}

		float* buffer = buffers[tid];
		//memset(buffer, 0, 3*width*height*sizeof(float));
        #else
        float* buffer = (float*)calloc(3*width*height, sizeof(float));
        #endif
        float* tmLuma = buffer                       ;
        float* srLuma = buffer +    (width * height) ;
        float* mask   = buffer + (2*(width * height));

#ifdef __AVX__
        static __m256 exp_bias8          = _mm256_set1_ps(d->exposure_bias);
        static __m256 const const_0_15   = _mm256_set1_ps(0.15f);
        static __m256 const const_0_05   = _mm256_set1_ps(0.05f);
        static __m256 const const_0_06   = _mm256_set1_ps(0.06f);
        static __m256 const const_0_50   = _mm256_set1_ps(0.50f);
        static __m256 const const_0_004  = _mm256_set1_ps(0.004f);
        static __m256 const const_0_066  = _mm256_set1_ps(0.02 / 0.3);
        static __m256 const const_1      = _mm256_set1_ps(1.f);
        static __m256 const const_0      = _mm256_set1_ps(0.f);
        static __m256 w8                 = _mm256_set1_ps(d->w);
        static __m256 tm_ldr_val8        = _mm256_set1_ps(d->tm_ldr_value);
        static __m256 tm_ldr_mul8        = _mm256_set1_ps(d->ldr_value_mult);
        static __m256 tm_ldr_div8        = _mm256_div_ps(const_1, tm_ldr_mul8);
        static __m256 desat8             = _mm256_set1_ps(d->desat);
        static __m256 desat_inv          = _mm256_sub_ps(const_1, desat8);
#endif

        for (int iPlane = 0; iPlane < 3; iPlane++){
            float lumaMult = 0.f;
            switch(iPlane){
                case 0:{
                    lumaMult = 0.2627;
                    break;
                }
                case 1:{
                    lumaMult = 0.678;
                    break;
                }
                case 2:{
                    lumaMult = 0.0593;
                    break;
                }
            }
            float* srcPlane = (float*)vsapi->getReadPtr(src, iPlane);
#ifndef __AVX__
            for(int j = 0; j < height; j++){
                float* srcPtr = srcPlane + (stride * j);
                float* slPtr  = srLuma   + (width  * j);
                float* tlPtr  = tmLuma   + (width  * j);
                for(float* end = srcPtr + width; srcPtr < end; srcPtr++, tlPtr++, slPtr++){
                    float val = *srcPtr;

                    if(iPlane == 0){
                        (*slPtr) = val * lumaMult;
                    }else{
                        (*slPtr) = (*slPtr) + val * lumaMult;
                    }

                    val = ((val * d->exposure_bias * (0.15 * val * d->exposure_bias + 0.05) + 0.004) / (val * d->exposure_bias * (0.15 * val * d->exposure_bias + 0.50) + 0.06) - 0.02 / 0.30) * d->w;

                    if (val < 0.f){
                        val = 0.f;
                    }else if (val > 1.f){
                        val = 1.f;
                    }

                    if(d->lin){
                        if(val < d->tm_ldr_value){
                            val = (*srcPtr) * d->ldr_value_mult;
                        }
                    }

                    (*tlPtr) = (*tlPtr) + val * lumaMult;
                }
            }
        }
#else
            __m256 lumaMult8 = _mm256_set1_ps(lumaMult);
            for(int j = 0; j < height; j++){
                float* srcPtr = srcPlane + (stride * j);
                float* slPtr  = srLuma   + (width  * j);
                float* tlPtr  = tmLuma   + (width  * j);
                for(float* end = srcPtr + width; srcPtr < end; srcPtr+=8, tlPtr+=8, slPtr+=8){
                    __m256 val   = _mm256_loadu_ps(srcPtr);
                    __m256 src   = val;
                    __m256 slVal = _mm256_loadu_ps(slPtr);
                    __m256 tlVal = _mm256_loadu_ps(tlPtr);

                    if(iPlane == 0){
                        _mm256_storeu_ps(slPtr, _mm256_mul_ps(val, lumaMult8));
                    }else{
                        _mm256_storeu_ps(slPtr, _mm256_fmadd_ps(val, lumaMult8, slVal));
                    }
                    __m256 val_exp = _mm256_mul_ps(val, exp_bias8);

                    //val = ((val * exposure_bias * (0.15 * val * exposure_bias + 0.05) + 0.004) / (val * exposure_bias * (0.15 * val * exposure_bias + 0.50) + 0.06) - 0.02 / 0.30) * w;
                    #if 1
                    val = _mm256_sub_ps(_mm256_div_ps(_mm256_fmadd_ps(val_exp, _mm256_fmadd_ps(val_exp, const_0_15, const_0_05), const_0_004),
                                                      _mm256_fmadd_ps(val_exp, _mm256_fmadd_ps(val_exp, const_0_15, const_0_50), const_0_06 )),
                                        const_0_066);
                    #else
                    val = _mm256_fmsub_ps(              _mm256_fmadd_ps(val_exp, _mm256_fmadd_ps(val_exp, const_0_15, const_0_05), const_0_004),
                                          _mm256_rcp_ps(_mm256_fmadd_ps(val_exp, _mm256_fmadd_ps(val_exp, const_0_15, const_0_50), const_0_06 )),
                                          const_0_066);
                    #endif
                    val = _mm256_mul_ps(val, w8);

                    val = _mm256_min_ps(_mm256_max_ps(val, const_0), const_1);

                    if(d->lin){
                        __m256 mask = _mm256_cmp_ps(val, tm_ldr_val8, 1);
                        __m256 res  = _mm256_mul_ps(src, tm_ldr_mul8);

                        val = _mm256_or_ps(_mm256_and_ps(mask, res), _mm256_andnot_ps(mask, val));
                    }

                    if(iPlane == 0){
                        _mm256_storeu_ps(tlPtr, _mm256_mul_ps(val, lumaMult8));
                    }else{
                        _mm256_storeu_ps(tlPtr, _mm256_fmadd_ps(val, lumaMult8, tlVal));
                    }
                }
            }
        }
#endif

        for(int j = 0; j < height; j++){
            float* mPtr  = mask   + (width * j);
            float* slPtr = srLuma + (width * j);
            float* tlPtr = tmLuma + (width * j);
#ifndef __AVX__
            for(float* end = mPtr + width; mPtr < end; mPtr++, slPtr++, tlPtr++){
                float val = ((*slPtr) * d->ldr_value_mult - d->tm_ldr_value) / d->ldr_value_mult;

                if(val < 0.f){
                    val = 0.f;
                }else if(val > 1.f){
                    val = 1.f;
                }

                (*mPtr)  = val;
                (*tlPtr) = (*tlPtr) / (*slPtr); //dentro tmLuma ora abbiamo scale
            }
#else
            for(float* end = mPtr + width; mPtr < end; mPtr+=8, slPtr+=8, tlPtr+=8){
                __m256 tlVal = _mm256_loadu_ps(tlPtr);
                __m256 slVal = _mm256_loadu_ps(slPtr);
                __m256 val   = _mm256_mul_ps(_mm256_fmsub_ps(slVal, tm_ldr_mul8, tm_ldr_val8), tm_ldr_div8);

                val = _mm256_min_ps(_mm256_max_ps(val, const_0), const_1);

                _mm256_storeu_ps(mPtr, val);
                _mm256_storeu_ps(tlPtr, _mm256_div_ps(tlVal, slVal)); //dentro tmLuma ora abbiamo scale
            }
#endif
        }

        for (int iPlane = 0; iPlane < 3; iPlane++){
            float* dstPlane = (float*)vsapi->getWritePtr(dst, iPlane);

            for(int j = 0; j < height; j++){
                float* dPtr  = dstPlane + (stride * j);
                float* slPtr = srLuma   + (width  * j);
                float* tlPtr = tmLuma   + (width  * j);
                float* mPtr  = mask     + (width  * j);
#ifndef __AVX__
                for(float* end = dPtr + width; dPtr < end; dPtr++, slPtr++, tlPtr++, mPtr++){
                    float val = (*dPtr);
                    float asat = (*slPtr) * d->desat + val * (1 - d->desat);

                    val = val + ((asat - val) * (*mPtr));

                    val = val * (*tlPtr);

                     if(val < 0.f){
                        val = 0.f;
                    }else if(val > 1.f){
                        val = 1.f;
                    }

                    (*dPtr) = val;
                }
#else
                for(float* end = dPtr + width; dPtr < end; dPtr+=8, slPtr+=8, tlPtr+=8, mPtr+=8){
                    __m256 val   = _mm256_loadu_ps(dPtr);
                    __m256 slVal = _mm256_loadu_ps(slPtr);
                    __m256 tlVal = _mm256_loadu_ps(tlPtr);
                    __m256 mVal  = _mm256_loadu_ps(mPtr);

                    __m256 asat = _mm256_fmadd_ps(slVal, desat8, _mm256_mul_ps(val, desat_inv));
                    val = _mm256_mul_ps(_mm256_fmadd_ps(_mm256_sub_ps(asat, val), mVal, val), tlVal);

                    val = _mm256_min_ps(_mm256_max_ps(val, const_0), const_1);

                    _mm256_storeu_ps(dPtr, val);
                }
#endif
            }
        }
        #ifndef static_buf
        free(buffer);
        #endif // static_buf
        vsapi->freeFrame(src);

        return dst;
    }

    return 0;
}

static void VS_CC tmFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    tmData *d = (tmData *)instanceData;
    vsapi->freeNode(d->node);
    #ifdef static_buf
    for (auto it = buffers.begin(); it != buffers.end(); ++it){
        free(it->second);
    }
    #endif
    free(d);
}

static void VS_CC tmCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi)
{
    tmData d;
    tmData *data;
    int err;

    d.node = vsapi->propGetNode(in, "clip", 0, 0);
    d.vi = vsapi->getVideoInfo(d.node);

    if (!isConstantFormat(d.vi) || d.vi->format->sampleType != stFloat || d.vi->format->bitsPerSample != 32)
    {
        vsapi->setError(out, "tmap: only constant format 32bit float input supported");
        vsapi->freeNode(d.node);
        return;
    }

	d.source_peak = vsapi->propGetFloat(in, "source_peak" , 0, &err);
	if(err){
        vsapi->setError(out, "tmap: source_peak is mandatory");
        vsapi->freeNode(d.node);
        return;
	}
	d.desat = vsapi->propGetFloat(in, "desat", 0, &err);
	if(err) d.desat = 0.5f;

	d.lin = !!vsapi->propGetInt(in, "lin", 0, &err);
	if(err) d.lin = true;

	d.show_satmask = !!vsapi->propGetInt(in, "show_satmask", 0, &err);
	if(err) d.show_satmask = false;
	d.show_clipped = !!vsapi->propGetInt(in, "show_clipped", 0, &err);
	if(err) d.show_clipped = false;

    //float ldr_value = 1.f / exposure_bias; //for example in linear light compressed (0-1 range ) hdr 1000 nits, 100nits becomes 0.1
    d.exposure_bias  = d.source_peak / LDR_nits;
    d.tm             = ((1.f*(0.15*1.f+0.10*0.50)+0.20*0.02) / (1.f*(0.15*1.f+0.50)+0.20*0.30)) - 0.02/0.30;
    d.w              = 1.f / (((d.exposure_bias*(0.15*d.exposure_bias+0.10*0.50)+0.20*0.02)/(d.exposure_bias*(0.15*d.exposure_bias+0.50)+0.20*0.30))-0.02/0.30);
    d.tm_ldr_value   = d.tm * d.w; //value of 100 nits after the tone mapping
    d.ldr_value_mult = d.tm_ldr_value/(1.f/d.exposure_bias); //0.1 (100nits) * ldr_value_mult=tm_ldr_value

    data = (tmData*) malloc(sizeof(d));
    *data = d;

    vsapi->createFilter(in, out, "tm", tmInit, tmGetFrame, tmFree, fmParallel, 0, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin)
{
    configFunc("com.monos.tmap", "tmap", "Hable Tonemapping", VAPOURSYNTH_API_VERSION, 1, plugin);
    registerFunc("tm", "clip:clip;"
                       "source_peak:float;"
                       "desat:float:opt;"
                       "lin:int:opt;"
                       "show_satmask:int:opt;"
                       "show_clipped:int:opt;",
                       tmCreate, 0, plugin);
}
