/*
    James William Fletcher (james@voxdsp.com)
        May 2022

    Info:
    
        Stripped back version of PoryDrive to
        remove any gfx rendering.

        For the purpose of creating datasets
        for neural networks as a multi-process
        model with file locking.

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/file.h>
#include <stdint.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/time.h>

#include <X11/Xlib.h>
#include <X11/extensions/Xrandr.h>

//#define uint GLushort
#define sint short
#define f32 float

#ifndef __x86_64__
    #define NOSSE
#endif

// uncommenting this define will enable the MMX random when using fRandFloat (it's a marginally slower)
#define SEIR_RAND

#include "../inc/vec.h"
#include "../inc/mat.h"

//*************************************
// globals
//*************************************

// game logic
double t = 0;   // time
double dt = 0;  // delta time
double timeout = 0; // timeout after

// render state matrices
mat projection;
mat view;
mat model;
mat modelview;
mat viewrot;

// game vars
#define NEWGAME_SEED 1337

// player vars
f32 pr; // rotation
f32 sr; // steering rotation
vec pp; // position
vec pv; // velocity
vec pd; // wheel direction
vec pbd;// body direction
f32 sp; // speed
uint mcp;// max collected porygon count
uint cp;// collected porygon count
uint cc;// collision count
double st=0; // start time
char tts[32];// time taken string

// ai/ml
uint auto_drive=0;
    f32 ad_min_dstep = 0.01f;
    f32 ad_max_dstep = 0.06f;
    f32 ad_min_speedswitch = 2.f;
    f32 ad_maxspeed_reductor = 0.5f;
uint neural_drive=0;
uint dataset_logger=0;

// porygon vars
vec zp; // position
vec zd; // direction
f32 zr; // rotation
f32 zs; // speed
double za;// alive state
f32 zt; // twitch radius

// configurable vars
f32 maxspeed = 0.0165f;
f32 acceleration = 0.0028f;
f32 inertia = 0.00022f;
f32 drag = 0.00038f;
f32 steeringspeed = 1.4f;
f32 steerinertia = 180.f;
f32 minsteer = 0.16f;
f32 maxsteer = 0.45f;
f32 steeringtransfer = 0.019f;
f32 steeringtransferinertia = 280.f;

char cname[256] = {0};

//*************************************
// utility functions
//*************************************
void timestamp(char* ts)
{
    const time_t tt = time(0);
    strftime(ts, 16, "%H:%M:%S", localtime(&tt));
}

void loadConfig(uint type)
{
    FILE* f = fopen("config.txt", "r");
    if(f)
    {
        sprintf(cname, "config.txt");
        
        if(type == 1)
        {
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] CONFIG: config.txt loaded.\n", strts);
        }
        else
            printf("\nDetected config.txt loading settings...\n");

        char line[256];
        while(fgets(line, 256, f) != NULL)
        {
            char set[64];
            memset(set, 0, 64);
            float val;
            
            if(sscanf(line, "%63s %f", set, &val) == 2)
            {
                if(type == 0)
                    printf("Setting Loaded: %s %g\n", set, val);

                if(strcmp(set, "maxspeed") == 0){maxspeed = val;}
                if(strcmp(set, "acceleration") == 0){acceleration = val;}
                if(strcmp(set, "inertia") == 0){inertia = val;}
                if(strcmp(set, "drag") == 0){drag = val;}
                if(strcmp(set, "steeringspeed") == 0){steeringspeed = val;}
                if(strcmp(set, "steerinertia") == 0){steerinertia = val;}
                if(strcmp(set, "minsteer") == 0){minsteer = val;}
                if(strcmp(set, "maxsteer") == 0){maxsteer = val;}
                if(strcmp(set, "steeringtransfer") == 0){steeringtransfer = val;}
                if(strcmp(set, "steeringtransferinertia") == 0){steeringtransferinertia = val;}
            }
        }
        fclose(f);
    }
    else
    {
        if(type == 1)
        {
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] CONFIG: No config.txt file detected.\n", strts);
        }
    }
}

static inline f32 fRandFloat(const float min, const float max)
{
    return min + randf() * (max-min); 
}

void timeTaken(uint ss)
{
    if(ss == 1)
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.2f Sec", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Min", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hr", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
    else
    {
        const double tt = t-st;
        if(tt < 60.0)
            sprintf(tts, "%.2f Seconds", tt);
        else if(tt < 3600.0)
            sprintf(tts, "%.2f Minutes", tt * 0.016666667);
        else if(tt < 216000.0)
            sprintf(tts, "%.2f Hours", tt * 0.000277778);
        else if(tt < 12960000.0)
            sprintf(tts, "%.2f Days", tt * 0.00000463);
    }
}

void configOriginal()
{
    maxspeed = 0.006f;
    acceleration = 0.001f;
    inertia = 0.0001f;
    drag = 0.00038f;
    steeringspeed = 1.2f;
    steerinertia = 233.f;
    minsteer = 0.1f;
    maxsteer = 0.7f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;

    char strts[16];
    timestamp(&strts[0]);
    sprintf(cname, "Original");
    printf("[%s] CONFIG: %s.\n", strts, cname);
}

void configScarlet()
{
    maxspeed = 0.0095f;
    acceleration = 0.0025f;
    inertia = 0.00015f;
    drag = 0.00038f;
    steeringspeed = 1.2f;
    steerinertia = 233.f;
    minsteer = 0.32f;
    maxsteer = 0.55f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;
    
    char strts[16];
    timestamp(&strts[0]);
    sprintf(cname, "Scarlet");
    printf("[%s] CONFIG: %s.\n", strts, cname);
}

void configScarletFast()
{
    maxspeed = 0.0165f;
    acceleration = 0.0028f;
    inertia = 0.00022f;
    drag = 0.00038f;
    steeringspeed = 1.4f;
    steerinertia = 180.f;
    minsteer = 0.16f;
    maxsteer = 0.3f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;
    
    char strts[16];
    timestamp(&strts[0]);
    sprintf(cname, "ScarletFast");
    printf("[%s] CONFIG: %s.\n", strts, cname);
}

void configHybrid()
{
    maxspeed = 0.0165f;
    acceleration = 0.0028f;
    inertia = 0.00022f;
    drag = 0.00038f;
    steeringspeed = 3.2f;
    steerinertia = 233.f;
    minsteer = 0.1f;
    maxsteer = 0.2f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;
    
    char strts[16];
    timestamp(&strts[0]);
    sprintf(cname, "Hybrid");
    printf("[%s] CONFIG: %s.\n", strts, cname);
}

float urandf()
{
    static const float FLOAT_UINT64_MAX = (float)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return (((float)s)+1e-7f) / FLOAT_UINT64_MAX;
}

float uRandFloat(const float min, const float max)
{
    return ( urandf() * (max-min) ) + min;
}

uint64_t urand()
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return s;
}

double glfwGetTime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return ((double)tv.tv_sec) + (((double)tv.tv_usec)/1000000.0);
}

int forceTrim(const char* file, const size_t trim)
{
    int f = open(file, O_WRONLY);
    if(f > -1)
    {
        const size_t len = lseek(f, 0, SEEK_END);

        uint c = 0;
        while(ftruncate(f, len-trim) == -1)
        {
            close(f);
            f = open(file, O_WRONLY);
            c++;
            if(c > 333)
                return -2;
        }

        close(f);
    }
    else
        return -1;
    return 0;
}

//*************************************
// render functions
//*************************************

void rCube(f32 x, f32 y)
{
    mIdent(&model);
    mTranslate(&model, x, y, 0.f);
    mMul(&modelview, &model, &view);

    // cube collisions
    const f32 dlap = vDistLa(zp, (vec){x, y, 0.f}); // porygon
    if(dlap < 0.15f)
    {
        vec nf;
        vSub(&nf, zp, (vec){x, y, 0.f});
        vNorm(&nf);
        vMulS(&nf, nf, 0.15f-dlap);
        vAdd(&zp, zp, nf);
    }

    //printf("pp: %f %f - %f\n", pp.x, pp.y, t);
    //printf("pv: %f %f - %f\n", pv.x, pv.y, t);

    // if car is moving compute collisions
    if(sp > inertia || sp < -inertia)
    {
        // front collision cube point
        vec cp1 = pp;
        vec cd1 = pbd;
        vMulS(&cd1, cd1, 0.0525f);
        vAdd(&cp1, cp1, cd1);

        // back collision cube point
        vec cp2 = pp;
        vec cd2 = pbd;
        vMulS(&cd2, cd2, -0.0525f);
        vAdd(&cp2, cp2, cd2);

        // do Axis-Aligned Cube collisions for points against rCube() being rendered
        const f32 dla1 = vDistLa(cp1, (vec){x, y, 0.f}); // front car
        const f32 dla0 = vDistLa(pp, (vec){x, y, 0.f}); // center car
        const f32 dla2 = vDistLa(cp2, (vec){x, y, 0.f}); // back car
        if(dla1 <= 0.097f)
        {
            vec nf;
            vSub(&nf, pp, (vec){x, y, 0.f});
            vNorm(&nf);
            vMulS(&nf, nf, 0.097f-dla1);
            vAdd(&pv, pv, nf);
        }
        else if(dla0 <= 0.097f)
        {
            vec nf;
            vSub(&nf, pp, (vec){x, y, 0.f});
            vNorm(&nf);
            vMulS(&nf, nf, 0.097f-dla0);
            vAdd(&pv, pv, nf);
        }
        else if(dla2 <= 0.097f)
        {
            vec nf;
            vSub(&nf, pp, (vec){x, y, 0.f});
            vNorm(&nf);
            vMulS(&nf, nf, 0.097f-dla2);
            vAdd(&pv, pv, nf);
        }
    }

    // check to see if cube needs to be blue
    const f32 dla = vDist(pp, (vec){x, y, 0.f}); // worth it to prevent the flicker

    static f32 colliding = 0.f;
    if(dla <= 0.13f)
    {
        if(colliding == 0.f)
        {
            colliding = x*y+x;
            cc++;

            // char strts[16];
            // timestamp(&strts[0]);
            // printf("[%s] Collisions: %u\n", strts, cc);
        }
    }
    else if(x*y+x == colliding)
    {
        colliding = 0.f;
    }
}

void rPorygon(f32 x, f32 y, f32 r)
{
    mIdent(&model);
    mTranslate(&model, x, y, 0.f);
    mRotZ(&model, r);

    if(za != 0.0)
        mScale(&model, 1.f, 1.f, 0.1f);

    mMul(&modelview, &model, &view);

    // returns direction
    mGetDirY(&zd, model);
    vInv(&zd);
}

void rCar(f32 x, f32 y, f32 z, f32 rx)
{
    // wheel spin speed
    static f32 wr = 0.f;
    const f32 speed = sp * 33.f;
    if(sp > inertia || sp < -inertia)
        wr += speed;

    // wheel; front left
    mIdent(&model);
    mTranslate(&model, x, y, z);
    mRotZ(&model, -rx);
    mTranslate(&model, 0.026343f, -0.054417f, 0.012185f);
    mRotZ(&model, sr);

    // returns direction
    mGetDirY(&pd, model);
    vInv(&pd);

    // body & window matrix

    mIdent(&model);
    mTranslate(&model, x, y, z);
    mRotZ(&model, -rx);

    // returns direction
    mGetDirY(&pbd, model);
    vInv(&pbd);
}

//*************************************
// game functions
//*************************************

void newGame(unsigned int seed)
{
    srand(urand());
    srandf(urand());
    
    pp = (vec){0.f, 0.f, 0.f};
    pv = (vec){0.f, 0.f, 0.f};
    pd = (vec){0.f, 0.f, 0.f};

    st = 0;

    cp = 0;
    cc = 0;
    pr = 0.f;
    sr = 0.f;
    sp = 0.f;

    zp = (vec){uRandFloat(-18.f, 18.f), uRandFloat(-18.f, 18.f), 0.f};
    zs = 0.3f;
    za = 0.0;
    zt = 8.f;
}

void randAutoDrive()
{
    ad_min_dstep = uRandFloat(0.01f, 0.03f);
    ad_max_dstep = uRandFloat(0.03f, 0.09f);
    ad_min_speedswitch = uRandFloat(2.f, 4.f);
    ad_maxspeed_reductor = uRandFloat(0.1f, 0.5f);
}

void randGame()
{
    const uint seed = urand();
    newGame(seed);

    zp = (vec){uRandFloat(-18.f, 18.f), uRandFloat(-18.f, 18.f), 0.f};
    zs = uRandFloat(0.3f, 1.f);
    zt = uRandFloat(8.f, 16.f);
    za = 0.0;

    // randAutoDrive();

    auto_drive = 1;
    dataset_logger = 1;

    char strts[16];
    timestamp(&strts[0]);
    printf("\n[%s] Rand Game Start [%u], DATASET LOGGER & AUTO DRIVE ON.\n", strts, seed);
}

//*************************************
// update & render
//*************************************
void main_loop()
{
//*************************************
// time delta for interpolation
//*************************************
    // static double lt = 0;
    // dt = t-lt;
    // lt = t;

//*************************************
// update stats
//*************************************
    // static double ltut = 3.0;
    // if(t > ltut)
    // {
    //     timeTaken(1);
    //     char title[256];
    //     const f32 dsp = fabsf(sp*(1.f/maxspeed)*130.f);
    //     printf("| %s | Speed %.f MPH | Porygon %u | %s\n", tts, dsp, cp, cname);
    //     ltut = t + 1.0;
    // }

//*************************************
// auto drive
//*************************************
    f32 tr = maxsteer * ((maxspeed-sp) * steerinertia);
    if(tr < minsteer){tr = minsteer;}
    
    // side winder 1
    /*
        vec lad = pp;
        vSub(&lad, lad, zp);
        vNorm(&lad);
        const f32 as = fabsf(vDot(pbd, lad)+1.f) * 0.5f;
        sr = tr * as;
        const f32 d = vDist(pp, zp);
        if(d < 2.f)
            sp = maxspeed * (d*0.5f)+0.003f;
        else
            sp = maxspeed;
    */

    // side winder 2
    if(auto_drive == 1) // stochastic state machine "ai"
    {
        vec lad = pp;
        vSub(&lad, lad, zp);
        vNorm(&lad);
        const f32 as = fabsf(vDot(pbd, lad)+1.f) * 0.5f;
        static f32 ld = 0.f, td = 1.f;
        const f32 d = vDist(pp, zp);
        f32 ds = d * 0.01f;
        if(ds < ad_min_dstep){ds = ad_min_dstep;}
        else if(ds > ad_max_dstep){ds = ad_max_dstep;}
        if(fabsf(ld-d) > ds && ld < d){td *= -1.f;}
        ld = d;
        sr = (tr * as) * td;
        if(d < ad_min_speedswitch)
            sp = maxspeed * (d*ad_maxspeed_reductor)+0.003f;
        else
            sp = maxspeed;
    }

    // neural net
    if(neural_drive == 1) // Feed-Forward Neural Network (FNN)
    {
        vec lad = pp;
        vSub(&lad, lad, zp);
        vNorm(&lad);
        const f32 angle = vDot(pbd, lad);
        const f32 dist = vDist(pp, zp);

        const float input[6] = {pbd.x, pbd.y, lad.x, lad.y, angle, dist};

        // write input to file
        FILE *f = fopen("/dev/shm/porydrive_input.dat", "wb");
        if(f != NULL)
        {
            const size_t wbs = 6 * sizeof(float);
            if(fwrite(input, 1, wbs, f) != wbs)
                printf("ERROR: neural write failed.\n");
            fclose(f);
        }

        // load last result
        float ret[2];
        f = fopen("/dev/shm/porydrive_r.dat", "rb");
        if(f != NULL)
        {
            if(fread(&ret, 2, sizeof(float), f) == sizeof(float))
            {
                // set new vars
                sr = ret[0];
                sp = ret[1];
            }
            fclose(f);
        }
    }
    
    // neural net dataset
    // input | output
    // body direction x&y, porygon direction x&y, angle between directions, distance between car and porygon | car wheel rotation, car speed
    if(dataset_logger == 1)
    {
        vec lad = pp;
        vSub(&lad, lad, zp);
        vNorm(&lad);
        const f32 angle = vDot(pbd, lad);
        const f32 dist = vDist(pp, zp);

        // input
        int eskip = 0;
        FILE* f = fopen("dataset_x.dat", "ab"); // append bytes
        if(f != NULL)
        {
            if(flock(fileno(f), LOCK_EX) == -1)
            {
                fclose(f);
                printf("File lock failed.\n");
                eskip = 1;
            }

            size_t r = 0;
            r += fwrite(&pbd.x, 1, sizeof(f32), f);
            r += fwrite(&pbd.y, 1, sizeof(f32), f);
            r += fwrite(&lad.x, 1, sizeof(f32), f);
            r += fwrite(&lad.y, 1, sizeof(f32), f);
            r += fwrite(&angle, 1, sizeof(f32), f);
            r += fwrite(&dist,  1, sizeof(f32), f);
            if(r != 24)
            {
                printf("Outch, just wrote corrupted bytes to dataset_x! (last %zu bytes).\n", r);
                if(forceTrim("dataset_x.dat", r) < 0)
                {
                    printf("Failed to repair X file. Exiting.\n");
                    rename("dataset_x.dat", "dataset_x.dat_dirty");
                    exit(0);
                }
                printf("Repaired.\n");
                eskip = 1;
            }

            if(flock(fileno(f), LOCK_UN) == -1)
            {
                fclose(f);
                printf("File unlock failed.\n");
            }

            fclose(f);
        }
        else
        {
            printf("Failed to fopen() X file. Skipping Y file.\n");
            eskip = 1; // failed to even open the first file... skip the second
        }

        // targets
        if(eskip == 0)
        {
            f = fopen("dataset_y.dat", "ab"); // append bytes
            if(f != NULL)
            {
                if(flock(fileno(f), LOCK_EX) == -1)
                {
                    fclose(f);
                    printf("File lock failed.\n");
                    if(forceTrim("dataset_x.dat", 24) < 0) // targets for this data failed to write, so wipe that
                    {
                        printf("Failed to revert X file. Exiting.\n");
                        rename("dataset_x.dat", "dataset_x.dat_dirty");
                        rename("dataset_y.dat", "dataset_y.dat_dirty");
                        exit(0);
                    }
                }

                size_t r = 0;
                r += fwrite(&sr,  1, sizeof(f32), f);
                r += fwrite(&sp,  1, sizeof(f32), f);
                if(r != 8)
                {
                    printf("Outch, just wrote corrupted bytes to dataset_y! (last %zu bytes).\n", r);
                    if(forceTrim("dataset_x.dat", 24) < 0) // targets for this data failed to write, so wipe that too
                    {
                        printf("Failed to revert X file. Exiting.\n");
                        rename("dataset_x.dat", "dataset_x.dat_dirty");
                        rename("dataset_y.dat", "dataset_y.dat_dirty");
                        exit(0);
                    }
                    if(forceTrim("dataset_y.dat", r) < 0)
                    {
                        printf("Failed to repair Y file. Exiting.\n");
                        rename("dataset_x.dat", "dataset_x.dat_dirty");
                        rename("dataset_y.dat", "dataset_y.dat_dirty");
                        exit(0);
                    }
                    printf("Repaired.\n");
                }

                if(flock(fileno(f), LOCK_UN) == -1)
                {
                    fclose(f);
                    printf("File unlock failed.\n");
                }

                fclose(f);
            }
            else
            {
                printf("Failed to fopen() Y file. Reverting X write.\n");
                if(forceTrim("dataset_x.dat", 24) < 0) // targets for this data failed to write, so wipe that
                {
                    printf("Failed to revert X file. Exiting.\n");
                    rename("dataset_x.dat", "dataset_x.dat_dirty");
                    rename("dataset_y.dat", "dataset_y.dat_dirty");
                    exit(0);
                }
            }
        }
    }

    // writing the targets to a seperate file makes file io errors more annoying to catch, but it does streamline
    // the process of loading that data into Keras.
    
    // dataset logging
    //printf("%f %f %f %f\n", (vAngle(pbd)*-1.f)+d2PI, vAngle(lad)+d2PI, vDot(pbd, lad)+1.f, vDist(pp, zp));
    //printf("%g %g %g %g :: %g\n", (vAngle(pbd)*-1.f)+d2PI, vAngle(lad)+d2PI, vDot(pbd, lad)+1.f, vDist(pp, zp), sr);
    //printf("%g %g %g %g :: %g :: %f\n", vAngle(pbd), vAngle(lad), vDot(pbd, lad), vDist(pp, zp), sr, sp);
    //printf("%g %g %g %g %g %g :: %g :: %f\n", pbd.x, pbd.y, lad.x, lad.y, vDot(pbd, lad), vDist(pp, zp), sr, sp);

//*************************************
// simulate car
//*************************************

    if(sp > 0.f)
        sp -= drag * dt;
    else
        sp += drag * dt;

    if(fabsf(sp) > maxspeed)
    {
        if(sp > 0.f)
            sp = maxspeed;
        else
            sp = -maxspeed;
    }

    if(sp > inertia || sp < -inertia)
    {
        vAdd(&pp, pp, pv);
        vMulS(&pv, pd, sp);
        pr -= sr * steeringtransfer * (sp*steeringtransferinertia);
    }

    if(pp.x > 17.5f){pp.x = 17.5f;}
    else if(pp.x < -17.5f){pp.x = -17.5f;}
    if(pp.y > 17.5f){pp.y = 17.5f;}
    else if(pp.y < -17.5f){pp.y = -17.5f;}

//*************************************
// simulate porygon
//*************************************

    if(za == 0.0)
    {
        vec inc;
        vMulS(&inc, zd, zs * dt);
        vAdd(&zp, zp, inc);
        zr += fRandFloat(-zt, zt) * dt;

        if(zp.x > 17.5f){zp.x = 17.5f; zr = fRandFloat(-PI, PI);}
        else if(zp.x < -17.5f){zp.x = -17.5f; zr = fRandFloat(-PI, PI);}
        if(zp.y > 17.5f){zp.y = 17.5f; zr = fRandFloat(-PI, PI);}
        else if(zp.y < -17.5f){zp.y = -17.5f; zr = fRandFloat(-PI, PI);}

        // front collision cube point
        vec cp1 = pp;
        vec cd1 = pbd;
        vMulS(&cd1, cd1, 0.0525f);
        vAdd(&cp1, cp1, cd1);

        // back collision cube point
        vec cp2 = pp;
        vec cd2 = pbd;
        vMulS(&cd2, cd2, -0.0525f);
        vAdd(&cp2, cp2, cd2);

        // do Axis-Aligned Cube collisions for both points against porygon
        const f32 dla1 = vDistLa(cp1, zp); // front car
        const f32 dla2 = vDistLa(cp2, zp); // back car
        if(dla1 < 0.04f || dla2 < 0.04f)
        {
            cp++;
            if(cp >= mcp)
            {
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] %u rounds completed, exiting...", strts, mcp);
                exit(0);
            }

            za = t+6.0;

            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] Porygon collected: %u, collisions: %u\n", strts, cp, cc);
            cc = 0;
        }
    }
    else if(t > za)
    {
        zp = (vec){uRandFloat(-18.f, 18.f), uRandFloat(-18.f, 18.f), 0.f};
        zs = uRandFloat(0.3f, 1.f);
        zt = uRandFloat(8.f, 16.f);
        za = 0.0;

        // randAutoDrive();
    }

//*************************************
// main render
//*************************************

    // render scene
    for(f32 i = -17.5f; i <= 18.f; i += 0.53f)
        for(f32 j = -17.5f; j <= 18.f; j += 0.53f)
            if((i < -0.1f || i > 0.1f) || (j < -0.1f || j > 0.1f)) // lol a branch for so little tut tut
                rCube(i, j);

    // render porygon
    rPorygon(zp.x, zp.y, zr);

    // render player
    rCar(pp.x, pp.y, pp.z, pr);
}

//*************************************
// Process Entry Point
//*************************************
int main(int argc, char** argv)
{
    // help
    printf("----\n");
    printf("PoryDriveCli\n");
    printf("James William Fletcher (james@voxdsp.com)\n");
    printf("This is the CLI trainer. No GFX. CPU Bound.\n");
    printf("----\n");

//*************************************
// execute update / render loop
//*************************************

    // how many rounds to run for
    mcp = 512;
    if(argc >= 2){mcp = atoi(argv[1]);}
    timeout = 0;
    if(argc >= 3){timeout = atof(argv[2]);}
    printf("Running for %u rounds with a timeout of %g seconds.\n----\n", mcp, timeout);

    // i did consider threading this, and having a log buffer
    // per thread that got aggregated by a logging thread
    // but it was just easier to multi-process it
    // and it's adequate. I've not witnessed the file locking
    // cause any impact on the CPS.

    // screen refresh rate
    const useconds_t wait = 1000000/144;

    // init
    configScarletFast();
    randGame();

    // reset
    const double st = glfwGetTime();
    t = glfwGetTime();
    dt = 1.0 / 144.0; // fixed timestep delta-time

    // "framerate" or Cycles Per Second (CPS) monitoring
    double ltt = t+32.0;
    uint fc = 0;
    double ltt2 = t+1.0;
    uint fc2 = 0;
    
    // event loop
    while(1)
    {
        usleep(wait);
        t = glfwGetTime();
        main_loop();

        // if CPS drops below 120, quit! bad data!!
        fc2++;
        if(t > ltt2)
        {
            if(fc2 < 120)
            {
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] CPS dropped to unacceptable level: %u\n", strts, fc2);
                exit(0);
            }
            fc2 = 0;
            ltt2 = t+1.0;
        }

        // user cycles per second counter
        fc++;
        if(t > ltt)
        {
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] CPS: %u\n", strts, fc/32);
            fc = 0;
            ltt = t+32.0;
        }

        if(timeout != 0 && t-st >= timeout)
            return 0;
    }

    // done
    return 0;
}
