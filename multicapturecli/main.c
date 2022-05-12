/*
    James William Fletcher (james@voxdsp.com)
        May 2022

    Info:

        This version produces scored datasets.

        > distance from porygon at round start : 0 - 36
            = x * 0.027777778

        > speed of porygon : 0.3 - 1
            = x

        > twitch radius of porygon : 8 - 16
            = (x-8) * 0.125

        === average them all for a 0-1 score

        < round time taken (60 seconds max)
            = x * 0.016666667

        < round collisions (333 max)
            = x * 0.003003003
            
        === average them all and 1 - x for a 0-1 score (higher is better)
    
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
double t = 0; // time
f32 dt = 0;   // delta time
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

// logging score
#define XMAX 57024
float dataset_x[XMAX];
uint dxi = 0;
#define YMAX 19008
float dataset_y[YMAX];
uint dyi = 0;
f32 start_dist = 0.f;
double round_start_time = 0;
f32 round_score = 0.f;
f32 minscore = 0.f;

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

int forceTrim(const int f, const size_t trim)
{
    if(f > -1)
    {
        const size_t len = lseek(f, 0, SEEK_END);
        if(ftruncate(f, len-trim) == -1)
            return -1;
    }
    else
        return -1;
    return 0;
}

int forceTrimLock(const char* file, const size_t trim)
{
    int f = open(file, O_WRONLY);
    if(f > -1)
    {
        while(flock(f, LOCK_EX) == -1)
            usleep(1000);

        const size_t len = lseek(f, 0, SEEK_END);

        if(ftruncate(f, len-trim) == -1)
        {
            close(f);
            return -1;
        }

        while(flock(f, LOCK_UN) == -1)
            usleep(1000);

        close(f);
    }
    else
        return -1;
    return 0;
}

void writeWarning(const char* s)
{
    FILE* f = fopen("WARNING_FLAGGED_ERROR.TXT", "a"); // just make it long so that it is noticable
    if(f != NULL)
    {
        char strts[16];
        timestamp(&strts[0]);
        fprintf(f, "[%s] %s\n", strts, s);
        printf("[%s] %s\n", strts, s);
        fclose(f);
    }
}

//*************************************
// render functions
//*************************************

void rCube(f32 x, f32 y)
{
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

    // official colliding count
    const f32 dla = vDist(pp, (vec){x, y, 0.f});
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
    pbd = (vec){0.f, 0.f, 0.f};

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

    round_start_time = t;
    start_dist = vDist(pp, zp);

    // randAutoDrive();

    auto_drive = 1;
    dataset_logger = 1;

    char strts[16];
    timestamp(&strts[0]);
    printf("\n[%s] Rand Game Start [%u], DATASET LOGGER & AUTO DRIVE ON.\n", strts, seed);
}

static inline uint isnorm(const f32 f)
{
    if(isnormal(f) == 1 || f == 0.f)
        return 1;
    return 0;
}

//*************************************
// update & render
//*************************************
void main_loop()
{
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
    // if(neural_drive == 1) // Feed-Forward Neural Network (FNN)
    // {
    //     vec lad = pp;
    //     vSub(&lad, lad, zp);
    //     vNorm(&lad);
    //     const f32 angle = vDot(pbd, lad);
    //     const f32 dist = vDist(pp, zp);

    //     const float input[6] = {pbd.x, pbd.y, lad.x, lad.y, angle, dist};

    //     // write input to file
    //     FILE *f = fopen("/dev/shm/porydrive_input.dat", "wb");
    //     if(f != NULL)
    //     {
    //         const size_t wbs = 6 * sizeof(float);
    //         if(fwrite(input, 1, wbs, f) != wbs)
    //             printf("ERROR: neural write failed.\n");
    //         fclose(f);
    //     }

    //     // load last result
    //     float ret[2];
    //     f = fopen("/dev/shm/porydrive_r.dat", "rb");
    //     if(f != NULL)
    //     {
    //         if(fread(&ret, 2, sizeof(float), f) == sizeof(float))
    //         {
    //             if(isnorm(ret[0]) == 1 && isnorm(ret[1]) == 1)
    //             {
    //                 // set new vars
    //                 sr = ret[0];
    //                 sp = ret[1];
    //             }
    //         }
    //         fclose(f);
    //     }
    // }


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

    // new round if timelimit exceeded
    const double roundtime = t-round_start_time;
    if(roundtime >= 60.0)
    {
        zp = (vec){uRandFloat(-18.f, 18.f), uRandFloat(-18.f, 18.f), 0.f};
        zs = uRandFloat(0.3f, 1.f);
        zt = uRandFloat(8.f, 16.f);
        za = 0.0;

        start_dist = vDist(pp, zp);
        round_start_time = t;

        dxi = 0, dyi = 0;
        round_score = 0.f;

        char strts[16];
        timestamp(&strts[0]);
        printf("[%s] Round took too long, starting new round.\n", strts);
        return;
    }

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
            if(cc <= 333 && roundtime <= 60.0)
            {
                const f32 score_startdist = start_dist*0.027777778f;
                const f32 score_poryspeed = zs;
                const f32 score_porytwitch= (zt-8.f) * 0.125f;
                const f32 score_timetaken = 1.f-(f32)(roundtime * 0.003003003);
                const f32 score_collisions= 1.f-(((f32)cc)*0.003003003f);
                round_score = (score_startdist + score_poryspeed + score_porytwitch + score_timetaken + score_collisions) / 5.f;
                printf("[%s] %g %g %g %g %g : %g\n", strts, score_startdist, score_poryspeed, score_porytwitch, score_timetaken, score_collisions, round_score);
            }
            else
            {
                round_score = 0.f;
                printf("[%s] This round did not qualify for logging. %g Round Time.\n", strts, roundtime);
            }
            cc = 0;
        }
    }
    else if(t > za)
    {
        zp = (vec){uRandFloat(-18.f, 18.f), uRandFloat(-18.f, 18.f), 0.f};
        zs = uRandFloat(0.3f, 1.f);
        zt = uRandFloat(8.f, 16.f);
        za = 0.0;

        start_dist = vDist(pp, zp);
        round_start_time = t;

        // randAutoDrive();

        dxi = 0, dyi = 0;
        round_score = 0.f;
        return;
    }

//*************************************
// dataset logging
//*************************************

    // neural net dataset
    if(dataset_logger == 1)
    {
        vec lad = pp;
        vSub(&lad, lad, zp);
        vNorm(&lad);
        const f32 angle = vDot(pbd, lad);
        const f32 dist = vDist(pp, zp);

        uint fail = 0;
        if(isnorm(pbd.x) == 0){fail++;}
        if(isnorm(pbd.y) == 0){fail++;}
        if(isnorm(lad.x) == 0){fail++;}
        if(isnorm(lad.y) == 0){fail++;}
        if(isnorm(angle) == 0){fail++;}
        if(isnorm(dist) == 0){fail++;}
        if(isnorm(sr) == 0){fail++;}
        if(isnorm(sp) == 0){fail++;}

        if(dxi >= XMAX-1 || dyi >= YMAX-1)
        {
            fail = 1;
            printf("Dataset log buffers are full, this should never happen.\n");
        }

        if(fail == 0)
        {
            // log x
            dataset_x[dxi++] = pbd.x;
            dataset_x[dxi++] = pbd.y;
            dataset_x[dxi++] = lad.x;
            dataset_x[dxi++] = lad.y;
            dataset_x[dxi++] = angle;
            dataset_x[dxi++] = dist;

            // log y
            dataset_y[dyi++] = sr;
            dataset_y[dyi++] = sp;
        }

        // write log buffer to file
        if(round_score >= minscore && dxi > 0 && dyi > 0)
        {
            int eskip = 0;

            char fnbx[32];
            sprintf(fnbx, "%.1f_x.dat", round_score);
            char fnby[32];
            sprintf(fnby, "%.1f_y.dat", round_score);

            int f = open(fnbx, O_APPEND | O_CREAT | O_WRONLY, S_IRWXU);
            if(f > -1)
            {
                if(flock(f, LOCK_EX) == -1)
                    usleep(1000);

                const size_t dxis = dxi*sizeof(f32);
                const ssize_t wb = write(f, &dataset_x[0], dxis);

                // this is very rare but if it fails... well.. we have a log
                if(wb != dxis)
                {
                    char emsg[256];
                    sprintf(emsg, "Just wrote corrupted bytes to %s! (last %zu bytes).", fnbx, wb);
                    writeWarning(emsg);
                    if(forceTrim(f, wb) < 0) // revert append to X dataset
                    {
                        writeWarning("Failed to revert X file write error. Exiting.");
                        exit(0); // locks, file handles, all cleaned automatically
                    }
                    writeWarning("Repaired.");
                    eskip = 1;
                }

                if(flock(f, LOCK_UN) == -1)
                    usleep(1000);

                close(f);
            }
            else
            {
                writeWarning("Failed to open X file. Skipping Y file.");
                eskip = 1;
            }

            if(eskip == 0)
            {
                f = open(fnby, O_APPEND | O_CREAT | O_WRONLY, S_IRWXU);
                if(f > -1)
                {
                    if(flock(f, LOCK_EX) == -1)
                        usleep(1000);

                    const size_t dyis = dyi*sizeof(f32);
                    const ssize_t wb = write(f, &dataset_y[0], dyis);

                    // this is very rare but if it fails... well.. we have a log
                    if(wb != dyis)
                    {
                        char emsg[256];
                        sprintf(emsg, "Just wrote corrupted bytes to %s! (last %zu bytes).", fnby, wb);
                        writeWarning(emsg);
                        if(forceTrimLock(fnbx, 24) < 0) // revert append to X dataset
                        {
                            writeWarning("Failed to revert X file write error. Exiting.");
                            exit(0); // locks, file handles, all cleaned automatically
                        }
                        if(forceTrim(f, wb) < 0) // clear corrupted write to Y dataset
                        {
                            writeWarning("Failed to revert Y file write error. Exiting.");
                            exit(0); // locks, file handles, all cleaned automatically
                        }
                        writeWarning("Repaired.");
                    }

                    if(flock(f, LOCK_UN) == -1)
                        usleep(1000);

                    close(f);
                }
                else
                {
                    // failed to open Y dataset for append so lets revert the last append to X dataset
                    writeWarning("Failed to open Y file.");
                    if(forceTrimLock(fnbx, 24) < 0)
                    {
                        writeWarning("Failed to revert X file after Y file open failed. Exiting.");
                        exit(0);
                    }
                }
            }

            dxi = 0, dyi = 0;
            round_score = 0.f;
        }
    }

    // writing the targets to a seperate file makes file io errors more annoying to catch, but it does streamline
    // the process of loading that data into Keras.

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
    printf("PoryDriveCli_scored\n");
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
    minscore = 0.01f;
    if(argc >= 4){minscore = atof(argv[3]);}
    if(minscore == 0.f){minscore = 0.01f;}
    printf("Running for %u rounds with a timeout of %g seconds.\n----\n", mcp, timeout);

    // i did consider threading this, and having a log buffer
    // per thread that got aggregated by a logging thread
    // but it was just easier to multi-process it and it's
    // adequate. I've not witnessed the file locking
    // cause any impact on the CPS assumably because the
    // writes are staggered by variable round times.

    // screen refresh rate
    const useconds_t wait_interval = 1000000/144;
    useconds_t wait = wait_interval;

    // init
    t = glfwGetTime();
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
        
        wait = wait_interval - (useconds_t)((glfwGetTime() - t) * 1000000.0);
        if(wait > wait_interval)
            wait = wait_interval;
    }

    // done
    return 0;
}
