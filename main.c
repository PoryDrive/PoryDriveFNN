/*
    James William Fletcher (github.com/mrbid)
        April 2022

    Info:
    
        PoryDrive2, a simple 3D driving game.

        This version is targeted at just Tensorflow Keras training.

        Logic is set to run at 144 times per second and the frames per second
        can be set to anything below that.


    Keyboard:

        ESCAPE = Focus/Unfocus Mouse Look
        F = FPS to console
        P = Player stats to console
        N = New Game
        W = Drive Forward
        A = Turn Left
        S = Drive Backward
        D = Turn Right
        Space = Breaks
        1-5 = Car Physics config selection (5 loads from file)


    Mouse:

        RIGHT CLICK/MOUSE4 = Zoom Snap Close/Ariel
        Scroll = Zoom in/out

    
    Notes:

        Although the 3D model is sourced ready made from a third-party I cannot
        stress enough how much work I had to put in to vertex colouring the segments,
        deleting non-visible faces and generally just cleaning up parts of the mesh
        before and after triangulating it. It was a few hours of work, more finicky
        than anything.

        It made me think that it would be nice to have a program that automatically
        deleted non-visible faces for me, for example, so a render pass from many
        angles across a sphere like peeling an orange top to bottom with a fixed
        increment step. Even still, you get a similar effect in Blender by selecting
        visible faces from multiple angles until you feel like you have all the
        important ones, then you can hide them and delete what was hiding behind them
        or just invert the selection and delete. But the mesh ideally still needs
        some minor cleaning after the fact.

        The windows are still double layered, it's probably not necesary but I left
        it that way.

        This game really is about that E34 car model from you know who automobile
        manafacturer, eveything else is well, garnish.

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
#define sint GLshort
#define f32 GLfloat

#include "inc/gl.h"
#define GLFW_INCLUDE_NONE
#include "inc/glfw3.h"

#ifndef __x86_64__
    #define NOSSE
#endif

// uncommenting this define will enable the MMX random when using fRandFloat (it's a marginally slower)
#define SEIR_RAND

#include "inc/esAux2.h"

#include "inc/res.h"
#include "assets/purplecube.h"
#include "assets/porygon.h"
#include "assets/dna.h"
#include "assets/body.h"
#include "assets/windows.h"
#include "assets/wheel.h"

//*************************************
// globals
//*************************************
GLFWwindow* window;
uint winw = 1024;
uint winh = 768;
double t = 0;   // time
f32 dt = 0;     // delta time
double fc = 0;  // frame count
double lfct = 0;// last frame count time
double lc = 0;  // logic count
double llct = 0;// last logic count time
f32 aspect;
double x,y,lx,ly;
double rww, ww, rwh, wh, ww2, wh2;
double uw, uh, uw2, uh2; // normalised pixel dpi

// render state id's
GLint projection_id;
GLint modelview_id;
GLint position_id;
GLint lightpos_id;
GLint solidcolor_id;
GLint color_id;
GLint opacity_id;
GLint normal_id;

// render state matrices
mat projection;
mat view;
mat model;
mat modelview;

// render state inputs
vec lightpos = {0.f, 0.f, 0.f};

// models
sint bindstate = -1;
sint bindstate2 = -1;
uint keystate[6] = {0};
ESModel mdlPurpleCube;
GLuint mdlBlueCubeColors;
ESModel mdlPorygon;
ESModel mdlDNA;
ESModel mdlBody;
GLuint mdlBodyColors2;
ESModel mdlWindows;
ESModel mdlWheel;

// game vars
#define FAR_DISTANCE 1000.f
#define NEWGAME_SEED 1337
uint RENDER_PASS = 0;
double st=0; // start time
char tts[32];// time taken string

// camera vars
uint focus_cursor = 1;
double sens = 0.001f;
f32 xrot = 2.6f;
f32 yrot = 1.f;
f32 zoom = -0.3f;

// player vars
f32 pr; // rotation
f32 sr; // steering rotation
vec pp; // position
vec pv; // velocity
vec pd; // wheel direction
vec pbd;// body direction
f32 sp; // speed
uint cp;// collected porygon count
uint cc;// collision count
f32 pc = 0.f;// is player colliding

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
f32 maxsteer = 0.3f;
f32 steering_deadzone = 0.013f;
f32 steeringtransfer = 0.023f;
f32 steeringtransferinertia = 280.f;
f32 suspension_pitch = 3.f;
f32 suspension_pitch_limit = 0.03f;
f32 suspension_roll = 30.f;
f32 suspension_roll_limit = 9.f;
uint sticky_collisions = 0;

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

                // car physics
                if(strcmp(set, "maxspeed") == 0){maxspeed = val;}
                if(strcmp(set, "acceleration") == 0){acceleration = val;}
                if(strcmp(set, "inertia") == 0){inertia = val;}
                if(strcmp(set, "drag") == 0){drag = val;}
                if(strcmp(set, "steeringspeed") == 0){steeringspeed = val;}
                if(strcmp(set, "steerinertia") == 0){steerinertia = val;}
                if(strcmp(set, "minsteer") == 0){minsteer = val;}
                if(strcmp(set, "maxsteer") == 0){maxsteer = val;}
                if(strcmp(set, "steering_deadzone") == 0){steering_deadzone = val;}
                if(strcmp(set, "steeringtransfer") == 0){steeringtransfer = val;}
                if(strcmp(set, "steeringtransferinertia") == 0){steeringtransferinertia = val;}
                if(strcmp(set, "suspension_pitch") == 0){suspension_pitch = val;}
                if(strcmp(set, "suspension_pitch_limit") == 0){suspension_pitch_limit = val;}
                if(strcmp(set, "suspension_roll") == 0){suspension_roll = val;}
                if(strcmp(set, "suspension_roll_limit") == 0){suspension_roll_limit = val;}
                if(strcmp(set, "sticky_collisions") == 0){sticky_collisions = (uint)val;}

                // auto drive
                if(strcmp(set, "ad_min_dstep") == 0){ad_min_dstep = val;}
                if(strcmp(set, "ad_max_dstep") == 0){ad_max_dstep = val;}
                if(strcmp(set, "ad_min_speedswitch") == 0){ad_min_speedswitch = val;}
                if(strcmp(set, "ad_maxspeed_reductor") == 0){ad_maxspeed_reductor = val;}
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
            sprintf(tts, "%.0f Sec", tt);
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
            sprintf(tts, "%.0f Seconds", tt);
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
    steering_deadzone = 0.013f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;

    suspension_pitch = 3.f;
    suspension_pitch_limit = 0.03f;
    suspension_roll = 30.f;
    suspension_roll_limit = 9.f;

    sticky_collisions = 1;

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
    steering_deadzone = 0.013f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;

    suspension_pitch = 3.f;
    suspension_pitch_limit = 0.03f;
    suspension_roll = 30.f;
    suspension_roll_limit = 9.f;

    sticky_collisions = 0;
    
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
    steering_deadzone = 0.013f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;

    suspension_pitch = 3.f;
    suspension_pitch_limit = 0.03f;
    suspension_roll = 30.f;
    suspension_roll_limit = 9.f;

    sticky_collisions = 0;
    
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
    steering_deadzone = 0.013f;
    steeringtransfer = 0.023f;
    steeringtransferinertia = 280.f;

    suspension_pitch = 3.f;
    suspension_pitch_limit = 0.03f;
    suspension_roll = 30.f;
    suspension_roll_limit = 9.f;

    sticky_collisions = 0;
    
    char strts[16];
    timestamp(&strts[0]);
    sprintf(cname, "Hybrid");
    printf("[%s] CONFIG: %s.\n", strts, cname);
}

float urandf()
{
    static const float FLOAT_UINT64_MAX = 1.f/(float)UINT64_MAX;
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return ((float)s) * FLOAT_UINT64_MAX;
}

float uRandFloat(const float min, const float max)
{
    return urandf() * (max-min) + min;
}

uint64_t urand()
{
    int f = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    uint64_t s = 0;
    ssize_t result = read(f, &s, sizeof(uint64_t));
    close(f);
    return s;
}

uint64_t microtime()
{
    struct timeval tv;
    struct timezone tz;
    memset(&tz, 0, sizeof(struct timezone));
    gettimeofday(&tv, &tz);
    return 1000000 * tv.tv_sec + tv.tv_usec;
}

#define isnorm isnormal
// static inline uint isnorm(const f32 f)
// {
//     return isnormal(f);
// }

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

__attribute__((always_inline)) inline void modelBind(const ESModel* mdl) // C code reduction helper (more inline opcodes)
{
    glBindBuffer(GL_ARRAY_BUFFER, mdl->cid);
    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(color_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdl->vid);
    glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(position_id);

    glBindBuffer(GL_ARRAY_BUFFER, mdl->nid);
    glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(normal_id);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdl->iid);
}

void iterBody()
{
    // maybe change this a little bit for the body, otherwise its a duplicate function that
    // could just be simplified down to one.
    static const uint mi = body_numvert*3;
    static f32 cd = 1.f;
    for(uint i = 0; i < mi; i++)
    {
        body_colors2[i] += fRandFloat(0.1f, 0.6f) * cd;
        if(body_colors2[i] >= 1.f)
            cd = -1.f;
        else if(body_colors2[i] <= 0.f)
            cd = 1.f;
    }
    esRebind(GL_ARRAY_BUFFER, &mdlBodyColors2, body_colors2, sizeof(body_colors2), GL_STATIC_DRAW);
}

void iterDNA()
{
    // so that you don't get confused I am exploiting the
    // fact that when a colour channel goes into minus
    // figures very strange things begin to happen to the
    // gradient between triangle verticies.
    static const uint mi = dna_numvert*3;
    static f32 cd = 1.f;
    for(uint i = 0; i < mi; i++) // lavalamp it
    {
        dna_colors[i] += fRandFloat(0.1f, 0.6f) * cd;

        // and this piece of code prevents it looking like a random mess,
        // gives some structure. This is the lava lamper.
        if(dna_colors[i] >= 1.f)
            cd = -1.f;
        else if(dna_colors[i] <= 0.f)
            cd = 1.f;
    }
    esRebind(GL_ARRAY_BUFFER, &mdlDNA.cid, dna_colors, sizeof(dna_colors), GL_STATIC_DRAW);

    iterBody();
}

void rCube(f32 x, f32 y)
{
    if(RENDER_PASS == 1)
    {
        mIdent(&model);
        mTranslate(&model, x, y, 0.f);
        mMul(&modelview, &model, &view);

        glUniform1f(opacity_id, 1.0f);

        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        
        if(bindstate != 1)
        {
            glBindBuffer(GL_ARRAY_BUFFER, mdlPurpleCube.vid);
            glVertexAttribPointer(position_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(position_id);

            glBindBuffer(GL_ARRAY_BUFFER, mdlPurpleCube.nid);
            glVertexAttribPointer(normal_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(normal_id);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mdlPurpleCube.iid);

            bindstate = 1;
            bindstate2 = -1;
        }
    }

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
            if(sticky_collisions){sp *= 0.5f;}
        }
        else if(dla0 <= 0.097f)
        {
            vec nf;
            vSub(&nf, pp, (vec){x, y, 0.f});
            vNorm(&nf);
            vMulS(&nf, nf, 0.097f-dla0);
            vAdd(&pv, pv, nf);
            if(sticky_collisions){sp *= 0.5f;}
        }
        else if(dla2 <= 0.097f)
        {
            vec nf;
            vSub(&nf, pp, (vec){x, y, 0.f});
            vNorm(&nf);
            vMulS(&nf, nf, 0.097f-dla2);
            vAdd(&pv, pv, nf);
            if(sticky_collisions){sp *= 0.5f;}
        }
    }

    // check to see if cube needs to be blue
    const f32 dla = vDist(pp, (vec){x, y, 0.f}); // worth it to prevent the flicker

    // official colliding count
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

    if(RENDER_PASS == 1)
    {
        // player colliding
        if(dla <= 0.17f)
        {
            if(pc == 0.f)
            {
                pc = x*y+x;
                cc++;

                // char strts[16];
                // timestamp(&strts[0]);
                // printf("[%s] Collisions: %u\n", strts, cc);
            }
        }
        else if(x*y+x == pc)
        {
            pc = 0.f;
        }

        const uint collision = (dla < 0.17f || dlap < 0.16f);
        if(collision == 1 && bindstate2 <= 1)
        {
            glBindBuffer(GL_ARRAY_BUFFER, mdlBlueCubeColors);
            glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(color_id);
            bindstate2 = 2;
        }
        else if(collision == 0 && bindstate2 != 1)
        {
            glBindBuffer(GL_ARRAY_BUFFER, mdlPurpleCube.cid);
            glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
            glEnableVertexAttribArray(color_id);
            bindstate2 = 1;
        }

        glDrawElements(GL_TRIANGLES, purplecube_numind, GL_UNSIGNED_SHORT, 0);
    }
}

void rPorygon(f32 x, f32 y, f32 r)
{
    if(RENDER_PASS == 1)
    {
        bindstate = -1;

        mIdent(&model);
        mTranslate(&model, x, y, 0.f);
        mRotZ(&model, r);

        if(za != 0.0)
            mScale(&model, 1.f, 1.f, 0.1f);

        mMul(&modelview, &model, &view);

        // returns direction
        mGetDirY(&zd, model);
        vInv(&zd);

        if(za != 0.0)
            glUniform1f(opacity_id, (za-t)/6.0);
        else
            glUniform1f(opacity_id, 1.0f);

        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        modelBind(&mdlPorygon);

        if(za != 0.0)
            glEnable(GL_BLEND);
        glDrawElements(GL_TRIANGLES, porygon_numind, GL_UNSIGNED_SHORT, 0);
        if(za != 0.0)
            glDisable(GL_BLEND);
    }
    else
    {
        mIdent(&model);
        mTranslate(&model, x, y, 0.f);
        mRotZ(&model, r);

        // returns direction
        mGetDirY(&zd, model);
        vInv(&zd);
    }
}

void rDNA(f32 x, f32 y, f32 z)
{
    static f32 dr = 0.f;
    dr += 1.f * dt;

    if(RENDER_PASS == 1)
    {
        bindstate = -1;

        mIdent(&model);
        mTranslate(&model, x, y, z);
        mRotZ(&model, dr);
        mMul(&modelview, &model, &view);

        glUniform1f(opacity_id, 1.0f);
        
        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        modelBind(&mdlDNA);
        glDrawElements(GL_TRIANGLES, dna_numind, GL_UNSIGNED_SHORT, 0);
    }
}

void rCar(f32 x, f32 y, f32 z, f32 rx)
{
    if(RENDER_PASS == 1)
    {
        bindstate = -1;

        // opaque
        glUniform1f(opacity_id, 1.0f);

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

        mRotY(&model, -wr);
        mMul(&modelview, &model, &view);

        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        modelBind(&mdlWheel);
        glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

        // wheel; back left

        mIdent(&model);
        mTranslate(&model, x, y, z);
        mRotZ(&model, -rx);
        mTranslate(&model, 0.026343f, 0.045294f, 0.012185f);
        mRotY(&model, -wr);
        mMul(&modelview, &model, &view);

        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        modelBind(&mdlWheel);
        glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

        // wheel; front right

        mIdent(&model);
        mRotZ(&model, PI);
        mTranslate(&model, -x, -y, -z);
        mRotZ(&model, -rx);
        mTranslate(&model, 0.026343f, 0.054417f, 0.012185f);
        mRotZ(&model, sr);
        mRotY(&model, wr);
        mMul(&modelview, &model, &view);

        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        modelBind(&mdlWheel);
        glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

        // wheel; back right

        mIdent(&model);
        mRotZ(&model, PI);
        mTranslate(&model, -x, -y, -z);
        mRotZ(&model, -rx);
        mTranslate(&model, 0.026343f, -0.045294f, 0.012185f);
        mRotY(&model, wr);
        mMul(&modelview, &model, &view);

        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        modelBind(&mdlWheel);
        glDrawElements(GL_TRIANGLES, wheel_numind, GL_UNSIGNED_SHORT, 0);

        // body & window matrix

        mIdent(&model);
        mTranslate(&model, x, y, z);
        mRotZ(&model, -rx);

        // returns direction
        mGetDirY(&pbd, model);
        vInv(&pbd);

        f32 sy = sp*suspension_pitch; // lol speed based and not torque (it will do for now)
        if(sy > suspension_pitch_limit){sy = suspension_pitch_limit;}
        if(sy < -suspension_pitch_limit){sy = -suspension_pitch_limit;}
        mRotY(&model, sy);
        f32 sx = sr*suspension_roll*sp; // turning suspension
        if(sx > suspension_roll_limit){sx = suspension_roll_limit;}
        if(sx < -suspension_roll_limit){sx = -suspension_roll_limit;}
        mRotX(&model, sx);
        mMul(&modelview, &model, &view);

        glUniformMatrix4fv(modelview_id, 1, GL_FALSE, (f32*) &modelview.m[0][0]);
        
        // body

        modelBind(&mdlBody);

        if(pc == 0.f && cp > 0)
        {
            glDisable(GL_CULL_FACE);
            glDrawElements(GL_TRIANGLES, body_numind, GL_UNSIGNED_SHORT, 0);
            glEnable(GL_CULL_FACE);
        }
        else
        {
            if(cp == 0)
            {
                glDisable(GL_CULL_FACE);
                glDrawElements(GL_TRIANGLES, body_numind, GL_UNSIGNED_SHORT, 0);
                glEnable(GL_CULL_FACE);
            }
            else
            {
                if(cp > 1)
                {
                    glDisable(GL_CULL_FACE);
                    glDrawElements(GL_TRIANGLES, body_numind, GL_UNSIGNED_SHORT, 0);
                    glEnable(GL_CULL_FACE);

                    glBindBuffer(GL_ARRAY_BUFFER, mdlBodyColors2);
                    glVertexAttribPointer(color_id, 3, GL_FLOAT, GL_FALSE, 0, 0);
                    glEnableVertexAttribArray(color_id);
                }

                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glDrawElements(GL_TRIANGLES, body_numind, GL_UNSIGNED_SHORT, 0);
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            }
        }

        // transparent
        glUniform1f(opacity_id, 0.3f);

        // windows

        modelBind(&mdlWindows);

        glEnable(GL_BLEND);
        glDepthMask(GL_FALSE);
        glDrawElements(GL_TRIANGLES, windows_numind, GL_UNSIGNED_SHORT, 0);
        glDisable(GL_BLEND);
        glDepthMask(GL_TRUE);
    }
    else
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
}

//*************************************
// game functions
//*************************************

void newGame(unsigned int seed)
{
    srand(seed);
    srandf(seed);

    char strts[16];
    timestamp(&strts[0]);
    printf("[%s] Game Start [%u].\n", strts, seed);
    
    glfwSetWindowTitle(window, "PoryDrive");
    
    pp = (vec){0.f, 0.f, 0.f};
    pv = (vec){0.f, 0.f, 0.f};
    pd = (vec){0.f, 0.f, 0.f};

    st = t;

    cp = 0;
    cc = 0;
    pr = 0.f;
    sr = 0.f;
    sp = 0.f;

    zp = (vec){uRandFloat(-18.f, 18.f), uRandFloat(-18.f, 18.f), 0.f};
    //zp = (vec){0.f, 0.3f, 0.f};
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
    const int seed = urand();
    newGame(seed);

    zp = (vec){uRandFloat(-18.f, 18.f), uRandFloat(-18.f, 18.f), 0.f};
    zs = uRandFloat(0.3f, 1.f);
    zt = uRandFloat(8.f, 16.f);
    za = 0.0;

    // randAutoDrive();
    configScarletFast();

    auto_drive = 1;
    dataset_logger = 1;
    focus_cursor = 0;

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
// keystates
//*************************************
    f32 tr = maxsteer * ((maxspeed-sp) * steerinertia);
    if(tr < minsteer){tr = minsteer;}

    if(keystate[0] == 1)
    {
        sr -= steeringspeed * dt;
        if(sr < -tr){sr = -tr;}
    }

    if(keystate[1] == 1)
    {
        sr += steeringspeed * dt;
        if(sr > tr){sr = tr;}
    }

    if(keystate[0] == 0 && keystate[1] == 0)
    {
        if(sr > steering_deadzone)
            sr -= steeringspeed * dt;
        else if(sr < -steering_deadzone)
            sr += steeringspeed * dt;
        else
            sr = 0.f;
    }
    
    if(keystate[2] == 1)
    {
        vec inc;
        sp += acceleration * dt;
        vMulS(&inc, pd, acceleration * dt);
        vAdd(&pv, pv, inc);
    }

    if(keystate[3] == 1)
    {    
        vec inc;
        sp -= acceleration * dt;
        vMulS(&inc, pd, -acceleration * dt);
        vAdd(&pv, pv, inc);
    }

    if(keystate[4] == 1)
    {
        sp *= 0.99f * (1.f-dt);
    }

//*************************************
// update title bar stats
//*************************************
    static double ltut = 3.0;
    if(t > ltut)
    {
        timeTaken(1);
        char title[512];
        const f32 dsp = fabsf(sp*(1.f/maxspeed)*130.f);
        sprintf(title, "| %s | Speed %.f MPH | Porygon %u | %s", tts, dsp, cp, cname);
        glfwSetWindowTitle(window, title);
        ltut = t + 1.0;
    }

//*************************************
// auto drive
//*************************************
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
            if(fread(&ret, sizeof(float), 2, f) == 2)
            {
                // lock range
                // if(ret[0] < -1.f){ret[0] = -1.f;}
                // if(ret[0] > 1.f){ret[0] = 1.f;}
                // if(ret[1] < -1.f){ret[1] = -1.f;}
                // if(ret[1] > 1.f){ret[1] = 1.f;}
                // printf("%f %f %u %u\n", ret[0], ret[1], isnorm(ret[0]), isnorm(ret[1]));

                if(isnorm(ret[0]) == 1 && isnorm(ret[1]) == 1)
                {
                    // set new vars
                    sr = ret[0];
                    sp = ret[1];
                }

                // printf("%f %f\n", sp, sr);
            }
            fclose(f);
        }
    }
    
    // neural net dataset
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
            size_t r = 0;
            if(isnorm(pbd.x) == 1)
            {
                r += fwrite(&pbd.x, 1, sizeof(f32), f);
            }else{printf("pbd.x not isnorm()\n");}
            if(isnorm(pbd.y) == 1)
            {
                r += fwrite(&pbd.y, 1, sizeof(f32), f);
            }else{printf("pbd.y not isnorm()\n");}
            if(isnorm(lad.x) == 1)
            {
                r += fwrite(&lad.x, 1, sizeof(f32), f);
            }else{printf("lad.x not isnorm()\n");}
            if(isnorm(lad.y) == 1)
            {
                r += fwrite(&lad.y, 1, sizeof(f32), f);
            }else{printf("lad.y not isnorm()\n");}
            if(isnorm(angle) == 1)
            {
                r += fwrite(&angle, 1, sizeof(f32), f);
            }else{printf("angle not isnorm()\n");}
            if(isnorm(dist) == 1)
            {
                r += fwrite(&dist,  1, sizeof(f32), f);
            }else{printf("dist not isnorm()\n");}
            if(r != 24)
            {
                printf("Outch, just wrote corrupted bytes to dataset_x! (last %zu bytes).\n", r);
                if(forceTrim("dataset_x.dat", r) < 0)
                {
                    printf("Failed to repair X file. Exiting.\n");
                    fclose(f);
                    rename("dataset_x.dat", "dataset_x.dat_dirty");
                    exit(0);
                }
                printf("Repaired.\n");
                eskip = 1;
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
                size_t r = 0;
                if(isnorm(sr) == 1)
                {
                    r += fwrite(&sr,  1, sizeof(f32), f);
                }else{printf("sr not isnorm()\n");}
                if(isnorm(sp) == 1)
                {
                    r += fwrite(&sp,  1, sizeof(f32), f);
                }else{printf("sp not isnorm()\n");}
                if(r != 8)
                {
                    printf("Outch, just wrote corrupted bytes to dataset_y! (last %zu bytes).\n", r);
                    if(forceTrim("dataset_x.dat", 24) < 0) // targets for this data failed to write, so wipe that too
                    {
                        printf("Failed to revert X file. Exiting.\n");
                        fclose(f);
                        rename("dataset_x.dat", "dataset_x.dat_dirty");
                        rename("dataset_y.dat", "dataset_y.dat_dirty");
                        exit(0);
                    }
                    if(forceTrim("dataset_y.dat", r) < 0)
                    {
                        printf("Failed to repair Y file. Exiting.\n");
                        fclose(f);
                        rename("dataset_x.dat", "dataset_x.dat_dirty");
                        rename("dataset_y.dat", "dataset_y.dat_dirty");
                        exit(0);
                    }
                    printf("Repaired.\n");
                }
                fclose(f);
            }
            else
            {
                printf("Failed to fopen() Y file. Reverting X write.\n");
                if(forceTrim("dataset_x.dat", 24) < 0) // targets for this data failed to write, so wipe that too
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
            za = t+6.0;
            iterDNA();

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
// camera
//*************************************

    if(focus_cursor == 1)
    {
        glfwGetCursorPos(window, &x, &y);

        xrot += (ww2-x)*sens;
        yrot += (wh2-y)*sens;

        if(yrot > 1.5f)
            yrot = 1.5f;
        if(yrot < 0.5f)
            yrot = 0.5f;

        glfwSetCursorPos(window, ww2, wh2);
    }

    mIdent(&view);
    mTranslate(&view, 0.f, -0.033f, zoom);
    mRotate(&view, yrot, 1.f, 0.f, 0.f);
    mRotate(&view, xrot, 0.f, 0.f, 1.f);
    mTranslate(&view, -pp.x, -pp.y, -pp.z);

//*************************************
// begin render
//*************************************
    if(RENDER_PASS == 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

    // render dna
    rDNA(0.f, 0.f, 0.1f);

    // render player
    rCar(pp.x, pp.y, pp.z, pr);


//*************************************
// swap buffers / display render
//*************************************
    if(RENDER_PASS == 1)
        glfwSwapBuffers(window);
}

//*************************************
// Input Handelling
//*************************************
static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // control
    if(action == GLFW_PRESS)
    {
        if(key == GLFW_KEY_A){ keystate[0] = 1; keystate[1] = 0; }
        else if(key == GLFW_KEY_D){ keystate[1] = 1; keystate[0] = 0; }
        else if(key == GLFW_KEY_W){ keystate[2] = 1; }
        else if(key == GLFW_KEY_S){ keystate[3] = 1; }
        else if(key == GLFW_KEY_SPACE){ keystate[4] = 1; }

        // new game
        else if(key == GLFW_KEY_N)
        {
            // end
            timeTaken(0);
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] Game End.\n", strts);
            printf("[%s] Porygon Collected: %u\n", strts, cp);
            printf("[%s] Time-Taken: %s or %g Seconds\n\n", strts, tts, t-st);
            
            // new
            newGame(urand());
        }

        else if(key == GLFW_KEY_R)
        {
            printf("%g %g %g\n", dna_colors[0], dna_colors[1], dna_colors[2]);
            iterDNA();
            cp++;
        }

        // stats
        else if(key == GLFW_KEY_P)
        {
            char strts[16];
            timestamp(&strts[0]);
            printf("[%s] Porygon Collected: %u\n", strts, cp);
        }

        // toggle mouse focus
        if(key == GLFW_KEY_ESCAPE)
        {
            focus_cursor = 1 - focus_cursor;
            if(focus_cursor == 0)
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            else
                glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
            glfwSetCursorPos(window, ww2, wh2);
            glfwGetCursorPos(window, &ww2, &wh2);
        }

        // physics types
        else if(key == GLFW_KEY_1)
            configOriginal();
        else if(key == GLFW_KEY_2)
            configScarlet();
        else if(key == GLFW_KEY_3)
            configScarletFast();
        else if(key == GLFW_KEY_4)
            configHybrid();
        else if(key == GLFW_KEY_5)
            loadConfig(1);
        // else if(key == GLFW_KEY_LEFT_SHIFT)
        //     configOriginal();

        // show average fps
        else if(key == GLFW_KEY_F)
        {
            if(t-lfct > 2.0)
            {
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] FPS: %g\n", strts, fc/(t-lfct));
                printf("[%s] LPS: %g\n", strts, lc/(t-llct));
                lfct = t;
                fc = 0;
                llct = t;
                lc = 0;
            }
        }

        // toggle auto drive
        else if(key == GLFW_KEY_O)
        {
            auto_drive = 1 - auto_drive;
            if(auto_drive == 0)
            {
                sp = 0.f;
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Auto Drive: OFF\n", strts);
            }
            else
            {
                configScarletFast();
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Auto Drive: ON\n", strts);
            }
        }

        // toggle neural drive
        else if(key == GLFW_KEY_I)
        {
            neural_drive = 1 - neural_drive;
            if(neural_drive == 0)
            {
                sp = 0.f;
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Neural Drive: OFF\n", strts);
            }
            else
            {
                configScarletFast();
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Neural Drive: ON\n", strts);
            }
        }

        // toggle auto drive
        else if(key == GLFW_KEY_L)
        {
            dataset_logger = 1 - dataset_logger;

            if(dataset_logger == 1)
            {
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Dataset Logger: ON\n", strts);
            }
            else
            {
                FILE* f = fopen("dataset_size.txt", "w");
                if(f != NULL)
                {
                    struct stat st;
                    stat("dataset_y.dat", &st);
                    fprintf(f, "%ld", st.st_size/8);
                    fclose(f);
                }
                char strts[16];
                timestamp(&strts[0]);
                printf("[%s] Dataset Logger: OFF\n", strts);
            }
        }
    }
    else if(action == GLFW_RELEASE)
    {
        if(key == GLFW_KEY_A){ keystate[0] = 0; }
        else if(key == GLFW_KEY_D){ keystate[1] = 0; }
        else if(key == GLFW_KEY_W){ keystate[2] = 0; }
        else if(key == GLFW_KEY_S){ keystate[3] = 0; }
        else if(key == GLFW_KEY_SPACE){ keystate[4] = 0; }
        // else if(key == GLFW_KEY_LEFT_SHIFT)
        //     configScarletFast();
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    if(yoffset == -1)
        zoom += 0.06f * zoom;
    else if(yoffset == 1)
        zoom -= 0.06f * zoom;
    
    if(zoom > -0.11f){zoom = -0.11f;}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if(action == GLFW_PRESS)
    {
        if(button == GLFW_MOUSE_BUTTON_4 || button == GLFW_MOUSE_BUTTON_RIGHT)
        {
            if(zoom != -0.3f)
                zoom = -0.3f;
            else
                zoom = -3.3f;
        }
    }
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    winw = width;
    winh = height;

    glViewport(0, 0, winw, winh);
    aspect = (f32)winw / (f32)winh;
    ww = winw;
    wh = winh;
    rww = 1/ww;
    rwh = 1/wh;
    ww2 = ww/2;
    wh2 = wh/2;
    uw = (double)aspect / ww;
    uh = 1 / wh;
    uw2 = (double)aspect / ww2;
    uh2 = 1 / wh2;

    mIdent(&projection);
    mPerspective(&projection, 60.0f, aspect, 0.01f, FAR_DISTANCE*2.f); 
}

//*************************************
// Process Entry Point
//*************************************
int main(int argc, char** argv)
{
    // allow custom msaa level
    int msaa = 16;
    if(argc >= 2){msaa = atoi(argv[1]);}

    // allow framerate cap
    double maxfps = 144.0;
    if(argc >= 3){maxfps = atof(argv[2]);}

    // trigger special mode
    if(argc == 4)
        winw = 420, winh = 240, msaa = 0, maxfps = 10.0, RENDER_PASS = 0;

    // help
    printf("----\n");
    printf("PoryDrive\n");
    printf("----\n");
    printf("James William Fletcher (github.com/mrbid)\n");
    printf("----\n");
    printf("Three command line arguments, msaa 0-16, maxfps, data logging mode 0-1.\n");
    printf("e.g; ./porydrive 16 144 0\n");
    printf("----\n");
    printf("~ Keyboard Input:\n");
    printf("ESCAPE = Focus/Unfocus Mouse Look\n");
    printf("F = FPS to console\n");
    printf("P = Player stats to console\n");
    printf("O = Toggle auto drive\n");
    printf("I = Toggle neural drive\n");
    printf("N = New Game\n");
    printf("W = Drive Forward\n");
    printf("A = Turn Left\n");
    printf("S = Drive Backward\n");
    printf("D = Turn Right\n");
    printf("Space = Break\n");
    printf("1-5 = Car Physics config selection (5 loads from file)\n");
    printf("L = Toggle dataset logging\n");
    printf("----\n");
    printf("~ Mouse Input:\n");
    printf("RIGHT/MOUSE4 = Zoom Snap Close/Ariel\n");
    printf("Scroll = Zoom in/out\n");
    printf("----\n");
    printf("~ How to play:\n");
    printf("Drive around and \"collect\" Porygon, each time you collect a Porygon a new one will randomly spawn somewhere on the map. A Porygon colliding with a purple cube will cause it to light up blue, this can help you find them. Upon right clicking the mouse you will switch between Ariel and Close views, in the Ariel view it is easier to see which of the purple cubes that the Porygon is colliding with.\n");
    printf("----\n");
    printf("~ Create custom car physics:\n");
    printf("It is possible to tweak the car physics by creating a config.txt file in the exec/working directory of the game, here is an example of such config file with the default car phsyics variables.\n");
    printf("~ config.txt:\n");
    printf("maxspeed 0.0095\n");
    printf("acceleration 0.0025\n");
    printf("inertia 0.00015\n");
    printf("drag 0.00038\n");
    printf("steeringspeed 1.2\n");
    printf("steerinertia 233\n");
    printf("minsteer 0.32\n");
    printf("maxsteer 0.55\n");
    printf("steeringtransfer 0.023\n");
    printf("steeringtransferinertia 280\n");
    printf("----\n");

    // init glfw
    if(!glfwInit()){printf("glfwInit() failed.\n"); exit(EXIT_FAILURE);}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_SAMPLES, msaa);
    window = glfwCreateWindow(winw, winh, "PoryDrive", NULL, NULL);
    if(!window)
    {
        printf("glfwCreateWindow() failed.\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    const GLFWvidmode* desktop = glfwGetVideoMode(glfwGetPrimaryMonitor());
    glfwSetWindowPos(window, (desktop->width/2)-(winw/2), (desktop->height/2)-(winh/2)); // center window on desktop
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(0); // 0 for immediate updates, 1 for updates synchronized with the vertical retrace, -1 for adaptive vsync

    // set icon
    glfwSetWindowIcon(window, 1, &(GLFWimage){16, 16, (unsigned char*)&icon_image.pixel_data});

    // hide cursor
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

    // reset mouse for camera rotation
    glfwSetCursorPos(window, ww2, wh2);
    glfwGetCursorPos(window, &ww2, &wh2);

//*************************************
// projection
//*************************************

    window_size_callback(window, winw, winh);

//*************************************
// bind vertex and index buffers
//*************************************

    // ***** BIND BODY *****
    esBind(GL_ARRAY_BUFFER, &mdlBody.vid, body_vertices, sizeof(body_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBody.nid, body_normals, sizeof(body_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlBody.iid, body_indices, sizeof(body_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBody.cid, body_colors, sizeof(body_colors), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBodyColors2, body_colors2, sizeof(body_colors2), GL_STATIC_DRAW);

    // ***** BIND WINDOWS *****
    esBind(GL_ARRAY_BUFFER, &mdlWindows.vid, windows_vertices, sizeof(windows_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWindows.nid, windows_normals, sizeof(windows_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlWindows.iid, windows_indices, sizeof(windows_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWindows.cid, windows_colors, sizeof(windows_colors), GL_STATIC_DRAW);

    // ***** BIND WHEEL *****
    esBind(GL_ARRAY_BUFFER, &mdlWheel.vid, wheel_vertices, sizeof(wheel_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWheel.nid, wheel_normals, sizeof(wheel_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlWheel.iid, wheel_indices, sizeof(wheel_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlWheel.cid, wheel_colors, sizeof(wheel_colors), GL_STATIC_DRAW);

    // ***** BIND PURPLE CUBE *****
    esBind(GL_ARRAY_BUFFER, &mdlPurpleCube.vid, purplecube_vertices, sizeof(purplecube_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPurpleCube.nid, purplecube_normals, sizeof(purplecube_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlPurpleCube.iid, purplecube_indices, sizeof(purplecube_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPurpleCube.cid, purplecube_colors, sizeof(purplecube_colors), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlBlueCubeColors, bluecube_colors, sizeof(bluecube_colors), GL_STATIC_DRAW);

    // ***** BIND PORYGON *****
    esBind(GL_ARRAY_BUFFER, &mdlPorygon.vid, porygon_vertices, sizeof(porygon_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPorygon.nid, porygon_normals, sizeof(porygon_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlPorygon.iid, porygon_indices, sizeof(porygon_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlPorygon.cid, porygon_colors, sizeof(porygon_colors), GL_STATIC_DRAW);

    // ***** BIND DNA *****
    esBind(GL_ARRAY_BUFFER, &mdlDNA.vid, dna_vertices, sizeof(dna_vertices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlDNA.nid, dna_normals, sizeof(dna_normals), GL_STATIC_DRAW);
    esBind(GL_ELEMENT_ARRAY_BUFFER, &mdlDNA.iid, dna_indices, sizeof(dna_indices), GL_STATIC_DRAW);
    esBind(GL_ARRAY_BUFFER, &mdlDNA.cid, dna_colors, sizeof(dna_colors), GL_STATIC_DRAW);

//*************************************
// compile & link shader programs
//*************************************

    //makeAllShaders();
    makeLambert3();

//*************************************
// configure render options
//*************************************

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glBlendEquation(GL_FUNC_ADD);
    
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glClearColor(0.13, 0.13, 0.13, 0.0);

    shadeLambert3(&position_id, &projection_id, &modelview_id, &lightpos_id, &normal_id, &color_id, &opacity_id);
    glUniformMatrix4fv(projection_id, 1, GL_FALSE, (f32*) &projection.m[0][0]);
    glUniform3f(lightpos_id, lightpos.x, lightpos.y, lightpos.z);

//*************************************
// execute update / render loop
//*************************************

    // init
    configScarlet();
    loadConfig(0);
    if(argc == 4)
        randGame();
    else
        newGame(NEWGAME_SEED);

    // reset
    t = glfwGetTime();
    lfct = t;
    dt = 1.0 / 144.0; // fixed timestep delta-time
    
    // lps accurate event loop
    const double fps_limit = 1.0 / maxfps;
    double rlim = 0.0;
    const useconds_t wait_interval = 1000000 / 144;
    useconds_t wait = wait_interval;
    while(!glfwWindowShouldClose(window))
    {
        usleep(wait);
        t = glfwGetTime();
        glfwPollEvents();

        if(maxfps < 144.0)
        {
            if(t > rlim)
            {
                RENDER_PASS = 1;
                rlim = t + fps_limit; // should be doing this after main_loop() at the very least but it's not a big deal.
                fc++;
            }
            else
                RENDER_PASS = 0;
        }
        else
        {
            RENDER_PASS = 1;
            fc++;
        }

        main_loop();

        // accurate lps
        wait = wait_interval - (useconds_t)((glfwGetTime() - t) * 1000000.0);
        if(wait > wait_interval)
            wait = wait_interval;
        lc++;
    }

    // fps accurate event loop (not quite)
    // const double fps_limit = 1.0 / maxfps;
    // double rlim = 0.0;
    // const useconds_t wait_interval = 1000000 / 144;
    // useconds_t wait = wait_interval;
    // while(!glfwWindowShouldClose(window))
    // {
    //     usleep(wait);
    //     t = glfwGetTime();
    //     glfwPollEvents();

    //     //printf("%f %f %f\n", maxfps, t, rlim);
    //     if(maxfps < 144.0)
    //     {
    //         //printf("%f %f\n", t, rlim);
    //         if(t > rlim)
    //             RENDER_PASS = 1;
    //         else
    //             RENDER_PASS = 0;
    //     }
    //     else
    //         RENDER_PASS = 1;

    //     main_loop();

    //     // accurate fps
    //     if(RENDER_PASS == 1)
    //     {
    //         rlim = t + fps_limit - (glfwGetTime() - t);
    //         fc++;
    //     }

    //     lc++;
    // }

    // max fps accurate event loop
    // const double fps_limit = 1.0 / 144.0;
    // double fpsllt = 0.f;
    // while(!glfwWindowShouldClose(window))
    // {
    //     usleep(100);
    //     t = glfwGetTime();
    //     if(t > fpsllt)
    //     {
    //         glfwPollEvents();
    //         main_loop();
    //         fc++;
    //         fpsllt = t + fps_limit;
    //     }
    // }

    // end
    timeTaken(0);
    char strts[16];
    timestamp(&strts[0]);
    printf("[%s] Game End.\n", strts);
    printf("[%s] Porygon Collected: %u\n", strts, cp);
    printf("[%s] Time-Taken: %s or %g Seconds\n\n", strts, tts, t-st);

    // done
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
    return 0;
}
